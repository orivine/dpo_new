#!/usr/bin/env python
# coding=utf-8

import os
import json
import torch
import argparse
import logging
from tqdm import tqdm
import datasets
import math
import deepspeed
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="生成用于AlpacaEval评估的模型输出 (多GPU版本)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/dpo_new/outputs/llama-3-8b-instruct-simpo-0_9-weighted-new",
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="alpaca_eval_outputs_simpo_weighted_0.9.json",
        help="输出文件路径"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="生成的最大token数量"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,  # 在多卡环境下，总批量大小将是 batch_size * gpu数量
        help="每张GPU的批处理大小"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="生成温度"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
        help="top-p采样参数"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="处理的最大样本数，用于测试"
    )
    parser.add_argument(
        "--use_flash_attention_2",
        action="store_true",
        help="是否使用Flash Attention 2加速 (需要安装flash-attn库且硬件支持)"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="DeepSpeed所需的本地进程排名参数"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=3,
        help="DeepSpeed ZeRO优化阶段 (0, 1, 2, 3)"
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="DeepSpeed配置文件路径"
    )
    return parser.parse_args()

def format_prompt(instruction):
    """根据llama3模板格式化prompt"""
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def main():
    args = parse_args()
    
    # 初始化分布式环境
    deepspeed.init_distributed()
    
    # 获取世界大小和本地排名
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    is_main_process = local_rank == 0
    
    if is_main_process:
        logger.info(f"使用 {world_size} 个GPU进行生成")
        logger.info(f"正在加载模型: {args.model_path}")

    # 模型配置
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
    }

    # 如果使用Flash Attention 2，模型需要在GPU上初始化，所以不使用"auto"设备映射
    if not args.use_flash_attention_2:
        model_kwargs["device_map"] = "auto"

    if args.use_flash_attention_2:
        try:
            if torch.__version__ >= "2.0" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                 model_kwargs["attn_implementation"] = "flash_attention_2"
                 if is_main_process:
                     logger.info("启用 Flash Attention 2")
            else:
                 if is_main_process:
                     logger.warning("Flash Attention 2 不可用或不被支持，将使用默认注意力机制。")
                     logger.warning(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA capability: {torch.cuda.get_device_capability() if torch.cuda.is_available() else 'N/A'}")
        except ImportError:
            if is_main_process:
                logger.warning("未安装 flash-attn 库，无法使用 Flash Attention 2。请运行 'pip install flash-attn --no-build-isolation'")
        except Exception as e:
             if is_main_process:
                 logger.warning(f"尝试启用 Flash Attention 2 时出错: {e}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if is_main_process:
            logger.info(f"Tokenizer 没有 pad_token，已设置为 eos_token ({tokenizer.eos_token})")
    tokenizer.padding_side = "left"

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    
    # 先将模型移至GPU，然后再应用Flash Attention
    if args.use_flash_attention_2 and "attn_implementation" in model_kwargs:
        model = model.to(f"cuda:{local_rank}")
        
    model.eval()
    
    # 初始化DeepSpeed引擎
    if args.deepspeed_config:
        # 从文件读取DeepSpeed配置
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
    else:
        ds_config = {
            "fp16": {
                "enabled": model_kwargs["torch_dtype"] == torch.float16
            },
            "bf16": {
                "enabled": model_kwargs["torch_dtype"] == torch.bfloat16
            },
            "zero_optimization": {
                "stage": args.zero_stage,
                "offload_param": {
                    "device": "cpu" if args.zero_stage >= 2 else "none"
                }
            },
            "train_batch_size": args.batch_size * world_size,
            "train_micro_batch_size_per_gpu": args.batch_size,
        }
    
    ds_engine = deepspeed.initialize(
        model=model,
        config=ds_config
    )[0]
    model = ds_engine.module

    # 加载数据集
    if is_main_process:
        logger.info("正在下载AlpacaEval评估数据集")
    
    try:
        eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
        if is_main_process:
            logger.info(f"成功加载评估数据集，共 {len(eval_set)} 个样本")
    except Exception as e:
        if is_main_process:
            logger.error(f"下载数据集失败: {e}")
        raise

    if args.max_samples is not None:
        eval_set = eval_set.select(range(min(args.max_samples, len(eval_set))))
        if is_main_process:
            logger.info(f"限制处理样本数为 {len(eval_set)}")

    # 每个进程处理全部数据的一部分
    total_samples = len(eval_set)
    samples_per_gpu = math.ceil(total_samples / world_size)
    start_idx = local_rank * samples_per_gpu
    end_idx = min(start_idx + samples_per_gpu, total_samples)
    
    if is_main_process:
        logger.info(f"每个GPU处理约 {samples_per_gpu} 个样本")
    
    local_eval_set = eval_set.select(range(start_idx, end_idx))
    
    # 批处理生成
    local_results = []
    num_batches = math.ceil(len(local_eval_set) / args.batch_size)
    progress_bar = tqdm(range(num_batches), desc=f"GPU {local_rank} Processing", disable=not is_main_process)
    
    for i in progress_bar:
        start_index = i * args.batch_size
        end_index = min((i + 1) * args.batch_size, len(local_eval_set))
        batch_data = local_eval_set[start_index:end_index]
        
        batch_instructions = batch_data['instruction']
        batch_prompts = [format_prompt(instr) for instr in batch_instructions]
        
        # 批处理编码
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)
        
        # 生成回答
        with torch.no_grad():
            # 使用DeepSpeed引擎进行生成
            outputs = ds_engine.module.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 解码并提取助手回复
        input_token_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_token_len:]
        batch_outputs_decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # 保存批处理结果
        for j, generated_text in enumerate(batch_outputs_decoded):
            original_instruction = batch_instructions[j]
            result = {
                "instruction": original_instruction,
                "output": generated_text.strip(),
                "generator": os.path.basename(args.model_path),
                "global_idx": start_idx + start_index + j  # 添加全局索引用于合并结果
            }
            local_results.append(result)
        
        if is_main_process:
            progress_bar.set_postfix({"Batch size": len(batch_instructions)})
    
    # 收集来自所有GPU的结果
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_results)
    
    # 主进程合并结果并保存
    if is_main_process:
        # 展平所有结果并按全局索引排序
        merged_results = []
        for gpu_results in all_results:
            if gpu_results:  # 确保结果不为空
                merged_results.extend(gpu_results)
        
        # 按全局索引排序
        if merged_results:
            merged_results.sort(key=lambda x: x.get("global_idx", 0))
            # 移除临时索引
            for result in merged_results:
                if "global_idx" in result:
                    result.pop("global_idx")
        
        # 保存结果
        if merged_results:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(merged_results, f, ensure_ascii=False, indent=4)
            logger.info(f"所有样本处理完成 ({len(merged_results)} 个)，结果已保存至 {args.output_file}")
        else:
            logger.warning("没有生成任何结果！请检查数据集和生成过程。")
        
        logger.info("接下来可以使用alpaca_eval进行评估:")
        logger.info(f"export OPENAI_API_KEY=<your_api_key>")
        logger.info(f"alpaca_eval --model_outputs '{args.output_file}' --annotators_config 'weighted_alpaca_eval_gpt4_turbo'")

if __name__ == "__main__":
    main() 