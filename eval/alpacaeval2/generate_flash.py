#!/usr/bin/env python
# coding=utf-8

import os
import json
import torch
import argparse
import logging
from tqdm import tqdm
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import math

# ... (日志设置和参数解析保持不变) ...
# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="生成用于AlpacaEval评估的模型输出")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data02/wenwantao/dpo_new/outputs/llama-3-8b-instruct-simpo",
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="alpaca_eval_outputs_1024.json",
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
        default=100, # 可以根据你的GPU显存调整
        help="批处理大小"
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用的设备 (代码中实际使用device_map，此参数影响数据类型和tokenizer移动)"
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
    return parser.parse_args()

def format_prompt(instruction):
    """根据llama3模板格式化prompt"""
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def main():
    args = parse_args()

    # ... (模型加载部分保持不变) ...
    logger.info(f"正在加载模型: {args.model_path}")

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
        "device_map": "auto"
    }

    if args.use_flash_attention_2:
        try:
            if torch.__version__ >= "2.0" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                 model_kwargs["attn_implementation"] = "flash_attention_2"
                 logger.info("启用 Flash Attention 2")
            else:
                 logger.warning("Flash Attention 2 不可用或不被支持，将使用默认注意力机制。")
                 logger.warning(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, CUDA capability: {torch.cuda.get_device_capability() if torch.cuda.is_available() else 'N/A'}")
        except ImportError:
            logger.warning("未安装 flash-attn 库，无法使用 Flash Attention 2。请运行 'pip install flash-attn --no-build-isolation'")
        except Exception as e:
             logger.warning(f"尝试启用 Flash Attention 2 时出错: {e}")


    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Tokenizer 没有 pad_token，已设置为 eos_token ({tokenizer.eos_token})")
    tokenizer.padding_side = "left"

    model.eval()

    # ... (数据集加载部分保持不变) ...
    logger.info("正在下载AlpacaEval评估数据集")
    try:
        eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
        logger.info(f"成功加载评估数据集，共 {len(eval_set)} 个样本")
    except Exception as e:
        logger.error(f"下载数据集失败: {e}")
        raise

    if args.max_samples is not None:
        eval_set = eval_set.select(range(min(args.max_samples, len(eval_set))))
        logger.info(f"限制处理样本数为 {len(eval_set)}")

    # ---- 批处理生成 ----
    results = []
    num_batches = math.ceil(len(eval_set) / args.batch_size)
    progress_bar = tqdm(range(num_batches), desc="Processing Batches")

    for i in progress_bar:
        start_index = i * args.batch_size
        end_index = min((i + 1) * args.batch_size, len(eval_set))
        batch_data = eval_set[start_index:end_index] # batch_data 是一个字典，例如 {'instruction': [...], 'output': [...]}

        # === 修改开始 ===
        # 直接从字典中获取指令列表
        batch_instructions = batch_data['instruction']
        # === 修改结束 ===

        # 准备批处理输入
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
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # 解码并提取助手回复 (批处理)
        input_token_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_token_len:]
        batch_outputs_decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 保存批处理结果
        for j, generated_text in enumerate(batch_outputs_decoded):
            # 使用索引 j 从 batch_instructions 列表中获取对应的原始指令
            original_instruction = batch_instructions[j]
            result = {
                "instruction": original_instruction,
                "output": generated_text.strip(),
                "generator": os.path.basename(args.model_path)
            }
            results.append(result)

        # ... (更新进度条和临时保存逻辑保持不变) ...
        progress_bar.set_postfix({"Last batch size": len(batch_instructions)}) # 使用len(batch_instructions)

        if (i + 1) % (100 // args.batch_size + 1) == 0 or (i + 1) == num_batches :
             with open(args.output_file, "w", encoding="utf-8") as f:
                 json.dump(results, f, ensure_ascii=False, indent=4)
             logger.info(f"已处理 {len(results)}/{len(eval_set)} 个样本，中间结果已保存")

    # ... (保存最终结果和后续提示保持不变) ...
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"所有样本处理完成 ({len(results)} 个)，结果已保存至 {args.output_file}")

    logger.info("接下来可以使用alpaca_eval进行评估:")
    logger.info(f"export OPENAI_API_KEY=<your_api_key>")
    logger.info(f"alpaca_eval --model_outputs '{args.output_file}' --annotators_config 'weighted_alpaca_eval_gpt4_turbo'")

if __name__ == "__main__":
    main()