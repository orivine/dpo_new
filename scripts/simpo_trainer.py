import inspect
import random
import warnings
import json
import os
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer
from trl.trainer import CPOTrainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_torch_fx_proxy

from trl.import_utils import is_peft_available, is_wandb_available
from simpo_config import SimPOConfig

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from transformers import TrainingArguments

from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)

# 导入实体抽取器
try:
    from scripts.entity_extractor import EntityExtractor
except ImportError:
    print("警告: 无法导入EntityExtractor模块，实体加权功能将被禁用")
    EntityExtractor = None

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


class SimPOTrainer(Trainer):
    r"""
    Initialize SimPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        args (`SimPOConfig`):
            The SimPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "simpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SimPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the SimPOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SimPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a SimPO dataset.")
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the SimPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        else:
            max_length = args.max_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the SimPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        else:
            max_prompt_length = args.max_prompt_length

        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the SimPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128
        else:
            max_target_length = args.max_target_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if args.disable_dropout:
            disable_dropout_in_model(model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

        if args.loss_type in ["hinge"] and args.label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = args.beta
        self.gamma_beta_ratio = args.gamma_beta_ratio
        self.sft_weight = args.sft_weight
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type

        # 添加实体加权相关参数
        self.use_entity_weighting = getattr(args, "use_entity_weighting", False)
        self.entity_weight_chosen = getattr(args, "entity_weight_chosen", 0.5)
        self.entity_weight_rejected = getattr(args, "entity_weight_rejected", 1.5)
        
        # 初始化实体抽取器
        self.entity_extractor = None
        if self.use_entity_weighting:
            # 打印当前Python路径，帮助调试
            import sys
            print(f"当前Python路径: {sys.path}")
            
            if EntityExtractor is None:
                print("警告: EntityExtractor类未定义，请确保entity_extractor.py文件位于正确的位置")
                warnings.warn("未找到EntityExtractor模块，将不使用实体加权功能")
                self.use_entity_weighting = False
            else:
                try:
                    api_key = getattr(args, "moonshot_api_key", None)
                    if api_key:
                        print(f"使用配置文件中的API密钥: {api_key[:8]}...")
                    else:
                        print("未找到API密钥，将尝试从环境变量获取")
                    
                    self.entity_extractor = EntityExtractor(api_key=api_key, cache_dir=args.output_dir)
                    print(f"实体抽取器初始化成功，将对关键token进行加权 (chosen={self.entity_weight_chosen}, rejected={self.entity_weight_rejected})")
                except Exception as e:
                    import traceback
                    print(f"实体抽取器初始化失败: {e}")
                    print("错误详情:")
                    traceback.print_exc()
                    warnings.warn(f"实体抽取器初始化失败: {e}. 将不使用实体加权功能.")
                    self.use_entity_weighting = False

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # 初始化掩码记录
        self.mask_records = []
        self.mask_record_file = os.path.join(args.output_dir, "mask_records.jsonl")
        print(f"掩码记录将保存到: {self.mask_record_file}")
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(self.mask_record_file), exist_ok=True)
        # 清空已有文件
        with open(self.mask_record_file, "w") as f:
            f.write("")

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc, load_from_cache_file=False)    # 添加load_from_cache_file=False参数，使得每次都调用tokenize_row函数，便于调试检查
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc, load_from_cache_file=False)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def record_mask_data(self, feature, entities, chosen_mask, rejected_mask, chosen_highlight=None, rejected_highlight=None):
        """
        记录掩码数据到文件
        
        Args:
            feature: 原始特征数据
            entities: 提取的实体列表
            chosen_mask: 选择回答的掩码
            rejected_mask: 拒绝回答的掩码
            chosen_highlight: 带有高亮标记的chosen文本
            rejected_highlight: 带有高亮标记的rejected文本
        """
        record = {
            # "prompt": feature["prompt"][:200] + "..." if len(feature["prompt"]) > 200 else feature["prompt"],
            "prompt": feature["prompt"],
            "chosen": feature["chosen"][:200] + "..." if len(feature["chosen"]) > 200 else feature["chosen"],
            "rejected": feature["rejected"][:200] + "..." if len(feature["rejected"]) > 200 else feature["rejected"],
            "entities": entities,
            "chosen_mask_stats": {
                "length": len(chosen_mask) if chosen_mask else 0,
                "non_zero_count": sum(1 for m in chosen_mask if m > 0) if chosen_mask else 0,
                "non_zero_percentage": round(sum(1 for m in chosen_mask if m > 0) / len(chosen_mask) * 100, 2) if chosen_mask and len(chosen_mask) > 0 else 0,
                "mask_preview": chosen_mask[:50] if chosen_mask else None,
            },
            "rejected_mask_stats": {
                "length": len(rejected_mask) if rejected_mask else 0,
                "non_zero_count": sum(1 for m in rejected_mask if m > 0) if rejected_mask else 0,
                "non_zero_percentage": round(sum(1 for m in rejected_mask if m > 0) / len(rejected_mask) * 100, 2) if rejected_mask and len(rejected_mask) > 0 else 0,
                "mask_preview": rejected_mask[:50] if rejected_mask else None,
            }
        }
        
        # 添加高亮文本
        if chosen_highlight:
            # record["chosen_highlight"] = chosen_highlight[:300] + "..." if len(chosen_highlight) > 300 else chosen_highlight
            record["chosen_highlight"] = chosen_highlight
        if rejected_highlight:
            # record["rejected_highlight"] = rejected_highlight[:300] + "..." if len(rejected_highlight) > 300 else rejected_highlight
            record["rejected_highlight"] = rejected_highlight
        
        simplified_record = {
            "prompt": feature["prompt"],
            "entities": entities
        }
        
        # 保存记录到文件
        with open(self.mask_record_file, "a", encoding="utf-8") as f:
            # f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.write(json.dumps(simplified_record, ensure_ascii=False) + "\n")
        
        # 打印简要统计信息
        chosen_percent = record['chosen_mask_stats']['non_zero_percentage']
        rejected_percent = record['rejected_mask_stats']['non_zero_percentage']
        # print(f"已记录掩码数据 - 实体数: {len(entities) if entities else 0}, "
            #   f"chosen掩码非零比例: {chosen_percent}%, "
            #   f"rejected掩码非零比例: {rejected_percent}%")
              
        # 也保存在内存中
        self.mask_records.append(record)

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a SimPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt_original = feature["prompt_original"]   # 原始prompt，未经过apply_chat_template处理
        prompt = feature["prompt"]                     # 经过apply_chat_template处理后的prompt
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        # 如果启用了实体加权，并且实体抽取器初始化成功，创建实体掩码
        chosen_mask = None
        rejected_mask = None
        entities = None
        chosen_highlight = None
        rejected_highlight = None
        if self.use_entity_weighting and self.entity_extractor is not None:
            try:
                # print(f"为数据创建实体掩码: prompt={prompt[:50]}...")
                masks_data = self.entity_extractor.create_entity_masks(
                    self.tokenizer, prompt_original, chosen, rejected      # 使用原始prompt提取实体，而不是apply_chat_template后的prompt，避免提取实体时收到干扰
                )
                chosen_mask = masks_data.get("chosen_mask")
                rejected_mask = masks_data.get("rejected_mask")
                entities = masks_data.get("entities", [])
                
                # 获取高亮文本
                chosen_highlight = masks_data.get("chosen_highlight")
                rejected_highlight = masks_data.get("rejected_highlight")
                
                # 记录掩码数据
                self.record_mask_data(
                    feature, 
                    entities, 
                    chosen_mask, 
                    rejected_mask,
                    chosen_highlight,
                    rejected_highlight
                )
            except Exception as e:
                warnings.warn(f"创建实体掩码失败: {e}")

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt. Avoid adding if it's already there
            bos_token_id = self.tokenizer.bos_token_id
            if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer. Avoid adding if it's already there
            eos_token_id = self.tokenizer.eos_token_id
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]
            
            # 处理掩码，确保长度与tokenized结果匹配
            if chosen_mask is not None and len(chosen_mask) > 0:
                # 确保掩码长度与chosen_tokens的input_ids长度一致，不足则填充
                pad_length = len(chosen_tokens["input_ids"]) - len(chosen_mask)
                if pad_length > 0:
                    chosen_mask = chosen_mask + [0] * pad_length
                # 如果掩码长度超过，则进行截断
                elif pad_length < 0:
                    chosen_mask = chosen_mask[:len(chosen_tokens["input_ids"])]
            
            if rejected_mask is not None and len(rejected_mask) > 0:
                # 确保掩码长度与rejected_tokens的input_ids长度一致
                pad_length = len(rejected_tokens["input_ids"]) - len(rejected_mask)
                if pad_length > 0:
                    rejected_mask = rejected_mask + [0] * pad_length
                elif pad_length < 0:
                    rejected_mask = rejected_mask[:len(rejected_tokens["input_ids"])]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])
            
            # 创建完整的序列掩码，为提示部分填充0
            if chosen_mask is not None:
                prompt_pad = [0] * len(chosen_tokens["prompt_input_ids"])
                chosen_sequence_tokens["token_weights"] = prompt_pad + chosen_mask
                
                # 确保掩码长度与序列长度一致
                token_weights_len = len(chosen_sequence_tokens["token_weights"])
                input_ids_len = len(chosen_sequence_tokens["input_ids"])
                
                if token_weights_len < input_ids_len:
                    chosen_sequence_tokens["token_weights"] += [0] * (input_ids_len - token_weights_len)
                elif token_weights_len > input_ids_len:
                    chosen_sequence_tokens["token_weights"] = chosen_sequence_tokens["token_weights"][:input_ids_len]
            
            if rejected_mask is not None:
                prompt_pad = [0] * len(rejected_tokens["prompt_input_ids"])
                rejected_sequence_tokens["token_weights"] = prompt_pad + rejected_mask
                
                # 确保掩码长度与序列长度一致
                token_weights_len = len(rejected_sequence_tokens["token_weights"])
                input_ids_len = len(rejected_sequence_tokens["input_ids"])
                
                if token_weights_len < input_ids_len:
                    rejected_sequence_tokens["token_weights"] += [0] * (input_ids_len - token_weights_len)
                elif token_weights_len > input_ids_len:
                    rejected_sequence_tokens["token_weights"] = rejected_sequence_tokens["token_weights"][:input_ids_len]

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]
            
            # 处理编码器-解码器模型的掩码
            if chosen_mask is not None:
                # 确保掩码长度与chosen_labels长度一致
                if len(chosen_mask) < len(batch["chosen_labels"]):
                    chosen_mask = chosen_mask + [0] * (len(batch["chosen_labels"]) - len(chosen_mask))
                elif len(chosen_mask) > len(batch["chosen_labels"]):
                    chosen_mask = chosen_mask[:len(batch["chosen_labels"])]
                batch["chosen_token_weights"] = chosen_mask
            
            if rejected_mask is not None:
                # 确保掩码长度与rejected_labels长度一致
                if len(rejected_mask) < len(batch["rejected_labels"]):
                    rejected_mask = rejected_mask + [0] * (len(batch["rejected_labels"]) - len(rejected_mask))
                elif len(rejected_mask) > len(batch["rejected_labels"]):
                    rejected_mask = rejected_mask[:len(batch["rejected_labels"])]
                batch["rejected_token_weights"] = rejected_mask

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        # 特殊处理token_weights，从list转换为torch.Tensor
        for k in batch:
            if not isinstance(batch[k], torch.Tensor) and k.endswith("_token_weights"):
                # 如果是嵌套列表，比如每个样本的 token_weights 形如 [0,1,0, ...]
                if isinstance(batch[k], list) and batch[k] and isinstance(batch[k][0], list):
                    # 这里我们使用之前计算得到的 max_length 作为目标长度
                    target_length = max_length
                    padded_list = []
                    for seq in batch[k]:
                        # 如果当前样本的掩码长度不足 target_length，则进行填充；如果超过则截断
                        if len(seq) < target_length:
                            padded_seq = seq + [0] * (target_length - len(seq))
                        else:
                            padded_seq = seq[:target_length]
                        padded_list.append(padded_seq)
                    try:
                        batch[k] = torch.tensor(padded_list, dtype=torch.float32)
                    except Exception as e:
                        print(f"转换 {k} 时出错：{e}")
                else:
                    try:
                        batch[k] = torch.tensor(batch[k], dtype=torch.float32)
                    except Exception as e:
                        print(f"转换 {k} 时出错：{e}")

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):       # FIXME：问题出在此处！因为提取出来的chosen_labels和rejected_labels是list，而不是tensor，所以无法进入concatenated_batch
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id    # 只有labels的pad_value是-100
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask") or k.endswith("_token_weights"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask") or k.endswith("_token_weights"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch   # 返回处理后的batch，其中包含concatenated_input_ids, concatenated_attention_mask, concatenated_labels, concatenated_token_weights。改名后的

    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps   # 此处暂时没有乘beta，常数后面再乘也没关系
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - self.gamma_beta_ratio

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)   # label_smoothing默认为0
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        # 分别计算chosen和rejected的logits
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        
        chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]
        rejected_labels = concatenated_batch["concatenated_labels"][len_chosen:]

        # 获取token权重，如果有的话
        token_weights_chosen = None
        token_weights_rejected = None
        if self.use_entity_weighting and "concatenated_token_weights" in concatenated_batch:
            token_weights = concatenated_batch["concatenated_token_weights"]
            token_weights_chosen = token_weights[:len_chosen] if token_weights is not None else None
            token_weights_rejected = token_weights[len_chosen:] if token_weights is not None else None

        # 分别计算chosen和rejected的log_probs，应用不同的权重
        chosen_logps = self.get_batch_logps(
            chosen_logits,
            chosen_labels,
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            token_weights=token_weights_chosen,    # 权重
            weight_chosen=self.entity_weight_chosen,    # 权值
            weight_rejected=self.entity_weight_rejected, 
            is_chosen=True,
        )
        
        rejected_logps = self.get_batch_logps(
            rejected_logits,
            rejected_labels,
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            token_weights=token_weights_rejected,
            weight_chosen=self.entity_weight_chosen,
            weight_rejected=self.entity_weight_rejected,
            is_chosen=False,
        )

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels)

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
        token_weights: Optional[torch.FloatTensor] = None,
        weight_chosen: float = 1.5,
        weight_rejected: float = 0.5,
        is_chosen: bool = True,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            token_weights: Optional tensor of weights for each token. Shape: (batch_size, sequence_length)
            weight_chosen: Weight multiplier for key entity tokens in chosen responses.
            weight_rejected: Weight multiplier for key entity tokens in rejected responses.
            is_chosen: Whether the inputs are chosen or rejected, used to determine the appropriate weight multiplier.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
            if token_weights is not None:
                token_weights = token_weights[:, 1:]

        # 添加调试信息：每个批次处理前打印掩码信息
        if token_weights is not None:
            print(f"损失函数调试 - {'chosen' if is_chosen else 'rejected'}回答:")
            print(f"  权重形状: {token_weights.shape}")
            # 计算非零权重的百分比
            non_zero_percentage = (token_weights > 0).float().mean().item() * 100
            print(f"  非零权重百分比: {non_zero_percentage:.2f}%")
            print(f"  将使用权重乘数: {weight_chosen if is_chosen else weight_rejected}")
        
        loss_mask = labels != label_pad_token_id   # torch.nonzero(loss_mask).shape = torch.Size([999, 2]), torch.nonzero(labels).shape = torch.Size([997, 2])，loss_mask不为零的元素更多的原因是因为，原本labels中有一些元素就是0，这样的元素会因为不是padding元素所以在loss_mask中为True (1)，而在labels检查的时候天然为0

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        # 应用token加权
        if token_weights is not None:
            # 将token_weights转换为与per_token_logps相同类型
            token_weights = token_weights.to(per_token_logps.dtype).to(per_token_logps.device)
            
            # 创建权重掩码，对关键token (token_weights=1) 应用权重
            weight_multiplier = torch.ones_like(token_weights)
            weight_value = weight_chosen if is_chosen else weight_rejected
            weight_multiplier = torch.where(token_weights > 0, weight_value, 1.0)   # 设置最终相乘的权重向量！
            
            # 添加调试信息：权重应用后
            # original_mean = per_token_logps[loss_mask].mean().item()
            # if average_log_prob:
                # result_berfore_weight = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            
            # 应用权重到per_token_logps！
            per_token_logps = per_token_logps * weight_multiplier
            
            # weighted_mean = per_token_logps[loss_mask].mean().item()        # TODO：输出值和形状:-0.8734683990478516, per_token_logps.shape = torch.Size([2, 815])   per_token_logps[loss_mask].shape = torch.Size([812])
            # print(f"  log概率均值: 原始={original_mean:.4f}, 加权后={weighted_mean:.4f}")

        if average_log_prob:
            result = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)    # TODO: 输出值和形状, 形状和per_token_logps相同
        else:
            result = (per_token_logps * loss_mask).sum(-1)
        
        return result

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
        ) = self.concatenated_forward(model, batch)

        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )

        loss = losses.mean()

        if self.sft_weight > 0.0:
            if not self.is_encoder_decoder:
                policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
                chosen_labels = chosen_labels[..., 1:].clone()
            loss_func = nn.CrossEntropyLoss()
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), chosen_labels.view(-1))
            loss = self.sft_weight * sft_loss + loss
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu()
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        return policy_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy"],
                        rows=[
                            [prompt, pol[len(prompt) :]]
                            for prompt, pol in zip(random_batch["prompt"], policy_output_decoded)
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "simpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
