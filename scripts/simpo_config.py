from dataclasses import dataclass
from typing import Dict, Literal, Optional
from transformers import TrainingArguments


@dataclass
class SimPOConfig(TrainingArguments):
    r"""
    SimPOConfig collects all training arguments related to the [`SimPOTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        beta (`float`, defaults to 2.0):
            The beta factor in SimPO loss.
        gamma_beta_ratio (`float`, defaults to 0.25):
            The ratio between the target reward margin (gamma) and beta in SimPO loss.
        sft_weight (`float`, defaults to 0.0):
            SFT loss weight added to the SimPO loss (0.0 is not using SFT).
        label_smoothing (`float`, defaults to 0):
            The label smoothing factor. This argument is required if you want to use the default data collator.
        loss_type (`str`, defaults to `sigmoid`):
            The type of loss to use. This argument is required if you want to use the default data collator.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `None`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model`.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
        use_entity_weighting (`bool`, defaults to `False`): 
            Whether to use entity-based token weighting in loss calculation.
        entity_weight_chosen (`float`, defaults to 1.5):
            Weight multiplier for key entity tokens in chosen responses (>1 to emphasize).
        entity_weight_rejected (`float`, defaults to 0.5):
            Weight multiplier for key entity tokens in rejected responses (<1 to de-emphasize).
        moonshot_api_key (`Optional[str]`, defaults to `None`):
            API key for Moonshot API. If None, will try to get from environment variable.
        apply_entity_weighting_during_eval (`bool`, defaults to `False`):
            Whether to apply entity weighting during evaluation. If False, entity weighting will only be applied during training.
        save_mask_records (`bool`, defaults to `False`):
            Whether to save mask records to a JSONL file. Set to False to reduce memory usage and avoid file conflicts.
    """
    
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    max_target_length: Optional[int] = None

    beta: float = 2.0
    gamma_beta_ratio: float = 0.25
    sft_weight: float = 0.0
    label_smoothing: float = 0
    loss_type: Literal["sigmoid", "hinge"] = "sigmoid"
    disable_dropout: bool = True

    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None

    model_init_kwargs: Optional[Dict] = None

    dataset_num_proc: Optional[int] = None
    
    # 实体加权相关参数
    use_entity_weighting: bool = True
    entity_weight_chosen: float = 0.5
    entity_weight_rejected: float = 1.5
    moonshot_api_key: Optional[str] = "sk-yAx4wtCUkyDakfury04dmMDtiCTT07tC2UypQMz7E2rkDVQj"
    apply_entity_weighting_during_eval: bool = False
    save_mask_records: bool = False

