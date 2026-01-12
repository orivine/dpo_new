#!/bin/bash

# 模型路径
MODEL_PATH="/root/autodl-tmp/dpo_new/outputs/llama-3-8b-instruct-simpo-0_9-weighted-new"
# 输出文件
OUTPUT_FILE="alpaca_eval_outputs_simpo_0_9_weighted_new.json"
# DeepSpeed配置文件
DS_CONFIG="eval/alpacaeval2/ds_generation_config.json"
# 每个GPU上的批量大小
BATCH_SIZE=12
# 生成的最大token数
MAX_NEW_TOKENS=1024

# 调用deepspeed启动多卡运行
deepspeed --num_gpus=4 generate_deepspeed.py \
    --model_path $MODEL_PATH \
    --output_file $OUTPUT_FILE \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature 0.9 \
    --top_p 1.0 \
    --use_flash_attention_2 \
    --deepspeed_config $DS_CONFIG 