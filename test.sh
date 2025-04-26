#!/bin/bash

# 设置Moonshot API密钥
# 如果没有设置环境变量，请在这里设置，否则可以通过环境变量传入
# export MOONSHOT_API_KEY="your_api_key_here"

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo.yaml
####
# WANDB_DISABLED=1 PYTHONPATH=/home/kewenjun/VL-Embedding_7/dpo_new python scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo.yaml 2>&1 | tee run.log
####

# 使用deepspeed配置文件启动训练, 8b模型
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_DISABLED=1 PYTHONPATH=/root/autodl-tmp/dpo_new ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file accelerate_configs/deepspeed_zero3.yaml \
scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo.yaml 2>&1 | tee run_8b.log


# 使用deepspeed配置文件启动训练, 1b模型
# CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_DISABLED=1 PYTHONPATH=/root/autodl-tmp/dpo_new ACCELERATE_LOG_LEVEL=info accelerate launch \
# --config_file accelerate_configs/deepspeed_zero3.yaml \
# scripts/run_simpo.py training_configs/llama-3_2-1b-instruct-simpo.yaml 2>&1 | tee run_1b.log

# 如果要禁用实体加权，可以添加以下参数
# WANDB_DISABLED=1 PYTHONPATH=/data02/wenwantao/Simpo_test2 python scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo.yaml --use_entity_weighting=False 2>&1 | tee run_test2.log

# 使用DeepSpeed进行分布式训练的命令
# PYTHONPATH=/data02/wenwantao/Simpo_test2 ACCELERATE_LOG_LEVEL=info accelerate launch \
# --config_file accelerate_configs/deepspeed_zero3.yaml \
# --num_processes=1 \
# scripts/run_simpo.py training_configs/llama-3-8b-instruct-simpo.yaml