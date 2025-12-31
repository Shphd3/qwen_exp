#!/bin/bash

# Environment Setup
source /root/anaconda3/bin/activate oracle-moe
export PYTHONPATH=$PYTHONPATH:/data/250010109/qwen_exp

# Configs
CONFIG_DIR="/data/250010109/qwen_exp/configs"
DATA_DIR="/data/share/109_cache_dir/hf_data/dclm_bin/global-shard_01_of_10"
OUTPUT_ROOT="/data/share/109_cache_dir/qwen_exp/output"

# Hyperparameters (Fixed Depth L=28)
SEQ_LEN=1024
LR=1e-4
BATCH_SIZE=8
# Global Batch Size 建议范围: 128 ~ 512
# 384 @ 1024 seq_len ≈ 393K tokens/step. 
# 对于 0.6B 模型，这是一个兼顾吞吐量和收敛稳定性的选择。
GLOBAL_BATCH_SIZE=512
SAVE_INTERVAL=1000
ZERO_STAGE=1
TOTAL_TOKENS=5000000000  # 5B tokens
WARMUP_STEPS=2000

# Get MLP Ratio from argument (Options: 0.5, 1.0, 2.0, 4.0)
RATIO=${1:-3.0}

# Model Name with Ratio suffix
MODEL_NAME="qwen3-0.6b-mlp-ratio-${RATIO}"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${NUM_GPUS} GPUs"

# Master Port
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

INTERMEDIATE_SIZE=$(python -c "print(int(1024 * ${RATIO}))")

echo "================================================================"
echo "Experiment: MLP Width Scaling Study"
echo "Model Name: ${MODEL_NAME}"
echo "Layers (L): 28 (Fixed)"
echo "MLP Ratio (k): ${RATIO} (Intermediate Size: ${INTERMEDIATE_SIZE})"
echo "Total Tokens: 10B"
echo "================================================================"

# Run Training
LOG_FILE="${OUTPUT_ROOT}/${MODEL_NAME}/logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG_FILE}")"

deepspeed --num_gpus=${NUM_GPUS} --master_port=${MASTER_PORT} train_qwen3.py \
    --local_batch_size ${BATCH_SIZE} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --save_interval ${SAVE_INTERVAL} \
    --seq_len ${SEQ_LEN} \
    --lr ${LR} \
    --total_tokens ${TOTAL_TOKENS} \
    --warmup_steps ${WARMUP_STEPS} \
    --zero_stage ${ZERO_STAGE} \
    --num_workers 8 \
    --config_dir ${CONFIG_DIR} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_ROOT} \
    --model_name ${MODEL_NAME} \
    --mlp_ratio ${RATIO} \
    --use_bf16 2>&1 | tee ${LOG_FILE}

echo "Experiment ${MODEL_NAME} finished."
