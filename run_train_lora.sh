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
GLOBAL_BATCH_SIZE=512
SAVE_INTERVAL=1000
ZERO_STAGE=1
TOTAL_TOKENS=10000000000  # 10B tokens
WARMUP_STEPS=2000

# Parse Arguments
# Usage: bash run_train_lora.sh <mlp_ratio> <lora_rank> <lora_mode>
MLP_RATIO=${1:-3.0}
LORA_RANK=${2:-0}
LORA_MODE=${3:-none}

# Model Name with LoRA mode
if [ "$LORA_MODE" = "none" ] || [ "$LORA_RANK" = "0" ]; then
    MODEL_NAME="qwen3-0.6b-ratio-${MLP_RATIO}-nolora"
    USE_LORA_FLAG=""
else
    MODEL_NAME="qwen3-0.6b-ratio-${MLP_RATIO}-lora-rank${LORA_RANK}-${LORA_MODE}"
    USE_LORA_FLAG="--use_lora --lora_rank ${LORA_RANK} --lora_mode ${LORA_MODE}"
fi

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${NUM_GPUS} GPUs"

# Master Port
MASTER_PORT=$(shuf -i 20000-65000 -n 1)

INTERMEDIATE_SIZE=$(python -c "print(int(1024 * ${MLP_RATIO}))")

echo "================================================================"
echo "Experiment: MLP + LoRA Ablation Study"
echo "Model Name: ${MODEL_NAME}"
echo "Layers (L): 28 (Fixed)"
echo "MLP Ratio: ${MLP_RATIO} (Intermediate Size: ${INTERMEDIATE_SIZE})"
echo "LoRA Rank: ${LORA_RANK}"
echo "LoRA Mode: ${LORA_MODE}"
echo "Total Tokens: ${TOTAL_TOKENS}"
echo "================================================================"

# Run Training
FULL_OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"
LOG_FILE="${FULL_OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"
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
    --mlp_ratio ${MLP_RATIO} \
    ${USE_LORA_FLAG} \
    --use_bf16 2>&1 | tee ${LOG_FILE}

echo "Experiment ${MODEL_NAME} finished."
