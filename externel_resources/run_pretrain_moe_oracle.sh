#!/bin/bash
# Parameters optimized for 2 GPUs and Shared Expert Pool Multi-Head MoE

LOCAL_BATCH_SIZE=12 
GLOBAL_BATCH_SIZE=768
SAVE_INTERVAL=1000
SEQ_LEN=1024
DATA_SHUFFLE=true
USE_BF16=true
TOTAL_TOKENS=50000000000
LR=1e-4

# Standard MoE Configuration (Baseline)
MOE_TYPE="moe" 
MOE_HEAD_NUM=8 # Not used in standard MoE, but kept for compatibility
MOE_INTERMEDIATE_SIZE=1536 # Aligned with external_resources/scripts/pretrain_baseline.sh
EXPERT_PER_TOKEN=2
NUM_EXPERTS=4 # Aligned with external_resources/scripts/pretrain_baseline.sh
GATING_REFERENCE="oracle" # switch | oracle
MOE_IMPLEMENTATION="cat" # Not used in standard MoE

NUM_WORKERS=16
CONFIG_DIR="./cache_dir/Qwen3-0.6B"
# Use the BINARY data directory
DATA_DIR="/data/share/109_cache_dir/hf_data/dclm_bin/global-shard_01_of_10"
DATASET_TYPE="bin"

OUTPUT_DIR="/data/share/109_cache_dir/multihead_moe/output"
MODEL_NAME="${MOE_TYPE}-baseline-${GATING_REFERENCE}-bin"
MODEL_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}"
CKPT_DIR="${MODEL_OUTPUT_DIR}/checkpoints"
LOG_DIR="${MODEL_OUTPUT_DIR}/logs"

ARGS=""
ARGS+=" --local_batch_size $LOCAL_BATCH_SIZE"
ARGS+=" --global_batch_size $GLOBAL_BATCH_SIZE"
ARGS+=" --save_interval $SAVE_INTERVAL"
ARGS+=" --seq_len $SEQ_LEN"
ARGS+=" --num_workers $NUM_WORKERS"
ARGS+=" --config_dir $CONFIG_DIR"
ARGS+=" --data_dir $DATA_DIR"
ARGS+=" --dataset_type $DATASET_TYPE"
ARGS+=" --total_tokens $TOTAL_TOKENS"
ARGS+=" --ckpt_dir $CKPT_DIR"
ARGS+=" --output_dir $OUTPUT_DIR"
ARGS+=" --model_name $MODEL_NAME"

if [ "$DATA_SHUFFLE" = "true" ]; then
    ARGS+=" --data_shuffle"
else
    ARGS+=" --no_data_shuffle"
fi

if [ "$USE_BF16" = "true" ]; then
    ARGS+=" --use_bf16"
else
    ARGS+=" --no_use_bf16"
fi
ARGS+=" --lr $LR"

ARGS+=" --moe_intermediate_size $MOE_INTERMEDIATE_SIZE"
ARGS+=" --expert_per_token $EXPERT_PER_TOKEN"
ARGS+=" --num_experts $NUM_EXPERTS"
ARGS+=" --gating_reference $GATING_REFERENCE"
ARGS+=" --moe_type $MOE_TYPE"
ARGS+=" --moe_head_num $MOE_HEAD_NUM"
ARGS+=" --moe_implementation $MOE_IMPLEMENTATION"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected ${NUM_GPUS} GPUs"

# Create logs directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Define log file name with timestamp
LOG_FILE="${LOG_DIR}/pretrain_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training... Logs will be saved to ${LOG_FILE}"

# Use a random master port to avoid conflicts
MASTER_PORT=$(shuf -i 29500-65535 -n 1)

# Set NCCL environment variables for stability
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

deepspeed --num_gpus=${NUM_GPUS} --master_port=${MASTER_PORT} pretrain_bin.py ${ARGS} 2>&1 | tee ${LOG_FILE}
