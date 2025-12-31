#!/bin/bash
# run_iso_train.sh

# Environment Setup
source /root/anaconda3/bin/activate oracle-moe
export PYTHONPATH=$PYTHONPATH:/data/250010109/qwen_exp

# Configs
CONFIG_DIR="/data/250010109/qwen_exp/configs"
DATA_DIR="/data/share/109_cache_dir/hf_data/dclm_bin/global-shard_01_of_10"
# 更新输出根目录为用户要求的 /data/share/109_cache_dir/qwen_exp/output
OUTPUT_ROOT="/data/share/109_cache_dir/qwen_exp/output"

# Hyperparameters
SEQ_LEN=1024
LR=1e-4
BATCH_SIZE=16
GLOBAL_BATCH_SIZE=512
TOTAL_TOKENS=5000000000 # 10B tokens
WARMUP_STEPS=2000

# Get MLP Ratio from argument (e.g., 0.5, 1.0, 3.0, 8.0)
RATIO=${1:-3.0}

# Model Name with -iso suffix to distinguish from non-iso experiments
MODEL_NAME="qwen3-0.6b-mlp-ratio-${RATIO}-iso"

echo "================================================================"
echo "Experiment: ISO-Parameter Width vs Depth Study"
echo "Model Name: ${MODEL_NAME}"
echo "Targeting 0.6B total parameters (equivalent to k=3.0, L=28)"
echo "MLP Ratio (k): ${RATIO}"
echo "Output Root: ${OUTPUT_ROOT}"
echo "Final Directory: ${OUTPUT_ROOT}/${MODEL_NAME}"
echo "================================================================"

# Run Training (直接使用 train_qwen3.py)
# 修复参数名错误: --learning_rate -> --lr
LOG_FILE="${OUTPUT_ROOT}/${MODEL_NAME}/logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "Logging to: ${LOG_FILE}"

deepspeed --master_port $(shuf -i 20000-65000 -n 1) train_qwen3.py \
    --config_dir $CONFIG_DIR \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_ROOT \
    --model_name $MODEL_NAME \
    --mlp_ratio $RATIO \
    --iso_parameter \
    --seq_len $SEQ_LEN \
    --lr $LR \
    --local_batch_size $BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --total_tokens $TOTAL_TOKENS \
    --warmup_steps $WARMUP_STEPS \
    --zero_stage 1 \
    --deepspeed 2>&1 | tee ${LOG_FILE}
