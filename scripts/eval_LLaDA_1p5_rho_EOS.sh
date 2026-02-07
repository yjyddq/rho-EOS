#!/bin/bash
set -e

LOG_FILE="./logs/run_llada1dot5_rho_EOS_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -i "$LOG_FILE") 2>&1

RHO_LOW=0.4
RHO_HIGH=0.6
SCHEDULER="exp"

# parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rho_low)
            RHO_LOW="$2"
            shift 2
            ;;
        --rho_high)
            RHO_HIGH="$2"
            shift 2
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--rho_low VALUE] [--rho_high VALUE] [--scheduler VALUE]"
            exit 1
            ;;
    esac
done

export HF_HOME=
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ----------- Auto select accelerate config -----------
# 1. Prioritize using Nvidia smi for statistics, and return to Python torch for failures
NUM_GPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$NUM_GPU" -eq 0 ]; then
    NUM_GPU=$(python - <<'PY'
import torch, sys
sys.stdout.write(str(torch.cuda.device_count()))
PY
)
fi

case "$NUM_GPU" in
    2)  ACCEL_CONFIG="accelerate_config_2gpu.yaml" ;;
    4)  ACCEL_CONFIG="accelerate_config_4gpu.yaml" ;;
    8)  ACCEL_CONFIG="accelerate_config.yaml" ;;
    *)  echo "Unsupported GPU count: $NUM_GPU"; exit 1 ;;
esac
echo "Detected $NUM_GPU GPU(s), using $ACCEL_CONFIG"
# -------------------------------------------------
BASE_OUTPUT_PATH="./results/rho-eos-1p5-${RHO_LOW}-${RHO_HIGH}-${SCHEDULER}-$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="GSAI-ML/LLaDA-1.5"


TASKS=("gsm8k" "math500")
LENGTHS=(64 128 256 512 1024)
for task in "${TASKS[@]}"; do
    for length in "${LENGTHS[@]}"; do
        echo "======================================================"
        echo "<<rho-EOS>> -> Task: ${task}, L_init: ${length}"
        echo "======================================================"
        OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"

        accelerate launch --config_file accelerate_config.yaml evaluation_script.py \
            -m dllm_eval \
            --model LLaDA_rho_EOS \
            --tasks "${task}" \
            --batch_size 8 \
            --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
            --gen_kwargs "low_density_threshold=${RHO_LOW},high_density_threshold=${RHO_HIGH},scheduler=${SCHEDULER},block_length=32,initial_gen_length=${length},max_gen_length=2048,cfg_scale=0.0,high_conf_threshold=0.9,low_conf_threshold=0.1,eos_confidence_threshold=0.5,expand_eos_confidence_threshold=0.9,expansion_factor=8,eos_check_tokens=32 "  \
            --num_fewshot 0  \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn

        python metrics/${task}.py \
            --model_path "${MODEL_PATH}" \
            --res_path "${OUTPUT_PATH}"
    done
done


TASKS=("humaneval" "mbpp")
LENGTHS=(64 128 256 512 1024)
for task in "${TASKS[@]}"; do
    for length in "${LENGTHS[@]}"; do
        echo "======================================================"
        echo "<<rho-EOS>> -> Task: ${task}, L_init: ${length}"
        echo "======================================================"
        OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"

        accelerate launch --config_file accelerate_config.yaml evaluation_script.py \
            -m dllm_eval \
            --model LLaDA_rho_EOS \
            --tasks "${task}" \
            --batch_size 8 \
            --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
            --gen_kwargs "low_density_threshold=${RHO_LOW},high_density_threshold=${RHO_HIGH},scheduler=${SCHEDULER},block_length=32,initial_gen_length=${length},max_gen_length=2048,cfg_scale=0.0,high_conf_threshold=0.9,low_conf_threshold=0.1,eos_confidence_threshold=0.5,expand_eos_confidence_threshold=0.9,expansion_factor=8,eos_check_tokens=32 "  \
            --num_fewshot 0  \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --confirm_run_unsafe_code

        python metrics/${task}.py \
            --model_path "${MODEL_PATH}" \
            --res_path "${OUTPUT_PATH}"
    done
done
