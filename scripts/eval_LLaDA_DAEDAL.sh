#!/bin/bash
set -e

LOG_FILE="./logs/run_llada_daedal_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -i "$LOG_FILE") 2>&1

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

BASE_OUTPUT_PATH="./results/daedal"
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"


TASKS=("gsm8k" "math500")
LENGTHS=(64 128 256 512 1024 2048)
for task in "${TASKS[@]}"; do
    for length in "${LENGTHS[@]}"; do
        echo "======================================================"
        echo "<<DAEDAL>> -> Task: ${task}, L_init: ${length}"
        echo "======================================================"
        OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"

        accelerate launch --config_file "$ACCEL_CONFIG" evaluation_script.py \
            -m dllm_eval \
            --model LLaDA_DAEDAL \
            --tasks "${task}" \
            --batch_size 8 \
            --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning>" \
            --gen_kwargs "block_length=32,initial_gen_length=${length},max_gen_length=2048,cfg_scale=0.0,high_conf_threshold=0.9,low_conf_threshold=0.1,eos_confidence_threshold=0.5,expand_eos_confidence_threshold=0.9,expansion_factor=8,eos_check_tokens=32"  \
            --num_fewshot 0  \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn
        
        # python metrics/${task}.py \
        #     --model_path "${MODEL_PATH}" \
        #     --res_path "${OUTPUT_PATH}"
    done
done


TASKS=("humaneval" "mbpp")
LENGTHS=(1024)
for task in "${TASKS[@]}"; do
    for length in "${LENGTHS[@]}"; do
        echo "======================================================"
        echo "<<DAEDAL>> -> Task: ${task}, L_init: ${length}"
        echo "======================================================"
        OUTPUT_PATH="${BASE_OUTPUT_PATH}/${task}_${length}"

        accelerate launch --config_file "$ACCEL_CONFIG" evaluation_script.py \
            -m dllm_eval \
            --model LLaDA_DAEDAL \
            --tasks "${task}" \
            --batch_size 8 \
            --model_args "pretrained=${MODEL_PATH},assistant_prefix=<reasoning> " \
            --gen_kwargs "block_length=32,initial_gen_length=${length},max_gen_length=2048,cfg_scale=0.0,high_conf_threshold=0.9,low_conf_threshold=0.1,eos_confidence_threshold=0.5,expand_eos_confidence_threshold=0.9,expansion_factor=8,eos_check_tokens=32 "  \
            --num_fewshot 0  \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --confirm_run_unsafe_code
        
        # python metrics/${task}.py \
        #     --model_path "${MODEL_PATH}" \
        #     --res_path "${OUTPUT_PATH}"
    done
done
