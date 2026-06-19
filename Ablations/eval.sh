#!/usr/bin/env bash
set -euo pipefail

GPU_ID=0
CHECK_INTERVAL=60
MEM_THRESHOLD=10000

MODEL_DIR="./grpo_Qwen25_15B_gsm8k_iterations2_checkpoints"
OUT_DIR="./eval/grpo_qwen25_15B_gsm8k_iterations2"
PORT=8002

echo "Waiting for GPU $GPU_ID to be free..."

while true; do
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID")

    if [ "$FREE_MEM" -ge "$MEM_THRESHOLD" ]; then
        echo "GPU $GPU_ID has enough free memory (${FREE_MEM} MiB). Starting training..."
        break
    fi

    echo "GPU $GPU_ID busy: ${FREE_MEM} MiB Free. Checking again in $CHECK_INTERVAL seconds..."
    sleep "$CHECK_INTERVAL" 
done

# uv run python3 ../Cascade\ 2/main.py
echo "Training finished successfully. Starting evaluations..."

mkdir -p "$OUT_DIR"

for CKPT in $(find "$MODEL_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V); do
    NAME=$(basename "$CKPT")

    if [ -d "$OUT_DIR/$NAME" ]; then
        echo "Skipping $NAME, already evaluated."
        continue
    fi

    echo "Evaluating $NAME"

    CUDA_VISIBLE_DEVICES=$GPU_ID uv run vllm serve "$CKPT" \
        --dtype auto \
        --port "$PORT" \
        --host 0.0.0.0 \
        #--hf-overrides '{"tokenizer_config": {"fix_mistral_regex": true}}' \
        --gpu_memory_utilization 0.15 &

    VLLM_PID=$!

    until curl -s "http://localhost:$PORT/v1/models" > /dev/null; do
        echo "Waiting for vLLM..."
        sleep 10
    done

    CUDA_VISIBLE_DEVICES=$GPU_ID uv run lm-eval run \
        # --model local-chat-completions \
        --model_args "model=$CKPT,base_url=http://localhost:$PORT/v1/chat/completions" \
        --tasks gsm8k \
        --output_path "$OUT_DIR/$NAME" \
        --log_samples \
        --apply_chat_template \
        --gen_kwargs '{"temperature":0.0,"max_gen_toks":1024}'

    kill "$VLLM_PID"
    wait "$VLLM_PID" 2>/dev/null || true
    sleep 10
done

echo "All evaluations completed."

# CUDA_VISIBLE_DEVICES=1 uv run vllm serve rloo_Qwen25_15B_gsm8k_steps16 \
#     --host 0.0.0.0 \
#     --port 8002 \
#     --dtype auto \
#     --gpu-memory-utilization=0.7

# CUDA_VISIBLE_DEVICES=1 uv run lm-eval \
#     --model local-chat-completions \
#     --model_args model=rloo_Qwen25_15B_gsm8k_steps16,base_url=http://localhost:8002/v1/chat/completions \
#     --output_path "eval/rloo_qwen25_15B_gsm8k_steps16" \
#     --task gsm8k \
#     --apply_chat_template \
#     --log_samples \
#     --gen_kwargs '{"temperature": 0.0, "max_gen_toks": 1024}'