#!/bin/bash

set -e

scp -r -P 8822 -i ~/.ssh/id_ed25519_meluxina \
    u104403@login.lxp.lu:/project/home/p201382/erisa/IF_RL_Fraction/new  \
    /data/home/erisa.kohansal/Workplace/Cascade2/verl-version/Meluxina/IF_RL_Fraction/new


# for i in $(seq 80 10 90); do

#     scp -r -P 8822 -i ~/.ssh/id_ed25519_meluxina \
#     u104403@login.lxp.lu:/project/home/p201382/erisa/IF_RL_Fraction/with_ds/checkpoints/merged_checkpoints/global_step_${i}  \
#     /data/home/erisa.kohansal/Workplace/Cascade2/verl-version/Meluxina/IF_RL_Fraction/with_ds/checkpoints/merged_checkpoints/global_step_${i}

#     if [ $? -eq 0 ]; then
#         echo "Checkpoint global_step_${i} transferred to hippo"
#     else
#         echo "FAILED: global_step_${i}"
#         failed+=("$i")
#     fi
# done

GPU_ID=1
PORT=8005

BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR="/data/home/erisa.kohansal/Workplace/Cascade2/verl-version/Meluxina/IF_RL_Fraction/new/checkpoints/merged_checkpoints"
OUT_DIR="${MODEL_DIR}/eval/ifeval"
LOG_DIR="${MODEL_DIR}/logs/ifeval"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

# echo "============================================================"
# echo "Evaluating $BASE_MODEL"
# echo "Output path: $OUT_DIR"
# echo "============================================================"

# CUDA_VISIBLE_DEVICES=$GPU_ID uv run vllm serve "$BASE_MODEL" \
#     --served-model-name "$BASE_MODEL" \
#     --dtype auto \
#     --port "$PORT" \
#     --host 0.0.0.0 \
#     --generation-config vllm \
#     --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
#     --max-model-len 4096 \
#     --gpu-memory-utilization 0.35 &

# VLLM_PID=$!

# echo "vLLM PID: $VLLM_PID"

# until curl -s "http://localhost:$PORT/v1/models" > /dev/null; do
#     echo "Waiting for vLLM..."
#     sleep 10
# done

# echo "vLLM is ready. Starting lm-eval..."

# CUDA_VISIBLE_DEVICES=$GPU_ID uv run lm_eval \
#     --model local-chat-completions \
#     --model_args "model=$BASE_MODEL,base_url=http://localhost:$PORT/v1/chat/completions" \
#     --tasks ifeval \
#     --output_path "$OUT_DIR" \
#     --log_samples \
#     --apply_chat_template \
#     --gen_kwargs '{"temperature":0.6,"top_p":0.95,"do_sample":true,"max_gen_toks":2700}' \
#     2>&1 | tee "$LOG_DIR/$BASE_MODEL.log"

# echo "Killing vLLM server..."
# kill "$VLLM_PID"
# wait "$VLLM_PID" 2>/dev/null || true

# sleep 10

for CKPT in $(find "$MODEL_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V); do
    NAME=$(basename "$CKPT")
    MODEL_PATH="$CKPT"
    MODEL_NAME="$NAME"
    OUTPUT_PATH="$OUT_DIR"

    # Extract checkpoint number, e.g. global_step_25 -> 25
    CKPT_NUM=${NAME#global_step_}


    if [ -d "$OUTPUT_PATH/$MODEL_NAME" ]; then
        echo "Skipping $MODEL_NAME, already evaluated."
        continue
    fi

    echo "============================================================"
    echo "Evaluating $MODEL_NAME"
    echo "Model path: $MODEL_PATH"
    echo "Output path: $OUTPUT_PATH"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID uv run vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --dtype auto \
        --port "$PORT" \
        --host 0.0.0.0 \
        --generation-config vllm \
        --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.2 &

        VLLM_PID=$!

        echo "vLLM PID: $VLLM_PID"

        until curl -s "http://localhost:$PORT/v1/models" > /dev/null; do
            echo "Waiting for vLLM..."
            sleep 10
        done

        echo "vLLM is ready. Starting lm-eval..."

    CUDA_VISIBLE_DEVICES=$GPU_ID uv run lm_eval \
        --model local-chat-completions \
        --model_args "model=$MODEL_NAME,base_url=http://localhost:$PORT/v1/chat/completions" \
        --tasks ifeval \
        --output_path "$OUTPUT_PATH" \
        --log_samples \
        --apply_chat_template \
        --gen_kwargs '{"temperature":0.0,"do_sample":false,"max_gen_toks":2700}' \
        2>&1 | tee "$LOG_DIR/$MODEL_NAME.log"

        echo "Killing vLLM server..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true

        sleep 10

    done

echo "All evaluations finished."

# 2026-07-25:02:03:38 INFO     [loggers.evaluation_tracker:119] Saving per-task samples to /data/home/erisa.kohansal/Workplace/Cascade2/verl-version/Meluxina/IF_RL_Fraction/with_ds/checkpoints/merged_checkpoints/eval/ifeval/global_step_80/*.jsonl
# local-chat-completions ({'model': 'global_step_80', 'base_url': 'http://localhost:8005/v1/chat/completions'}), gen_kwargs: ({'temperature': 0.0, 'do_sample': False, 'max_gen_toks': 2700}), limit: None, num_fewshot: None, batch_size: 1
# |Tasks |Version|Filter|n-shot|        Metric         |   |Value |   |Stderr|
# |------|------:|------|-----:|-----------------------|---|-----:|---|------|
# |ifeval|      4|none  |     0|inst_level_loose_acc   |↑  |0.7506|±  |   N/A|
# |      |       |none  |     0|inst_level_strict_acc  |↑  |0.7290|±  |   N/A|
# |      |       |none  |     0|prompt_level_loose_acc |↑  |0.6451|±  |0.0206|
# |      |       |none  |     0|prompt_level_strict_acc|↑  |0.6192|±  |0.0209|