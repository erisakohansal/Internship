#!/bin/bash

set -e

GPU_ID=0
PORT=8002

BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR="IF-RL-Binary_checkpoints"
OUT_DIR="eval/ifeval"
LOG_DIR="logs/ifeval"

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

# local-chat-completions ({'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'base_url': 'http://localhost:8002/v1/chat/completions'}), gen_kwargs: ({'temperature': 0.6, 'top_p': 0.95, 'do_sample': True, 'max_gen_toks': 2700}), limit: None, num_fewshot: None, batch_size: 1
# |Tasks |Version|Filter|n-shot|        Metric         |   |Value |   |Stderr|
# |------|------:|------|-----:|-----------------------|---|-----:|---|------|
# |ifeval|      4|none  |     0|inst_level_loose_acc   |↑  |0.5755|±  |   N/A|
# |      |       |none  |     0|inst_level_strict_acc  |↑  |0.5420|±  |   N/A|
# |      |       |none  |     0|prompt_level_loose_acc |↑  |0.4750|±  |0.0215|
# |      |       |none  |     0|prompt_level_strict_acc|↑  |0.4418|±  |0.0214|

for CKPT in $(find "$MODEL_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V); do
    NAME=$(basename "$CKPT")
    MODEL_PATH="$CKPT"
    MODEL_NAME="$NAME"
    OUTPUT_PATH="$OUT_DIR"

    # Extract checkpoint number, e.g. checkpoint-25 -> 25
    CKPT_NUM=${NAME#checkpoint-}

    # Evaluate only every 5 checkpoints
    if (( CKPT_NUM % 5 != 0 )); then
        echo "Skipping $NAME because it is not a multiple of 5."
        continue
    fi

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
        --gpu-memory-utilization 0.17 &

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
        --gen_kwargs '{"temperature":0.6,"top_p":0.95,"do_sample":true,"max_gen_toks":2700}' \
        2>&1 | tee "$LOG_DIR/$MODEL_NAME.log"

        echo "Killing vLLM server..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true

        sleep 10

    done

echo "All evaluations finished."


# (EngineCore pid=1172166) INFO 06-05 17:51:34 [gpu_model_runner.py:5746] Graph capturing finished in 7 secs, took 0.39 GiB
# (EngineCore pid=1172166) INFO 06-05 17:51:34 [gpu_worker.py:617] CUDA graph pool memory: 0.39 GiB (actual), 0.39 GiB (estimated), difference: 0.0 GiB (0.0%).
# (EngineCore pid=1172166) INFO 06-05 17:51:34 [core.py:281] init engine (profile, create kv cache, warmup model) took 12.25 seconds
# (EngineCore pid=1172166) The tokenizer you are loading from 'IF-RL-Binary_checkpoints/checkpoint-2' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
# (EngineCore pid=1172166) INFO 06-05 17:51:34 [vllm.py:754] Asynchronous scheduling is enabled.
# (APIServer pid=1171865) INFO 06-05 17:51:34 [api_server.py:576] Supported tasks: ['generate']
# (APIServer pid=1171865) WARNING 06-05 17:51:35 [model.py:1376] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
# (APIServer pid=1171865) The tokenizer you are loading from 'IF-RL-Binary_checkpoints/checkpoint-2' with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e. This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
# (APIServer pid=1171865) INFO 06-05 17:51:35 [hf.py:320] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.