CHECKPOINT_ROOT="/mnt/tier2/project/p201382/erisa/IF_RL_dragon/checkpoints/dragon_ifrl/ifrl_qwen25_1nodes"
MERGED_ROOT="${CHECKPOINT_ROOT}/merged_checkpoints"

mkdir -p "$MERGED_ROOT"

for checkpoint in "$CHECKPOINT_ROOT"/global_step_*; do
    step=$(basename "$checkpoint")
    actor_dir="$checkpoint/actor"
    target_dir="$MERGED_ROOT/$step"

    if [ ! -d "$actor_dir" ]; then
        echo "Skipping $step: actor directory not found"
        continue
    fi

    echo "Merging $step..."

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$target_dir"

    if [ $? -eq 0 ]; then
        echo "Successfully merged $step"
    else
        echo "ERROR: merge failed for $step"
    fi
done