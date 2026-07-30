CHECKPOINT_ROOT="/mnt/tier2/project/p201382/erisa/IF_RL_Fraction/new/checkpoints"
MERGED_ROOT="${CHECKPOINT_ROOT}/merged_checkpoints"

# Range of global steps to merge (inclusive)
START_STEP=10
END_STEP=180

mkdir -p "$MERGED_ROOT"

for checkpoint in "$CHECKPOINT_ROOT"/global_step_*; do
    step=$(basename "$checkpoint")
    # extract numeric part from name like global_step_123
    if [[ "$step" =~ global_step_([0-9]+) ]]; then
        num=${BASH_REMATCH[1]}
    else
        echo "Skipping $step: not a global_step pattern"
        continue
    fi

    # Only process steps within the requested range
    if [ "$num" -lt "$START_STEP" ] || [ "$num" -gt "$END_STEP" ]; then
        echo "Skipping $step: step $num outside desired range $START_STEP-$END_STEP"
        continue
    fi

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