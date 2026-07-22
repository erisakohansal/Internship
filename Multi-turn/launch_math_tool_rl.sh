#!/bin/bash

export TOKENIZERS_PARALLELISM=false

# Qwen2.5-Instruct's tokenizer already ships a Hermes-style tool-calling
# chat template (tools/<tool_call>/<tool_response> rendering) by default, so
# it is not overridden here. chat_template.j2 in this folder is a copy of
# that same style kept for reference (e.g. if this pipeline is ever pointed
# at a base/non-instruct model that lacks a native tool-calling template).
PWD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_PATH="$PWD_DIR/reward.py"
TOOL_PATH="$PWD_DIR/tools/calculator_tool.py"
TRAIN_FILE="$PWD_DIR/math_calculator_train.parquet"
VAL_FILE="$PWD_DIR/math_calculator_val.parquet"

test -f "$REWARD_PATH" || { echo "Reward file not found: $REWARD_PATH"; exit 1; }
test -f "$TOOL_PATH" || { echo "Tool file not found: $TOOL_PATH"; exit 1; }
test -f "$TRAIN_FILE" || { echo "Train parquet not found -- run prepare_data.py first"; exit 1; }
test -f "$VAL_FILE" || { echo "Val parquet not found -- run prepare_data.py first"; exit 1; }

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PWD_DIR/math_tool_rl_checkpoints}"

# Override TOTAL_STEPS=3 (or similar) for the tool-firing smoke test / short
# trial run described in the plan, before committing to the full run.
TOTAL_STEPS="${TOTAL_STEPS:-200}"

.venv/bin/python -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=128 \
  data.prompt_key=prompt \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.tie_word_embeddings=false \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_model_len=3200 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.function_tool_path="$TOOL_PATH" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=256 \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=strict \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  +algorithm.filter_groups.enable=True \
  +algorithm.filter_groups.metric=acc \
  +algorithm.filter_groups.max_num_gen_batches=10 \
  reward.reward_manager.name=naive \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=math_tool_reward_fn \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.save_freq=10 \
  trainer.val_before_train=True \
  trainer.test_freq=10 \
  trainer.default_local_dir="$CHECKPOINT_PATH" \
  trainer.project_name=math_tool_rl \
  trainer.experiment_name=math_tool_rl \
  trainer.logger='["wandb"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  2>&1 | tee -a "$PWD_DIR/output.txt"
