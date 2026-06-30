#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

PWD="/data/home/erisa.kohansal/Workplace/Cascade2/verl-version"
REWARD_PATH="$PWD/reward.py"
echo "Using reward file: $REWARD_PATH"
test -f "$REWARD_PATH" || { echo "Reward file not found"; exit 1; }
CHECKPOINT_PATH="$PWD/if_rl_verl_binary_checkpoints"

.venv/bin/python3 dataset.py 2>&1 | tee output.txt

.venv/bin/python3 -m verl.trainer.main_ppo \
  data.train_files=IF-RL-train.parquet \
  data.val_files=IF-RL-test.parquet \
  data.train_batch_size=128 \
  data.prompt_key=prompt \
  data.max_prompt_length=5000 \
  data.max_response_length=4000 \
  actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.15 \
  actor_rollout_ref.rollout.max_model_len=9000 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  algorithm.rollout_correction.rollout_is=null \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.bypass_mode=false \
  algorithm.rollout_correction.loss_type=ppo_clip \
  +algorithm.filter_groups.enable=true \
  +algorithm.filter_groups.metric=acc \
  +algorithm.filter_groups.max_num_gen_batches=10 \
  reward.reward_manager.name=dapo \
  +reward.reward_kwargs.max_resp_len=4000 \
  +reward.reward_kwargs.overlong_buffer_cfg.enable=true \
  +reward.reward_kwargs.overlong_buffer_cfg.len=512 \
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
  +reward.reward_kwargs.overlong_buffer_cfg.log=true \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=if_reward_fn \
  trainer.total_training_steps=180 \
  trainer.save_freq=10 \
  trainer.val_before_train=true \
  trainer.test_freq=10 \
  trainer.default_local_dir=if_rl_verl_binary_checkpoints \
  trainer.project_name=if_rl_verl_binary \
  trainer.experiment_name=if_rl_verl_binary \
  trainer.logger='["wandb"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  2>&1 | tee -a output.txt