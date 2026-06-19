# my_grpo_config.yaml
defaults:
  - ppo_trainer    # inherits ALL defaults from verl's base config
  - _self_         # your overrides come after

# --- only the things you actually change ---

data:
  train_files: ./data/gsm8k_train.parquet
  val_files: ./data/gsm8k_val.parquet
  prompt_key: prompt
  max_prompt_length: 512
  max_response_length: 1024
  train_batch_size: 256

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-1.5B
  actor:
    loss_type: grpo           # or dapo
    ppo_mini_batch_size: 256
    ppo_micro_batch_size_per_gpu: 8
  rollout:
    temperature: 1.0
    n: 8                      # number of samples per prompt (the G in GRPO)

algorithm:
  adv_estimator: grpo

reward_model:
  enable: false               # using custom reward fn instead

custom_reward_function:
  path: ./reward_fn.py
  name: compute_reward

trainer:
  max_steps: 1000
  project_name: my_project
  experiment_name: qwen2.5_grpo_gsm8k
  logger: ['wandb']
  n_gpus_per_node: 4