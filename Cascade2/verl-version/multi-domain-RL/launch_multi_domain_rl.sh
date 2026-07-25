#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH -p gpu
#SBATCH -A p201382
#SBATCH -q default
#SBATCH --time 10:00:00
#SBATCH --job-name=multi_domain_rl
#SBATCH --output=slurm/logs/multi_domain_%j.out
#SBATCH --error=slurm/logs/multi_domain_rl_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=erisa.kohansal@linguacustodia.com


set -xeuo pipefail

module load env/staging/2024.1
module load CUDA-Python/12.6.0-gfbf-2024a-CUDA-12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
module load CMake/3.29.3-GCCcore-13.3.0
module load Ninja/1.12.1-GCCcore-13.3.0

source /project/scratch/p201382/erisa/Internship/Cascade2/verl-version/.venv/bin/activate
PWD="/project/scratch/p201382/erisa/Internship/Cascade2/verl-version/multi-domain-RL"
cd $PWD

export PYTHONPATH=""
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

REWARD_PATH="$PWD/reward.py"
echo "Using reward file: $REWARD_PATH"
test -f "$REWARD_PATH" || { echo "Reward file not found"; exit 1; }
CHECKPOINT_PATH="/project/home/p201382/erisa/Multi-Domain-RL/checkpoints"
TRAIN_FILE="$PWD/multi-domain-train.parquet"
TEST_FILE="$PWD/multi-domain-test.parquet"
MAX_PROMPT_LEN=5000
MAX_RESPONSE_LEN=4000                                             
MAX_MODEL_LEN=$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))  
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct} 

python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size=128 \
  data.prompt_key=prompt \
  data.max_prompt_length=${MAX_PROMPT_LEN} \
  data.max_response_length=${MAX_RESPONSE_LEN} \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
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
  actor_rollout_ref.actor.optim.override_optimizer_config='{fused: true}' \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is=null \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.bypass_mode=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  +algorithm.filter_groups.enable=True \
  +algorithm.filter_groups.metric=acc \
  +algorithm.filter_groups.max_num_gen_batches=10 \
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=if_reward_fn \
  reward.reward_manager.source=register \
  reward.reward_manager.name=dapo_overlong_penalty \
  +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LEN} \
  +reward.reward_kwargs.overlong_penalty.enable=True \
  +reward.reward_kwargs.overlong_penalty.log=True \
  trainer.total_training_steps=70 \
  trainer.save_freq=10 \
  trainer.val_before_train=True \
  trainer.test_freq=10 \
  trainer.default_local_dir="$CHECKPOINT_PATH" \
  trainer.project_name=multi_domain_rl \
  trainer.experiment_name=multi_domain_rl \
  trainer.logger='["wandb"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.validation_data_dir="$CHECKPOINT_PATH/Validation" \
  trainer.log_val_generations=8 \
  trainer.resume_mode=disable \
  "$@"