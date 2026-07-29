#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH -p gpu
#SBATCH -A p201382
#SBATCH -q default
#SBATCH --time 5:00:00
#SBATCH --job-name=calculator
#SBATCH --output=slurm/logs/calculator_%j.out
#SBATCH --error=slurm/logs/calculator_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=erisa.kohansal@linguacustodia.com

set -xeuo pipefail

module load env/staging/2024.1
module load CUDA-Python/12.6.0-gfbf-2024a-CUDA-12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
module load CMake/3.29.3-GCCcore-13.3.0
module load Ninja/1.12.1-GCCcore-13.3.0

source /project/scratch/p201382/erisa/Internship/Cascade2/verl-version/.venv/bin/activate
PWD="/project/scratch/p201382/erisa/Internship/multi"
cd $PWD

export PYTHONPATH=""
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

REWARD_PATH="$PWD/reward.py"
echo "Using reward file: $REWARD_PATH"
test -f "$REWARD_PATH" || { echo "Reward file not found"; exit 1; }
CHECKPOINT_PATH="/project/home/p201382/erisa/Multi-turn/checkpoints"
TRAIN_FILE="$PWD/gsm8k/train.parquet" 
TEST_FILE="$PWD/gsm8k/test.parquet" 
MAX_PROMPT_LEN=5000
MAX_RESPONSE_LEN=4000                                             
MAX_MODEL_LEN=$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))  
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct} 

# example : https://github.com/verl-project/verl/blob/v0.5.0/examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh

python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size=128 \
  data.prompt_key=prompt \
  data.max_prompt_length=${MAX_PROMPT_LEN} \
  data.max_response_length=${MAX_RESPONSE_LEN} \
  data.return_raw_chat=True \
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
  actor_rollout_ref.actor.use_dynamic_bsz=True \
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
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.function_tool_path="$PWD/tool.py" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=3 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=256 \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=strict \
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
  reward.custom_reward_function.name=calculator_reward_fn \
  reward.reward_manager.source=register \
  reward.reward_manager.name=dapo \
  trainer.total_training_steps=50 \
  trainer.save_freq=5 \
  trainer.val_before_train=True \
  trainer.test_freq=5 \
  trainer.default_local_dir="$CHECKPOINT_PATH" \
  trainer.project_name=multiturn \
  trainer.experiment_name=multiturn \
  trainer.logger='["wandb"]' \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.validation_data_dir="$CHECKPOINT_PATH/Validation" \
  trainer.log_val_generations=8 \
  trainer.resume_mode=disable \
  "$@"