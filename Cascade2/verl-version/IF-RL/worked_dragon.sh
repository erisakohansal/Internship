#!/bin/bash -l
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH -p gpu
#SBATCH -A p201382
#SBATCH -q default
#SBATCH --time 8:00:00
#SBATCH --job-name=ifrl_q25
#SBATCH --output=slurm/logs/ifrl_qwen25_%j.out
#SBATCH --error=slurm/logs/ifrl_qwen25_%j.err

# ---------------------------------------------------------------------------
# Nemotron-Cascade instruction-following RL — Qwen2.5 (pure softmax), single turn.
# Qwen2.5 has no GDN linear attention, so the Qwen3.5 packing/SP hazards don't
# apply: run the FAST recipe (SP + dynamic bsz on stock FSDP). Reward is computed
# inside the agent (examples.ifrl_cascade2.agent:if_single_turn -> reward.py's
# if_reward_fn): fraction of the row's verifiable instructions the answer follows.
#
# REQUIRES in the venv: the reward's checkers must import —
#   verifiable_instructions (instructions_registry) and the dataset module
#   (FormatData). reward.py imports them at load time; without them the agent
#   import fails. Install/point PYTHONPATH at the Nemotron-Cascade IF-eval code.
# DATA: the parquets live in this example folder (IF-RL-{train,test}.parquet).
# ---------------------------------------------------------------------------

set -xeuo pipefail

module load env/staging/2024.1
module load CUDA-Python/12.6.0-gfbf-2024a-CUDA-12.6.0
module load NCCL/2.22.3-GCCcore-13.3.0-CUDA-12.6.0
module load CMake/3.29.3-GCCcore-13.3.0
module load Ninja/1.12.1-GCCcore-13.3.0

source /project/scratch/p201382/erisa/Internship/Cascade2/verl-version/.venv/bin/activate
cd /project/scratch/p201382/erisa/Internship/Cascade2/verl-version/dragon-agentic
export PYTHONPATH=""

pip install -v --no-deps .

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_HCA=mlx5*
export NCCL_IB_CUDA_SUPPORT=1

export TRAIN_FILE=${TRAIN_FILE:-/project/scratch/p201382/erisa/Internship/Cascade2/verl-version/IF-RL/IF-RL-binary-train.parquet}
export TEST_FILE=${TEST_FILE:-/project/scratch/p201382/erisa/Internship/Cascade2/verl-version/IF-RL/IF-RL-binary-test.parquet}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
N_GPUS=${N_GPUS:-$SLURM_GPUS_PER_NODE}
NNODES=${NNODES:-$SLURM_NNODES}
RAY_PORT=6379

PROJECT=${PROJECT:-dragon_ifrl}
EXPERIMENT=${EXPERIMENT:-ifrl_qwen25_${NNODES}nodes}
RUN_DIR=${RUN_DIR:-checkpoints/$PROJECT/$EXPERIMENT}

# Node info
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$RAY_PORT

echo "Starting Ray head on ${head_node} (${head_node_ip})..."
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address="$head_node_ip" --port=$RAY_PORT \
  --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$N_GPUS" --block &
sleep 10

for ((i = 1; i < NNODES; i++)); do
  node=${nodes_array[$i]}
  echo "Starting Ray worker on ${node}..."
  srun --overlap --nodes=1 --ntasks=1 -w "$node" \
    ray start --address="${head_node_ip}:${RAY_PORT}" \
    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$N_GPUS" --block &
done
sleep 10

export RAY_ADDRESS="${head_node_ip}:${RAY_PORT}"

# --- Sizing (Qwen2.5 softmax: SP + dynamic bsz are safe, so run fast) ------------
SP_SIZE=4
MAX_PROMPT_LEN=5000
MAX_RESPONSE_LEN=4000                                           # dataset max_completion_length ~4000
MAX_MODEL_LEN=$(( MAX_PROMPT_LEN + MAX_RESPONSE_LEN ))             # 5120
SEQ_PER_GPU=$(( MAX_MODEL_LEN / SP_SIZE ))                        # 1280 (one seq after SP shard)
PACK=${PACK:-4}
MAX_TOKEN_LEN_PER_GPU=$(( SEQ_PER_GPU * PACK ))                   # 5120
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$(( MAX_TOKEN_LEN_PER_GPU * 2 ))   # 10240
OFFLOAD=${OFFLOAD:-True}                                          # colocated 40GB; clears GPU for vLLM
GPU_MEM=${GPU_MEM:-0.6}

TOTAL_GPUS=$(( N_GPUS * NNODES ))
DATA_TRAIN_BATCH=128
PPO_MINI_BATCH=128
ROLLOUT_N=16

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=null \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${DATA_TRAIN_BATCH} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LEN} \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LEN} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    trainer.v1.trainer_mode=sync \
    actor_rollout_ref.rollout.agent.default_agent_loop=dragon \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/dragon_agent_loop.yaml \
    +actor_rollout_ref.rollout.custom.agent='examples.ifrl_cascade2.agent:if_single_turn' \
    +actor_rollout_ref.rollout.custom.reward='examples.ifrl_cascade2.reward:reward' \
    +actor_rollout_ref.rollout.custom.max_turns=1 \
    +actor_rollout_ref.rollout.custom.enable_thinking=False \
    trainer.rollout_data_dir="${RUN_DIR}/rollouts" \
    trainer.log_val_generations=8 \
    trainer.critic_warmup=0 \
    trainer.logger=[console,wandb] \
    trainer.project_name=if_rl_verl_binary \
    trainer.experiment_name=if_rl_verl_binary \
    trainer.default_local_dir="${RUN_DIR}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=25 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    trainer.validation_data_dir="${RUN_DIR}/Validation_ifrl" \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    "$@" 2>&1 | tee -a output-binary-test.txt