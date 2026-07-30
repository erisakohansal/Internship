# AGENTS.md

## 1. Project overview

This repository contains work related to reproducing parts of the NVIDIA Nemotron Cascade 2 reinforcement-learning pipeline.

The current priority is debugging the Instruction-Following RL stage implemented with VERL.

The same general task previously trained successfully with TRL, but the VERL implementation produces substantially worse evaluation results. The VERL code now runs end-to-end, so the goal is no longer merely to make it execute. The goal is to determine, with controlled comparisons and concrete evidence, why its learning behavior differs from the TRL implementation.

Do not assume the problem comes from a single configuration flag. Trace the complete training pipeline:

1. Dataset formatting.
2. Prompt construction and chat templating.
3. vLLM generation.
4. Response tokenization and decoding.
5. Instruction-following verification.
6. Overlong-response handling.
7. Reward placement.
8. Group construction for GRPO.
9. Advantage calculation.
10. PPO loss computation.
11. Rollout/training log-probability consistency.
12. Optimizer steps and effective batch semantics.
13. Validation generation and evaluation.

The debugging process must distinguish:

- confirmed bugs;
- configuration differences;
- expected framework differences;
- untested hypotheses;
- metrics that are merely named confusingly.

Do not repeatedly suggest previously eliminated hypotheses without new evidence.

---

## 2. User and communication preferences

The primary developer is Erisa.

She is working on an internship project involving LLM post-training, RLVR, agentic RL, Nemotron Cascade 2, TRL, and VERL.

When helping:

- Be technically precise.
- Explain conclusions in plain language.
- Avoid overwhelming her with many speculative changes at once.
- Prefer controlled experiments where only one relevant variable changes.
- State explicitly whether something is:
  - confirmed;
  - strongly suspected;
  - possible but unverified;
  - ruled out.
- Do not imply that a configuration is wrong merely because it differs from another framework.
- Check the exact installed VERL version before relying on current upstream behavior.
- When suggesting shell commands, explain:
  - what the command does;
  - what important flags mean;
  - whether it only reads information or modifies files/system state.
- For dangerous or destructive commands, ask before running them.
- Prefer concise progress reports, but give detailed explanations for important findings.
- When correcting French messages, preserve the original wording as much as possible unless a rewrite is requested.

The user may be frustrated because this issue has taken several weeks. Do not interpret frustration as permission to make broad or uncontrolled changes.

---

## 3. Current primary investigation

The primary question is:

> Why does the VERL implementation of Nemotron Cascade 2 IF-RL perform substantially worse than the TRL implementation, even though the VERL run executes and its training/validation rewards increase?

The supervisor tested a separate reward library or integration with VERL and reportedly obtained better results. This comparison may be useful, but the implementations and experimental conditions must be verified before drawing conclusions.

The debugging target is not merely “make reward increase.” The target is to explain the discrepancy in downstream evaluation performance between the TRL and VERL runs.

---

## 4. Repository orientation

Before modifying anything:

1. Find repository-level and nested `AGENTS.md` files.
2. Inspect the repository structure.
3. Identify:
   - launch scripts;
   - dataset conversion scripts;
   - custom agent loops;
   - reward functions;
   - custom reward managers;
   - VERL configuration files;
   - TRL training scripts;
   - evaluation scripts;
   - rollout artifacts;
   - validation generations;
   - training logs;
   - checkpoint directories.
4. Inspect the current Git status.
5. Preserve all unrelated user changes.

Use `rg --files` and `rg` for file discovery and text search.

Do not assume paths mentioned in this document still exist. Resolve them from the actual checkout.

Likely relevant names include:

- `IF-RL`
- `if_rl`
- `reward.py`
- `dataset.py`
- `test.sh`
- `verl.trainer.main_ppo`
- `DAPORewardManagerNemotron`
- `dapo_overlong_penalty`
- `if_reward_fn`
- `verifiable_instructions`
- `instructions_registry`
- `Validation_ifrl`
- `rollouts`
- `output-binary-test.txt`

Search for these names before asking the user where files are.

---

## 5. Environment context

The work has been run on MeluXina HPC.

Previously working module stack:

- `env/release/2024.1`
- `Python/3.12.3-GCCcore-13.3.0`
- `CMake/3.29.3-GCCcore-13.3.0`
- `CUDA/12.6.0`

Previously discussed Python packages:

- PyTorch approximately `2.12.1+cu126`
- vLLM approximately `0.11.0`
- VERL target approximately `0.8.0`
- flash-attn approximately `2.6.3`

These versions may have changed. Verify them from the active environment and repository instead of treating them as guaranteed.

Important HPC considerations:

- Login nodes and GPU compute nodes have different purposes.
- Do not launch GPU-intensive training on a login node.
- The user may operate through Slurm.
- Home-directory inode quota has caused installation failures before.
- Some environments were moved to project scratch.
- Required modules must be loaded before activating or using the virtual environment.
- A missing `libpython3.12.so.1.0` previously resulted from an incorrect module/venv setup.
- Avoid unnecessary package reinstalls.
- Do not modify the shared supervisor environment.
- Never assume a shared Claude/Codex installation or authentication can be reused safely without verifying ownership and configuration.

---

## 6. Current VERL IF-RL configuration

A previously shared launch configuration contained the following important settings:

```bash
data.train_batch_size=128
data.max_prompt_length=5000
data.max_response_length=4000

actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=True

actor_rollout_ref.actor.strategy=fsdp
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.clip_ratio_low=0.2
actor_rollout_ref.actor.clip_ratio_high=0.28
actor_rollout_ref.actor.loss_agg_mode=token-mean
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
actor_rollout_ref.actor.ppo_epochs=1
actor_rollout_ref.actor.entropy_coeff=0.0
actor_rollout_ref.actor.optim.lr=3e-6
actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
actor_rollout_ref.actor.optim.weight_decay=0.0

actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.top_k=-1
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.do_sample=True
actor_rollout_ref.rollout.val_kwargs.temperature=0.0
actor_rollout_ref.rollout.val_kwargs.do_sample=False
actor_rollout_ref.rollout.n=16
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
actor_rollout_ref.rollout.calculate_log_probs=False

algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
algorithm.rollout_correction.rollout_is=null
algorithm.rollout_correction.rollout_rs=null
algorithm.rollout_correction.bypass_mode=False
algorithm.norm_adv_by_std_in_grpo=True

reward.reward_manager.source=register
reward.reward_manager.name=dapo_overlong_penalty
reward.reward_kwargs.max_resp_len=4000
reward.reward_kwargs.overlong_penalty.enable=True
reward.reward_kwargs.overlong_penalty.log=True

trainer.total_training_steps=180
trainer.val_before_train=True
trainer.test_freq=10
trainer.save_freq=10
trainer.nnodes=1
trainer.n_gpus_per_node=4