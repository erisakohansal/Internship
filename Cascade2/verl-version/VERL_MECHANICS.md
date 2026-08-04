# VERL Mechanics — Engineering Deep Dive

Source-grounded reference for the VERL fork vendored at [`verl/`](verl/) (`verl==0.9.0.dev0`, commit `9481350e`). Every claim below is either traced to an exact file:line in that checkout (linked — click through or open the path relative to this file's own directory, `Cascade2/verl-version/`), quoted from a code comment/docstring, or attributed to a specific paper — not inferred from general RL knowledge unless explicitly marked `[general RL knowledge]`. Where a number looks like a threshold, its source is labeled; several thresholds discussed earlier in this project's conversation came from prior-notes/supervisor discussion, not from VERL itself, and are labeled accordingly.

**Link convention:** all links are relative to this file (`Cascade2/verl-version/VERL_MECHANICS.md`). VERL package files live under [`verl/verl/`](verl/verl/) (the outer `verl/` is the git checkout, the inner `verl/` is the Python package) — e.g. [`verl/verl/trainer/ppo/core_algos.py`](verl/verl/trainer/ppo/core_algos.py). Docs live under [`verl/docs/`](verl/docs/). Project-specific files (reward functions, launch scripts) are linked relative to this directory directly, e.g. [`multi-domain-RL/reward.py`](multi-domain-RL/reward.py).

**Read this before anything else:**

1. **Two trainer generations coexist in this checkout.** [`verl/verl/trainer/ppo/ray_trainer.py`](verl/verl/trainer/ppo/ray_trainer.py)`::RayPPOTrainer` (the "legacy" trainer) is decorated `@deprecated` ("will be removed in v0.9.0"). `config.trainer.use_v1` defaults to `true` ([`ppo_trainer.yaml:201`](verl/verl/trainer/config/ppo_trainer.yaml#L201)), which selects [`verl/verl/trainer/ppo/v1/trainer_base.py`](verl/verl/trainer/ppo/v1/trainer_base.py) + [`trainer_sync.py`](verl/verl/trainer/ppo/v1/trainer_sync.py) / [`trainer_colocate_async.py`](verl/verl/trainer/ppo/v1/trainer_colocate_async.py) / [`trainer_separate_async.py`](verl/verl/trainer/ppo/v1/trainer_separate_async.py) — this is the live default path. The core **algorithm math is shared** between both ([`core_algos.py`](verl/verl/trainer/ppo/core_algos.py), [`losses.py`](verl/verl/workers/utils/losses.py), [`metric_utils.py`](verl/verl/trainer/ppo/metric_utils.py), [`rollout_corr_helper.py`](verl/verl/trainer/ppo/rollout_corr_helper.py) are not duplicated per-trainer), so citations into those files are accurate regardless of trainer generation. Citations into `ray_trainer.py` itself describe *pipeline orchestration* faithfully as an algorithm description, but on the live default path that orchestration actually runs from `v1/trainer_base.py`'s `step()`. Each citation below is labeled `[legacy]` or `[v1]` where the distinction matters.
2. **Public docs (verl.readthedocs.io) may describe a different config schema than what's vendored here.** Confirmed concretely: an online doc page describes `rollout_is_level`/`rollout_is_mode`, but this repo's actual schema is `algorithm.rollout_correction.{rollout_is, rollout_is_threshold, rollout_rs, rollout_rs_threshold, bypass_mode, rollout_is_batch_normalize}` ([`verl/verl/trainer/config/algorithm/rollout_correction.yaml`](verl/verl/trainer/config/algorithm/rollout_correction.yaml), and confirmed live in your own `launch.sh` files). **Always check the vendored source/yaml before trusting an online doc page or an LLM's memory of VERL — including this document, if VERL is later upgraded.**
3. **Config values used throughout the worked example are your actual [`multi-domain-RL/launch.sh`](multi-domain-RL/launch.sh)**: `train_batch_size=128`, `rollout.n=16`, `ppo_mini_batch_size=128`, `ppo_epochs=1`, `clip_ratio_low=0.2`/`clip_ratio_high=0.28`, `loss_agg_mode=token-mean`, `algorithm.adv_estimator=grpo`, `norm_adv_by_std_in_grpo=True`, `algorithm.rollout_correction.{rollout_is=null, rollout_rs=null, bypass_mode=False}`, `calculate_log_probs=True`.

---

## Table of Contents

1. [Architecture & Execution Model](#1-architecture--execution-model)
2. [One Batch, End to End](#2-one-batch-end-to-end)
3. [Parameter Reference](#3-parameter-reference)
4. [Metric Rubric](#4-metric-rubric)
5. [Problem → Symptom → Metric → Fix](#5-problem--symptom--metric--fix)
6. [Research Papers Behind VERL's Algorithms](#6-research-papers-behind-verls-algorithms)
7. [Off-Policy Sources — Complete Catalog](#7-off-policy-sources--complete-catalog)
8. [File Map](#8-file-map)

---

## 1. Architecture & Execution Model

### 1.1 Ray worker groups and colocation

Roles are an `Enum`: `Actor, Rollout, ActorRollout, Critic, RefPolicy, RewardModel, ActorRolloutRef, Env, TeacherModel` ([`verl/verl/trainer/ppo/utils.py:27-40`](verl/verl/trainer/ppo/utils.py#L27-L40)). A `RayWorkerGroup` ([`verl/verl/single_controller/ray/base.py`](verl/verl/single_controller/ray/base.py)) is a set of Ray actors managed as one logical unit, placed on GPU bundles organized as a Ray `PlacementGroup` (`RayResourcePool.get_placement_groups()`, [line 131](verl/verl/single_controller/ray/base.py#L131)).

**Colocated (default):** `actor_rollout_ref.hybrid_engine` defaults to `true` ([`ppo_trainer.yaml:53`](verl/verl/trainer/config/ppo_trainer.yaml#L53)). `[v1]` `_init_resource_pool_mgr` ([`trainer_base.py:578-632`](verl/verl/trainer/ppo/v1/trainer_base.py#L578-L632)) places actor, rollout, critic, and reference policy all into one `"global_pool"` sized `[trainer.n_gpus_per_node] × trainer.nnodes`, then `_setup` ([lines 210-221](verl/verl/trainer/ppo/v1/trainer_base.py#L210-L221)) fuses them into one colocated worker class via `create_colocated_worker_cls`/`RayWorkerGroup.spawn()`. **Actor and rollout are literally the same GPU processes** — this is why weight sync in this mode is a same-process CUDA IPC hop, not a network broadcast (§1.4). `[legacy]` [`ray_trainer.py:334`](verl/verl/trainer/ppo/ray_trainer.py#L334) hard-asserts `hybrid_engine == True` — the legacy trainer only ever supports colocation.

**Disaggregated:** `hybrid_engine=False` plus a *separate* `rollout.nnodes`/`rollout.n_gpus_per_node` (distinct from `trainer.nnodes`/`trainer.n_gpus_per_node`). Used by the `separate_async` v1 trainer mode ([`trainer_separate_async.py:51-56`](verl/verl/trainer/ppo/v1/trainer_separate_async.py#L51-L56), asserts `rollout.nnodes > 0` and a real network `checkpoint_engine.backend != "naive"`) and the older `verl.experimental.one_step_off_policy`/`fully_async_policy` recipes ([`verl/verl/experimental/one_step_off_policy/`](verl/verl/experimental/one_step_off_policy/), [`verl/verl/experimental/fully_async_policy/`](verl/verl/experimental/fully_async_policy/)).

Colocated vs. disaggregated is a **resource-pool configuration choice**, not a different code path for the model itself. (Separate terminology overload to be aware of: "prefill/decode disaggregation" inside a single vLLM/SGLang replica, e.g. [`verl/verl/workers/rollout/sglang_rollout/sglang_pd_replica.py`](verl/verl/workers/rollout/sglang_rollout/sglang_pd_replica.py), is an unrelated axis — splitting one inference engine's prefill and decode across GPUs, not actor/rollout placement.)

### 1.2 Rollout generation — async is the only supported mode, not just the default

`RolloutConfig.mode` defaults to `"async"` ([`verl/verl/workers/config/rollout.py:159`](verl/verl/workers/config/rollout.py#L159)); the validator explicitly **rejects** `mode="sync"` ([lines 277-285](verl/verl/workers/config/rollout.py#L277-L285), raises with the message *"Rollout mode 'sync' has been removed. Please set `actor_rollout_ref.rollout.mode=async`"*). The old synchronous vLLM SPMD path is gone: `ServerAdapter.generate_sequences` unconditionally raises `NotImplementedError` ([`verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py:252-268`](verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L252-L268), citing *"The vLLM SPMD mode was retired in PR #4411"*).

The engine underneath is vLLM's native async engine: [`verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:35`](verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L35) imports `from vllm.v1.engine.async_llm import AsyncLLM`, wrapped in an OpenAI-compatible HTTP server (`vLLMHttpServer`, [line 86](verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L86), constructed at [line 412](verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L412)). Requests are scheduled and stepped as they arrive with continuous batching, not one blocking `LLM.generate(batch)` call.

**Dispatch from the AgentLoop layer is concurrent per-sample.** [`verl/verl/experimental/agent_loop/agent_loop.py`](verl/verl/experimental/agent_loop/agent_loop.py)`::AgentLoopWorker.generate_sequences` ([line 567](verl/verl/experimental/agent_loop/agent_loop.py#L567)) creates one `asyncio.create_task` per sample ([lines 649-653](verl/verl/experimental/agent_loop/agent_loop.py#L649-L653)) and awaits them together via `asyncio.gather` ([line 654](verl/verl/experimental/agent_loop/agent_loop.py#L654)) — every prompt in a generation batch is an independent coroutine hitting the async HTTP server concurrently. `AgentLoopManager.generate_sequences` ([lines 1204-1235](verl/verl/experimental/agent_loop/agent_loop.py#L1204-L1235)) fans this out further across multiple `AgentLoopWorker` Ray actors.

**But the caller still awaits the whole call before proceeding.** This "async" describes *how requests inside one generation call are scheduled* (concurrently, overlapping with each other), not whether the trainer moves on to the next pipeline stage before generation for the current batch finishes — that's a separate question, §1.3–1.4.

### 1.3 The training step itself — standard synchronous data-parallel PPO

[`verl/verl/workers/engine/base.py`](verl/verl/workers/engine/base.py)`::BaseEngine.train_batch` ([lines 113-130](verl/verl/workers/engine/base.py#L113-L130)):
```python
self.optimizer_zero_grad()
outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
grad_norm = self.optimizer_step()
```
FSDP implementation, [`verl/verl/workers/engine/fsdp/transformer_impl.py`](verl/verl/workers/engine/fsdp/transformer_impl.py)`::forward_backward_batch` ([lines 638-675](verl/verl/workers/engine/fsdp/transformer_impl.py#L638-L675)): an explicit `torch.distributed.all_reduce` ([lines 644-646](verl/verl/workers/engine/fsdp/transformer_impl.py#L644-L646)) reduces the **global valid-token count** for loss normalization across the DP group — this is *not* gradient synchronization. Gradient sync itself happens inside FSDP's own backward pass (reduce-scatter, `loss.backward()`, [line 670](verl/verl/workers/engine/fsdp/transformer_impl.py#L670)) — standard FSDP mechanics, not something VERL orchestrates by hand. `optimizer_step()` ([lines 686-732](verl/verl/workers/engine/fsdp/transformer_impl.py#L686-L732)) clips gradients, **skips the update entirely if the grad norm is non-finite** (see §5), otherwise calls `optimizer.step()` once, synchronously, across all DP ranks.

**Multiple optimizer steps per rollout batch, standard PPO mini-batching.** [`verl/verl/workers/engine_workers.py`](verl/verl/workers/engine_workers.py)`::train_mini_batch` ([lines 234-302](verl/verl/workers/engine_workers.py#L234-L302)) loops `for epochs (ppo_epochs) × mini-batches` and calls `train_batch` once per mini-batch — one call to `_update_actor` triggers `ppo_epochs × num_mini_batches` synchronous forward/backward/optimizer-step cycles against the *same fixed* rollout batch (old_log_prob and advantages are computed once and held fixed across this inner loop — `[legacy]` comment at [`ray_trainer.py:1534-1536`](verl/verl/trainer/ppo/ray_trainer.py#L1534-L1536): *"π_old computed once per data batch, serves as stable reference during mini-batch updates"*). Dispatch is fully blocking: `RayWorkerGroup`'s `func_generator` wrapper calls `ray.get(output)` when `blocking=True` ([`single_controller/ray/base.py:49-67`](verl/verl/single_controller/ray/base.py#L49-L67), the default for `update_actor`), so the single driver process waits for **all** DP ranks to finish their mini-batch loop.

`[v1]` The default trainer's `step()` ([`trainer_base.py:412-464`](verl/verl/trainer/ppo/v1/trainer_base.py#L412-L464)) is a strictly sequential pipeline: sample from replay buffer (blocks/polls) → reward → balance batch → `old_log_prob` → `ref_log_prob` → values → advantage → `_update_critic` → `_update_actor`. Each stage is a blocking RPC; no stage overlaps another in the default (`sync`) trainer mode. **There is no asynchronous/pipelined training path anywhere in this codebase** — what varies across trainer modes is whether *generation of the next batch* overlaps with *training on the current batch*, never whether the forward/backward/optimizer-step is itself asynchronous.

### 1.4 Weight synchronization and the staleness spectrum

Abstraction: [`verl/verl/checkpoint_engine/base.py`](verl/verl/checkpoint_engine/base.py)`::CheckpointEngine` ([lines 96-200](verl/verl/checkpoint_engine/base.py#L96-L200)), concrete backends via `CheckpointEngineRegistry`: `"naive"` (colocated CUDA IPC, `ColocatedCheckpointEngine`, [line 220](verl/verl/checkpoint_engine/base.py#L220)) vs. network backends ([`nccl_checkpoint_engine.py`](verl/verl/checkpoint_engine/nccl_checkpoint_engine.py), `nixl_checkpoint_engine.py`, `mooncake_checkpoint_engine.py`, `hccl_checkpoint_engine.py`, `kimi_checkpoint_engine.py`, all siblings in [`verl/verl/checkpoint_engine/`](verl/verl/checkpoint_engine/)).

`CheckpointEngineManager.update_weights()` ([`checkpoint_engine/base.py:469-514`](verl/verl/checkpoint_engine/base.py#L469-L514)): if `backend == "naive"`, `ray.get(actor_wg.update_weights(...))` — synchronous, same-GPU IPC. Otherwise: abort in-flight rollout requests → release kv-cache → build a process group → `ray.get(actor + rollout update_weights)` (NCCL/etc. broadcast) → resume kv-cache → resume generation.

**Default `trainer_mode=sync` is a hard barrier.** `PPOTrainerSync` ([`trainer_sync.py`](verl/verl/trainer/ppo/v1/trainer_sync.py), docstring: *"1. Trainer and rollout are colocated. 2. Partial rollout is disabled."*): `on_sample_end` puts rollout replicas to sleep immediately after sampling a batch; `on_step_end` wakes them with the newly-updated weights only after the optimizer step and checkpoint save finish. **Generation for step N+1 never starts until training on step N has finished and pushed new weights.** No stale-weight generation, no overlap — by design.

**Overlapped ("one-step-off"/streaming) modes exist but are not the default:**
- `PPOTrainerColocateAsync` ([`trainer_colocate_async.py`](verl/verl/trainer/ppo/v1/trainer_colocate_async.py), `trainer.v1.trainer_mode=colocate_async`, docstring: *"Partial rollout is enabled."*): pre-seeds `num_warmup_batches` (default 1) generation requests before training starts; after weight sync, resumes in-flight/aborted requests rather than starting cold.
- `PPOTrainerSeparateAsync` ([`trainer_separate_async.py`](verl/verl/trainer/ppo/v1/trainer_separate_async.py), `trainer.v1.trainer_mode=separate_async`, docstring: *"Trainer and rollout are separate, trainer may switch to rollout if idle."*): a standalone, continuously-serving rollout replica pool with its own `CheckpointEngineManager`; requires a real network checkpoint-engine backend. `parameter_sync_step` (default **4**, [`ppo_trainer.yaml:225`](verl/verl/trainer/config/ppo_trainer.yaml#L225)) controls how often weights sync.
- The **replay buffer / TransferQueue** machinery is what actually implements streaming and staleness generically across modes: [`verl/verl/trainer/ppo/v1/replay_buffer.py`](verl/verl/trainer/ppo/v1/replay_buffer.py)`::ReplayBuffer` tracks `max_off_policy_threshold` (default **8**, [`ppo_trainer.yaml:231`](verl/verl/trainer/config/ppo_trainer.yaml#L231)) and `max_off_policy_strategy` (default **`"drop"`**, [`ppo_trainer.yaml:236`](verl/verl/trainer/config/ppo_trainer.yaml#L236), alternative `"wait"`). `trajectory_staleness = (global_steps − prompt_global_steps + 1) / parameter_sync_step`; over-stale trajectories are dropped or waited on accordingly. Generation dispatch ([`AgentLoopWorkerTQ.generate_sequences`](verl/verl/trainer/ppo/v1/agent_loop_tq.py), [lines 54-100](verl/verl/trainer/ppo/v1/agent_loop_tq.py#L54-L100)) is genuinely fire-and-forget in this path — comment at [line 95](verl/verl/trainer/ppo/v1/agent_loop_tq.py#L95): *"fire-and-forget background tasks"* — with results written asynchronously into TransferQueue; `ReplayBuffer.sample()` polls until enough finished trajectories exist.
- **In the default `sync` mode, this generic machinery degenerates to strict on-policy behavior**: exactly one generation batch is ever in flight (a new batch is queued only after the previous one is fully sampled), so staleness never has room to accumulate even though the underlying dispatch/replay-buffer code is shared across all trainer modes.
- Older, separate experimental prototypes of the same idea exist under [`verl/verl/experimental/`](verl/verl/experimental/): `one_step_off_policy` ([`verl/docs/advance/one_step_off.md`](verl/docs/advance/one_step_off.md) — a one-batch-ahead pipeline with NCCL `sync_rollout_weights`, ~300ms per the doc) and `fully_async_policy` ([`verl/docs/advance/fully_async.md`](verl/docs/advance/fully_async.md) — fully decoupled Rollouter/Trainer/MessageQueue/ParameterSynchronizer, with an explicit `async_training.staleness_threshold` parameter: *"`staleness_threshold=0` indicates synchronous training... `staleness_threshold>0` indicates asynchronous training"*).

### 1.5 Verdict on "rollout is async, trainer is sync"

**Partially accurate — true at the request level, but the outer loop is also synchronous by default.**

- **True, and stronger than usually assumed:** the rollout *engine* isn't just async by default, async is the *only* supported mode — sync SPMD generation was removed from the codebase entirely.
- **True:** the trainer's forward/backward/optimizer-step is a standard synchronous, blocking, data-parallel update.
- **Needs a caveat:** in the default configuration, the *outer* generate→train→sync-weights→repeat loop is *also* fully synchronous and non-overlapping — rollout sleeps during training and only wakes with fresh weights afterward. Genuine overlap (rollout running ahead on stale weights while training proceeds) requires explicitly selecting `trainer.v1.trainer_mode=colocate_async` or `separate_async` (or the older experimental recipes) — none of which is the default.

**Precise phrasing:** *"By default, VERL's rollout engine is asynchronous at the request level, but the outer PPO training loop is synchronous and non-overlapping: the trainer blocks on the full generation call before training, and blocks on training/weight-sync before the next generation call. Streaming/overlapped execution is available but must be explicitly selected."*

### 1.6 Documented failure modes tied to this architecture

- **Training/inference numerical mismatch.** [`rollout_corr_helper.py`](verl/verl/trainer/ppo/rollout_corr_helper.py) module docstring ([lines 14-23](verl/verl/trainer/ppo/rollout_corr_helper.py#L14-L23)) names this explicitly: *"Policy mismatch between rollout and training implementations (e.g., vLLM BFloat16 vs FSDP FP32)"*, *"Model update staleness (training on trajectories from older checkpoints)"*, *"General distribution shifts between data collection and training."* Cites the paper *"When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"* ([line 59](verl/verl/trainer/ppo/rollout_corr_helper.py#L59)). This is the entire motivation for `rollout_is`/`rollout_rs` (§3.9).
- **Tokenization/re-tokenization drift in multi-turn agent loops.** [`verl/verl/workers/rollout/schemas.py:606-665`](verl/verl/workers/rollout/schemas.py#L606-L665) re-tokenizes the full rendered conversation from scratch and compares it against the incrementally-built token ids, controlled by `tokenization_sanity_check_mode` (default `"strict"`, [`rollout.py:60`](verl/verl/workers/config/rollout.py#L60)) — directly relevant to your multi-turn Workplace Assistant rollouts.
- **GPU idle time from the synchronous pipeline** (the stated motivation for every async recipe): [`verl/docs/advance/one_step_off.md:11-19`](verl/docs/advance/one_step_off.md#L11-L19) — *"Model updates must wait for the longest output in the generation phase to complete... GPUs remain idle."* Cited example: DAPO 32B's rollout phase ≈70% of total step time.
- **Partial-rollout interruption handling.** Before every non-colocated weight sync, `abort_replicas()` runs first (*"abort and save all unfinished requests for partial rollout"*, [`checkpoint_engine/base.py:482-483`](verl/verl/checkpoint_engine/base.py#L482-L483)), and kv-cache is explicitly released/restored around the broadcast so it can write into buffers without racing an in-flight generation step.
- **Documented race-condition guard:** `[legacy]` [`ray_trainer.py:1456-1460`](verl/verl/trainer/ppo/ray_trainer.py#L1456-L1460) — *"Keep them in a single agent-loop/vLLM request to avoid sending a second rollout after replicas have been put to sleep, which can leave async vLLM engines in an invalid state for multi-turn agent workloads."*
- **FSDP checkpoint-load buffer-ordering deadlock risk** (unrelated to rollout, but a documented sync hazard): [`verl/verl/utils/fsdp_utils.py:503-507`](verl/verl/utils/fsdp_utils.py#L503-L507) — mismatched buffer ordering across ranks during checkpoint load can cause a size mismatch on the same broadcast collective, i.e. an NCCL deadlock; fixed by sorting buffers deterministically before `dist.broadcast`.

---

## 2. One Batch, End to End

Walking through exactly one rollout batch of [`multi-domain-RL/launch.sh`](multi-domain-RL/launch.sh), with real numbers.

**1. Prompt sampling.** 128 prompts drawn from [`multi-domain-RL/data/multi-domain-RL-train.parquet`](multi-domain-RL/data/) (blended MCQA ~55% / Workplace Assistant tool-calling ~30% / Structured Output ~15%, per Cascade 2 §4.3, [`Cascade2.pdf`](Cascade2.pdf)).

**2. Async generation → 2048 trajectories.** `rollout.n=16` responses per prompt → 128×16 = **2048 trajectories**. Dispatched as 2048 concurrent `asyncio` coroutines against vLLM's `AsyncLLM` (§1.2). For Workplace Assistant prompts, generation is a real multi-turn agent loop (`multi_turn.enable=True`, `max_assistant_turns=6`, `max_user_turns=5`, `max_tool_response_length=2048`, tool schemas from [`multi-domain-RL/tools.yaml`](multi-domain-RL/tools.yaml)): assistant tokens and tool-call tokens are model-generated; tool *responses* are appended as context, not generated.

**3. Masking.** Assistant-generated tokens get `loss_mask=True` ([`schemas.py::add_assistant_message`, lines 428-451](verl/verl/workers/rollout/schemas.py#L428-L451)); tool-observation tokens get `loss_mask=False` ([`add_tool_response_messages`, lines 453-514](verl/verl/workers/rollout/schemas.py#L453-L514)). `response_mask = loss_mask × attention_mask` ([`agent_loop.py:727-778`](verl/verl/experimental/agent_loop/agent_loop.py#L727-L778)) — this is the mask used everywhere downstream (loss, entropy/KL aggregation, IS-ratio averaging).

**4. Rollout log-prob capture (diagnostics only).** `calculate_log_probs=True` → vLLM also returns per-token log-probs of what it generated, populating `rollout_log_probs`. Because `bypass_mode=False` and `rollout_is=null` here, this data feeds **only** diagnostic metrics (`training/rollout_probs_diff_*`, `rollout_corr/k3_kl`, etc. — §4.9) and never touches the loss.

**5. Reward.** [`multi_domain_reward_fn`](multi-domain-RL/reward.py) (line 28) dispatches on `extra_info["agent_ref"]` to `mcqa_reward_fn` ([line 201](multi-domain-RL/reward.py)) / `structured_reward_fn` ([line 339](multi-domain-RL/reward.py)) / `workplace_reward_fn` ([line 593](multi-domain-RL/reward.py)) (all strictly binary, no partial credit — see [`multi-domain-RL/AGENTS.md`](multi-domain-RL/AGENTS.md)), passed through `DAPORewardManagerNemotron` ([`multi-domain-RL/other/dapo_overlong_penalty.py`](multi-domain-RL/other/dapo_overlong_penalty.py), overlong penalty currently disabled in this launch script).

**6. GRPO advantage.** The 16 rollouts sharing a prompt form a group. `A_i = (score_i − mean_g) / (std_g + ε)` since `norm_adv_by_std_in_grpo=True` (the default; [`core_algos.py:267-331`](verl/verl/trainer/ppo/core_algos.py#L267-L331)). **If every rollout in a Workplace Assistant group scores 0 (e.g. because the tool loop truncates before finishing), `mean_g=0` and every rollout in that group gets `A_i = 0` — zero gradient signal from that group, mechanically guaranteed, independent of any sampling/filtering.** (`filter_groups` does not actually run in this checkout — see §3.6 — so it plays no role in group composition either way.)

**7. Old-policy log-prob (decoupled mode).** Because `bypass_mode=False`, VERL performs a **separate actor forward pass** to compute `old_log_prob` (`[legacy]` [`ray_trainer.py:1533-1546`](verl/verl/trainer/ppo/ray_trainer.py#L1533-L1546) branch), distinct from both `rollout_log_probs` (vLLM) and the log-probs computed later during the actual training update (`π_θ`). This pass also produces `actor/entropy` and the `perf/mfu/actor_infer` metric.

**8. Policy loss.** `algorithm.rollout_correction.loss_type` only matters when `bypass_mode=True` (§3.9) — here it's irrelevant, and the loss path is the standard clipped-surrogate objective (`compute_policy_loss_vanilla`, [`core_algos.py:1278-1369`](verl/verl/trainer/ppo/core_algos.py#L1278-L1369)) with `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, dual-clip floor `clip_ratio_c=3.0` (default), aggregated via `loss_agg_mode=token-mean` (every valid token across the whole mini-batch weighted equally, regardless of which sequence it's in — see §3.2 for why this matters for a length-imbalanced multi-domain batch).

**9. Optimizer step.** `ppo_mini_batch_size=128` × `rollout.n=16` = 2048 = the full rollout batch; `ppo_epochs=1` → **exactly one `optimizer_step()` call** (§1.3). `use_dynamic_bsz=True` only repacks this single mini-batch into memory-sized micro-batches (Karmarkar-Karp balanced partitioning, §3.8) for gradient accumulation — it does not add optimizer steps.

**10. Gradient clipping.** `grad_clip=1.0` (VERL default, not overridden here; confirmed both in [`optimizer.py:53`](verl/verl/workers/config/optimizer.py#L53) and [`trainer/config/actor/dp_actor.yaml:29`](verl/verl/trainer/config/actor/dp_actor.yaml#L29)) applied inside `optimizer_step`; the `grad_norm` metric logged is always the **pre-clip** value (§3.7). If non-finite, the update is silently skipped (§5).

**11. Weight sync.** Default `trainer_mode=sync`: rollout replicas were asleep during steps 5-10, woken with the newly-updated weights only after the optimizer step and checkpoint save (`trainer.save_freq=10`) — the next rollout batch (step 1 again) cannot start generating until this completes.

**12. Metrics logged this step:** reward/score, advantage, pg_loss/clipfrac/ppo_kl, entropy, grad_norm/lr, response-length/truncation, timing/MFU, `global_seqlen` balance stats, and (since `calculate_log_probs=True`) the rollout-mismatch diagnostics — full catalog in §4.

---

## 3. Parameter Reference

### 3.1 Clipping — `clip_ratio_low`, `clip_ratio_high`, `clip_ratio_c`

Defaults: `clip_ratio=0.2`, `clip_ratio_low=0.2`, `clip_ratio_high=0.2`, `clip_ratio_c=3.0` ([`verl/verl/workers/config/actor.py:158-163`](verl/verl/workers/config/actor.py#L158-L163)).

`compute_policy_loss_vanilla` ([`core_algos.py:1278-1369`](verl/verl/trainer/ppo/core_algos.py#L1278-L1369)):
```python
ratio = exp(clamp(log_prob - old_log_prob, -20, 20))
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * clamp(ratio, 1-clip_ratio_low, 1+clip_ratio_high)
clip_pg_losses1 = max(pg_losses1, pg_losses2)               # standard PPO clip
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = min(pg_losses3, clip_pg_losses1)          # dual-clip floor
pg_losses = where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```
Dual-clip (`clip_ratio_c`, must be `>1.0`, [lines 1324-1327](verl/verl/trainer/ppo/core_algos.py#L1324-L1327)) only engages on **negative-advantage** tokens — it stops the loss diverging to `-∞·A` when the ratio collapses toward 0 on a token that should be suppressed (dual-clip PPO, arXiv:1912.09729).

**Metrics — two counterintuitive facts** ([lines 1364-1368](verl/verl/trainer/ppo/core_algos.py#L1364-L1368)):
- `actor/pg_clipfrac = mean(pg_losses2 > pg_losses1)` fires on **either** clip side combined — you **cannot** attribute it to `clip_ratio_high` vs. `clip_ratio_low` specifically from this metric alone. **`actor/pg_clipfrac_higher` does not exist anywhere in this codebase** — don't look for it.
- `actor/pg_clipfrac_lower = mean((clip_pg_losses1 > pg_losses3) & (advantages<0))` measures **dual-clip (`clip_ratio_c`) activations on negative-advantage tokens** — despite the name, it is *not* the low-side ratio-clip fraction. Lowering `clip_ratio_c` raises this metric.
- `actor/ppo_kl = mean(-(log_prob - old_log_prob))` — a k1-style approximate KL between new and old policy, monitoring-only, not part of the loss.

The base `clip_ratio` is reused as a *divergence threshold* (not a ratio-clip) in the alternate `dppo_tv`/`dppo_kl` losses ([lines 1372-1535](verl/verl/trainer/ppo/core_algos.py#L1372-L1535)), and as the ratio-clip for the sequence-level `gspo` loss ([lines 1538-1611](verl/verl/trainer/ppo/core_algos.py#L1538-L1611)) — which formula actually runs is selected by `actor.policy_loss.loss_mode` (registry in `get_policy_loss_fn`, [`core_algos.py:70-85`](verl/verl/trainer/ppo/core_algos.py#L70-L85)), independent of the clip values themselves.

### 3.2 `loss_agg_mode`

Valid values ([`verl/verl/workers/config/actor.py:213-220`](verl/verl/workers/config/actor.py#L213-L220)): `token-mean`, `seq-mean-token-sum`, `seq-mean-token-mean`, `seq-mean-token-sum-norm`. Implemented in `agg_loss` ([`core_algos.py:1138-1199`](verl/verl/trainer/ppo/core_algos.py#L1138-L1199)):

| Mode | Formula | Effect on a length-imbalanced, multi-domain batch |
|---|---|---|
| `token-mean` (**current default here**) | `sum(loss·mask) / global_valid_token_count` — [lines 1168-1173](verl/verl/trainer/ppo/core_algos.py#L1168-L1173) | **Every token counts equally regardless of sequence length.** A 4000-token Workplace Assistant response and three 200-token MCQA responses in the same mini-batch: the long one contributes ~87% of the gradient. |
| `seq-mean-token-sum` | per-sequence loss = token-sum, then mean over sequences — [lines 1174-1181](verl/verl/trainer/ppo/core_algos.py#L1174-L1181) | Every *sequence* gets equal weight regardless of length — but a token deep in a long sequence has less individual leverage than one in a short sequence. |
| `seq-mean-token-sum-norm` | as above, divided by a fixed `loss_scale_factor` (default = `response_length`, or a user constant via `actor.loss_scale_factor`) — [lines 1182-1186](verl/verl/trainer/ppo/core_algos.py#L1182-L1186) | Decouples the token-sum from the padded response horizon, so changing `max_response_length` doesn't silently rescale the loss. |
| `seq-mean-token-mean` | per-sequence loss = token-mean, then mean over sequences — [lines 1187-1195](verl/verl/trainer/ppo/core_algos.py#L1187-L1195) | The only mode giving *both* every sequence equal weight *and* every token within a sequence equal weight — long and short domains contribute equally per-example. `gspo` and `sapo` force this mode regardless of config. |

**Implication for this Cascade repro:** with `token-mean` (current setting) and Workplace Assistant responses running much longer than MCQA/structured ones, the long-sequence domain structurally dominates the per-step gradient magnitude *whenever it does have nonzero advantage* — worth knowing when comparing gradient contributions across domains.

### 3.3 Entropy — `entropy_coeff`

Default `0` ([`actor.py:166`](verl/verl/workers/config/actor.py#L166)). Formula ([`torch_functional.py:224-238`](verl/verl/utils/torch_functional.py#L224-L238)): `entropy = logsumexp(logits,-1) − Σ softmax(logits)·logits` (per-token Shannon entropy, nats). Applied in `ppo_loss` ([`losses.py:122-129`](verl/verl/workers/utils/losses.py#L122-L129)): `policy_loss -= entropy_coeff * entropy_loss`. Entropy is only computed at all if `calculate_entropy=True` **or** `entropy_coeff != 0` (auto-derived, `[legacy]` [`ray_trainer.py:1299-1301`](verl/verl/trainer/ppo/ray_trainer.py#L1299-L1301)) — setting a nonzero coefficient is sufficient on its own.

The logged `actor/entropy_loss` metric is the **raw, unscaled** `agg_loss`-aggregated entropy (higher = more uniform/exploratory policy) — the coefficient only scales its contribution to `policy_loss`, it is not baked into the logged value. Two separate entropy metrics exist because they come from two different forward passes: `actor/entropy` (from the old-log-prob recompute pass, §2 step 7) and `actor/entropy_loss` (from the actual training-update forward pass) — they can legitimately differ.

### 3.4 KL — two independent mechanisms, don't conflate them

**(a) KL as an in-reward penalty** (`algorithm.use_kl_in_reward`, default `False`; estimator via `algorithm.kl_penalty`, default `"kl"`; controller `algorithm.kl_ctrl`, `type: fixed|adaptive`, `kl_coef=0.001`). `[legacy]` `apply_kl_penalty` ([`ray_trainer.py:78-117`](verl/verl/trainer/ppo/ray_trainer.py#L78-L117)):
```python
kld = kl_penalty(old_log_prob, ref_log_prob, kl_penalty=kl_penalty_type)
token_level_rewards = token_level_scores - kl_ctrl.value * kld
```
Subtracted directly from the **reward** before advantage computation — not from the loss. Metrics: `actor/reward_kl_penalty` (mean KL), `actor/reward_kl_penalty_coeff` (current β; adaptive controller formula, `AdaptiveKLController.update`, arXiv:1909.08593: `mult = 1 + clip(kl/target − 1, −0.2, 0.2) · n_steps/horizon`).

**(b) KL as an explicit loss term** (`actor.use_kl_loss`, default `False`; `actor.kl_loss_coef=0.001`; `actor.kl_loss_type="low_var_kl"`). `ppo_loss` ([`losses.py:132-142`](verl/verl/workers/utils/losses.py#L132-L142)): `policy_loss += kl_loss * kl_loss_coef`. Metrics: `actor/kl_loss`, `actor/kl_coef`.

**KL estimator formulas** (`kl_penalty_forward`, [`core_algos.py:2154-2189`](verl/verl/trainer/ppo/core_algos.py#L2154-L2189)):

| Name | Formula | Notes |
|---|---|---|
| `kl` / `k1` | `logprob − ref_logprob` | signed, can be negative |
| `abs` | `\|logprob − ref_logprob\|` | |
| `mse` / `k2` | `0.5·(logprob − ref_logprob)²` | correct gradient, biased-high KL estimate |
| `low_var_kl` / `k3` (default `kl_loss_type`) | `r=exp(clamp(ref−log,−20,20)); clamp(r − (ref−log) − 1, −10,10)` | Schulman's low-variance unbiased-in-expectation estimator (joschu.net/blog/kl-approx.html) |
| `full` | `NotImplementedError` | not usable |

A `"+"` suffix (e.g. `"k3+"`) keeps k1/k3's forward *value* but substitutes k2's *gradient* via a straight-through trick — comment: k1/k3 have the right expected value but a biased expected gradient; k2 has the right gradient.

Cascade 2's config sets `use_kl_loss=False` and `use_kl_in_reward=False` — both mechanisms are off, consistent with the paper's stated removal of the KL term entirely ([`Cascade2.pdf`](Cascade2.pdf) §4.1.2).

### 3.5 Advantage estimation — `adv_estimator`, `norm_adv_by_std_in_grpo`

Registry ([`core_algos.py:88-150`](verl/verl/trainer/ppo/core_algos.py#L88-L150)) includes: `gae`, `grpo`, `grpo_vectorized`, `gdpo`, `grpo_passk`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `rloo`, `rloo_vectorized`, `opo`, `remax`, `gpg`, `optimal_token_baseline`, `tir_optimal_token_baseline`.

**GRPO** (`compute_grpo_outcome_advantage`, [`core_algos.py:267-331`](verl/verl/trainer/ppo/core_algos.py#L267-L331)):
```python
scores = token_level_rewards.sum(-1)          # scalar outcome reward per rollout
mean_g, std_g = group_stats(scores)            # grouped by prompt uid; singleton group → mean=0, std=1 (lines 315-317)
A_i = (score_i - mean_g) / (std_g + eps)  if norm_adv_by_std_in_grpo else  score_i - mean_g
```
**Singleton-group trap:** if a group ever has only 1 sample, `mean=0, std=1` unconditionally ([lines 315-317](verl/verl/trainer/ppo/core_algos.py#L315-L317)) — the "advantage" becomes the *raw reward itself*, not zero. Not currently a live risk here since `rollout.n=16` always produces full groups and `filter_groups` doesn't actually shrink them (§3.6), but worth remembering if group composition ever changes.

`norm_adv_by_std_in_grpo=False` recovers **Dr. GRPO** (arXiv:2503.20783) — removes the `/std` term that inflates advantage magnitude for low-variance (near-all-correct or near-all-wrong) groups. Cascade 2's own config keeps `norm_adv_by_std_in_grpo=True` (original GRPO), per its launch scripts.

**GAE** (`compute_gae_advantage_return`, [`core_algos.py:215-263`](verl/verl/trainer/ppo/core_algos.py#L215-L263)) — standard recursive TD(λ), requires a critic; not used in the Cascade repro (no critic configured). `gamma`/`lam` ([`algorithm.py:651-652`](verl/verl/trainer/config/algorithm.py#L651-L652)) only matter here.

Other estimators worth knowing exist, not currently used: `rloo`/`rloo_vectorized` (leave-one-out baseline), `opo` (length-weighted baseline), `grpo_passk` (only the top-scorer per group gets nonzero advantage), `gdpo` (per-reward-dimension decoupled normalization, needs `algorithm.gdpo_reward_keys` — potentially relevant if you ever want per-domain-weighted advantages instead of a single blended reward), `optimal_token_baseline`/`tir_optimal_token_baseline` (variance-weighted per-timestep baseline).

### 3.6 `filter_groups` — defined but not implemented in this checkout

**Confirmed by direct source grep** (two independent passes agree): `FilterGroupsConfig` is defined ([`verl/verl/trainer/config/algorithm.py:43-56`](verl/verl/trainer/config/algorithm.py#L43-L56): `enable`, `metric` ∈ {`acc`,`score`,`seq_reward`,`seq_final_reward`}, `max_num_gen_batches`) and attached as `AlgorithmConfig.filter_groups` ([line 660](verl/verl/trainer/config/algorithm.py#L660)), but **no code anywhere in this checkout reads it**. The DAPO-style dynamic-sampling loop (drop degenerate all-same-reward groups, resample up to `max_num_gen_batches` extra rounds) lives in an external `recipe` git submodule (`verl-recipe.git`) that is **not checked out** in this repo (`ls verl/recipe/` is empty). No metric reports dropped groups or extra generation rounds either.

**`+algorithm.filter_groups.enable=True` in your `launch.sh` files is currently a silent no-op.** It does not drop or resample any groups. This matters directly for the Workplace Assistant reward investigation ([`multi-domain-RL/AGENTS.md`](multi-domain-RL/AGENTS.md)) — the flat reward there is fully explained by GRPO's own math on all-zero groups (§2 step 6), with no filter_groups mechanism involved at all.

### 3.7 Gradient / optimizer

`OptimizerConfig.clip_grad: float = 1.0` ([`verl/verl/workers/config/optimizer.py:53`](verl/verl/workers/config/optimizer.py#L53)) is the actually-consumed default (also mirrored in [`verl/verl/trainer/config/actor/dp_actor.yaml:29`](verl/verl/trainer/config/actor/dp_actor.yaml#L29)). `grad_clip` is a **deprecated alias** that overwrites `clip_grad` with a warning if set ([lines 54-61](verl/verl/workers/config/optimizer.py#L54-L61)); a second, unrelated legacy `grad_clip` field also exists directly on `FSDPActorConfig` ([`actor.py:306`](verl/verl/workers/config/actor.py#L306)) but isn't the one `optimizer_step` reads.

Clipping ([`transformer_impl.py:686-732`](verl/verl/workers/engine/fsdp/transformer_impl.py#L686-L732)):
```python
grad_norm = self.module.clip_grad_norm_(clip_grad)   # FSDP1
# or fsdp2_clip_grad_norm_(...) / torch.nn.utils.clip_grad_norm_(...)
```
All three return the **pre-clip** total norm — PyTorch's `clip_grad_norm_` contract computes the norm, scales in place, and returns the *original* value. **No post-clip norm is ever logged anywhere in this codebase.** If the returned norm is non-finite, the update is skipped and grads are zeroed ([lines 720-725](verl/verl/workers/engine/fsdp/transformer_impl.py#L720-L725)) — silently, visible only as `nan`/`inf` in the `actor/grad_norm` metric, no separate flag.

A standalone `compute_grad_norm()` ([`torch_functional.py:365-385`](verl/verl/utils/torch_functional.py#L365-L385)) exists but **has zero callers anywhere in `verl/`** — it computes a sum-of-squares (not even the norm, per its own docstring caveat), and does not feed the actual `grad_norm` metric. Disregard it.

Metric plumbing: `optimizer_step()` return → `outputs["metrics"]["grad_norm"]` ([`engine/base.py:126-132`](verl/verl/workers/engine/base.py#L126-L132)) → renamed `actor/grad_norm` (`rename_dict`, `[legacy]` [`ray_trainer.py:1334`](verl/verl/trainer/ppo/ray_trainer.py#L1334)).

`lr`/`betas`/`weight_decay`: `OptimizerConfig` ([`optimizer.py:47-52`](verl/verl/workers/config/optimizer.py#L47-L52)) — `lr` has no default (must be set), `weight_decay=0.01` default, `betas=(0.9,0.999)` default (Cascade sets `(0.9,0.95)` explicitly). LR scheduler: `lr_scheduler_type` (`"constant"`/`"cosine"`, default constant) + `lr_warmup_steps_ratio`/`lr_warmup_steps` ([`transformer_impl.py:479-510`](verl/verl/workers/engine/fsdp/transformer_impl.py#L479-L510)).

### 3.8 Batching / memory — `use_dynamic_bsz`, `ppo_max_token_len_per_gpu`, `ppo_micro_batch_size_per_gpu`

`use_dynamic_bsz: bool=False` default (Cascade sets `True`); mutually exclusive with `ppo_micro_batch_size_per_gpu` ([`actor.py:200-211`](verl/verl/workers/config/actor.py#L200-L211)). Effective token budget scales with Ulysses sequence-parallel size (`max_token_len = data["max_token_len_per_gpu"] * sp_size`, [`engine/utils.py:73-76`](verl/verl/workers/engine/utils.py#L73-L76)), then `rearrange_micro_batches` ([`seqlen_balancing.py:348-468`](verl/verl/utils/seqlen_balancing.py#L348-L468)) packs sequences into micro-batches via a **Karmarkar-Karp-style balanced partition** (workload ≈ token count, sorted descending then interleaved from both ends to reduce pipeline-bubble at warm-up/cool-down) — this lets long and short sequences share GPU memory efficiently instead of padding every micro-batch to the longest sequence in a fixed-size batch.

Imbalance is logged as `global_seqlen/{min,max,minmax_diff,balanced_min,balanced_max,mean}` ([`seqlen_balancing.py:257-302`](verl/verl/utils/seqlen_balancing.py#L257-L302), called from `_balance_batch`, `[legacy]` [`ray_trainer.py:1145-1213`](verl/verl/trainer/ppo/ray_trainer.py#L1145-L1213)) — this reflects **DP-rank workload imbalance**, not the dynamic-bsz micro-batch imbalance specifically (that isn't separately logged).

### 3.9 Rollout correction — `rollout_is`, `rollout_rs`, `bypass_mode`, `loss_type`

Config: [`verl/verl/trainer/config/algorithm/rollout_correction.yaml`](verl/verl/trainer/config/algorithm/rollout_correction.yaml), defaults `rollout_is: null, rollout_is_threshold: 2.0, rollout_rs: null, rollout_rs_threshold: null, bypass_mode: false, loss_type: ppo_clip`.

| Setting | What it does | File:line |
|---|---|---|
| `rollout_is: null` / `"token"` / `"sequence"` | off / per-token ratio π_θ/π_rollout / per-sequence product ratio; multiplies the pg-loss term; always **detached** before use ([lines 621-623](verl/verl/trainer/ppo/rollout_corr_helper.py#L621-L623), plus `torch.no_grad()` wrap in bypass mode, [`core_algos.py:2434`](verl/verl/trainer/ppo/core_algos.py#L2434)) | [`rollout_corr_helper.py:520-655`](verl/verl/trainer/ppo/rollout_corr_helper.py#L520-L655) |
| `bypass_mode` | `True`: `old_log_probs := rollout_log_probs` — identifies π_old with π_rollout, skips the extra actor forward pass (§2 step 7). `False` (current): 3 distinct policies (π_rollout, π_old, π_θ) | `[legacy]` [`ray_trainer.py:1533-1546`](verl/verl/trainer/ppo/ray_trainer.py#L1533-L1546), [`rollout_corr_helper.py:1107-1143`](verl/verl/trainer/ppo/rollout_corr_helper.py#L1107-L1143) |
| `loss_type: ppo_clip` vs `reinforce` | **Only meaningful under `bypass_mode=True`.** `ppo_clip` explicitly does **not** also apply `rollout_is_weights` (comment: would double-count the correction the clip already performs). `reinforce` does apply it, detached. | [`core_algos.py:2351-2486`](verl/verl/trainer/ppo/core_algos.py#L2351-L2486) |
| `calculate_log_probs` | Pure compute-and-expose switch (vLLM returns its own per-token log-probs). Zero effect on gradients **iff** `bypass_mode=False` and `rollout_is=null` (current defaults) — otherwise feeds diagnostics only. | [`verl/verl/workers/config/rollout.py:218`](verl/verl/workers/config/rollout.py#L218) |

`rollout_is_threshold` accepts either a single float (one-sided TIS upper clamp — unbiased, no lower truncation) or a `"lower_upper"` string like `"0.5_2.0"` (two-sided **masking**: zero out any weight outside `[lower, upper]`, the "IcePop" style). **This confirms Cascade 2 MOPD's ε_low=0.5/ε_high=2.0 must be the two-sided masking mode** — plain one-sided TIS has no lower bound by design, to preserve unbiasedness.

A named preset, `RolloutCorrectionConfig.bypass_pg_token_icepop()` ([`algorithm.py:369-391`](verl/verl/trainer/config/algorithm.py#L369-L391)), implements almost exactly "bypass_mode + reinforce + token IS + IcePop threshold" — confirming that configuration is reachable and correctly detached, though (per the debugging discussion in [`multi-domain-RL/AGENTS.md`](multi-domain-RL/AGENTS.md)) it changes what "old policy" means and should be treated as a deliberate deviation from Cascade 2's decoupled on-policy setup, not adopted casually.

### 3.10 Overlong / length shaping — core VERL vs. custom reward manager

A **built-in** `"dapo"` reward manager exists in core VERL ([`verl/verl/workers/reward_manager/dapo.py:25-155`](verl/verl/workers/reward_manager/dapo.py#L25-L155), `@register("dapo")`), separate from your custom [`dapo_overlong_penalty.py`](multi-domain-RL/other/dapo_overlong_penalty.py). Its soft penalty:
```python
expected_len = max_resp_len - overlong_buffer_cfg.len
exceed_len = valid_response_length - expected_len
overlong_reward = min(-exceed_len / overlong_buffer_cfg.len * overlong_buffer_cfg.penalty_factor, 0)
```
— zero penalty until the response enters the last `overlong_buffer_len` tokens before the cap, then a linearly-ramping negative penalty (capped at `-penalty_factor`) added at the final valid-response position. This is a **third** near-duplicate implementation alongside your project's [`multi-domain-RL/other/dapo_overlong_penalty.py`](multi-domain-RL/other/dapo_overlong_penalty.py) and [`verl/verl/experimental/reward_loop/reward_manager/dapo_overlong_penalty.py`](verl/verl/experimental/reward_loop/reward_manager/dapo_overlong_penalty.py) / [`dapo.py`](verl/verl/experimental/reward_loop/reward_manager/dapo.py) — confirm at config-selection time (`reward.reward_manager.name`) which one is actually active; your launch scripts use the custom `dapo_overlong_penalty` registered manager with `overlong_penalty.enable=False`.

### 3.11 Rollout sampling — `temperature`, `top_p`, `top_k`, `do_sample`

`SamplingConfig` (`temperature=1.0, top_k=-1, top_p=1.0, do_sample=True, n=1`, [`rollout.py:39-44`](verl/verl/workers/config/rollout.py#L39-L44)), duplicated at `RolloutConfig` top level for training generation, with a separate `val_kwargs` override for validation.

**`top_p`, `top_k`, `do_sample`, `n` are confirmed generation-only** — they flow into `SamplingParams` and never appear in `core_algos.py`/`losses.py`.

**`temperature` is the one exception — it re-enters the training-time forward pass:**
```
verl/verl/workers/engine/fsdp/transformer_impl.py:1142   logits_rmpad = logits_rmpad / temperature_rmpad.clamp(min=1e-8)
verl/verl/workers/engine/fsdp/transformer_impl.py:1255   logits = logits / temperature.clamp(min=1e-8)
```
([lines 1142](verl/verl/workers/engine/fsdp/transformer_impl.py#L1142), [1255](verl/verl/workers/engine/fsdp/transformer_impl.py#L1255)) This keeps the actor's recomputed `log_prob`/entropy numerically consistent with the distribution actually sampled from during rollout — meaning `temperature` genuinely affects the PPO ratio and the entropy bonus, unlike the other sampling knobs.

---

## 4. Metric Rubric

Every "normal range" claim is labeled by source. Where VERL's own code gives no numeric guidance, that's stated explicitly — treat unlabeled thresholds as fabricated if you ever see them elsewhere.

### 4.1 Reward / score

All computed in `compute_data_metrics()`, [`verl/verl/trainer/ppo/metric_utils.py:411-590`](verl/verl/trainer/ppo/metric_utils.py#L411-L590).

| Metric | Formula | Normal range |
|---|---|---|
| `critic/score/{mean,max,min}` | stats of `token_level_scores.sum(-1)` (raw reward, pre-KL) — [lines 435, 456-462, 538-540](verl/verl/trainer/ppo/metric_utils.py#L435) | `[no documented threshold — task-specific; for binary rewards, mean = fraction correct]` |
| `critic/rewards/{mean,max,min}` | stats of `token_level_rewards.sum(-1)` (post-KL-penalty if `use_kl_in_reward`) — [lines 436, 464-470, 542-544](verl/verl/trainer/ppo/metric_utils.py#L436) | identical to `score` when KL-in-reward is off (Cascade's case) |
| `response/aborted_ratio` | `mean(response_length == 0)` — [lines 493, 569](verl/verl/trainer/ppo/metric_utils.py#L493) | `[general RL knowledge: should be ~0; nonzero indicates generation failures/timeouts]` |

### 4.2 Advantage / return

[`metric_utils.py:472-489, 546-552`](verl/verl/trainer/ppo/metric_utils.py#L472-L552).

| Metric | Formula | Normal range |
|---|---|---|
| `critic/advantages/{mean,max,min}` | stats over `response_mask`-selected advantages | `[no documented threshold]` — mean ≈ 0 is expected for group-normalized advantages by construction |
| `critic/returns/{mean,max,min}` | stats over returns (GAE path only) | `[no documented threshold]` |
| `critic/vf_explained_var` | `1 − Var(returns−values)/Var(returns)` — [`metric_utils.py:514-522`](verl/verl/trainer/ppo/metric_utils.py#L514-L522) | `[general RL knowledge]` 1.0 = value fn perfectly predicts returns; 0 = no better than predicting the mean; **negative = value function worse than the mean baseline (red flag)**. Not applicable to Cascade's GRPO-only setup (no critic). |

### 4.3 Policy loss / PPO

Registered loss functions in [`core_algos.py`](verl/verl/trainer/ppo/core_algos.py), keyed under `actor/*`.

| Metric | Loss type | Formula | File:line |
|---|---|---|---|
| `actor/pg_loss` | all | `agg_loss(pg_losses, response_mask, loss_agg_mode)` | [`losses.py:119`](verl/verl/workers/utils/losses.py#L119) |
| `actor/pg_clipfrac` | vanilla/dppo_tv/dppo_kl/gspo/geo_mean/cispo | fraction where clipped loss term won (either side) | [`core_algos.py:1346,1442,1527,1602,1996,2046`](verl/verl/trainer/ppo/core_algos.py#L1346) |
| `actor/pg_clipfrac_lower` | vanilla/dppo_tv/dppo_kl/geo_mean | dual-clip lower-bound activation fraction (Ye et al. 2019, arXiv:1912.09729) | [`core_algos.py:1350,1443,1528,1997`](verl/verl/trainer/ppo/core_algos.py#L1350) |
| `actor/pg_clipfrac` (clip_cov) | clip_cov | `mean(corr == 0)` — fraction of tokens whose covariance term was zeroed | [`core_algos.py:1822,1834`](verl/verl/trainer/ppo/core_algos.py#L1822) |
| `actor/ppo_kl` (kl_cov) | kl_cov | uses `\|negative_approx_kl\|` (absolute KL) instead of signed | [`core_algos.py:1882-1915`](verl/verl/trainer/ppo/core_algos.py#L1882-L1915) |
| `actor/ppo_kl` (reinforce) | reinforce | KL between current policy and **rollout** log-prob (not old_log_prob) | [`core_algos.py:2341-2345`](verl/verl/trainer/ppo/core_algos.py#L2341-L2345) |

Value-function loss (critic side, not used in Cascade's no-critic GRPO setup): `critic/vf_loss`, `critic/vf_clipfrac`, `critic/vpred_mean` — [`losses.py:167-184`](verl/verl/workers/utils/losses.py#L167-L184), [`core_algos.py:2084-2123`](verl/verl/trainer/ppo/core_algos.py#L2084-L2123).

Normal ranges: `[no documented threshold — trend matters, not absolute value]`; for `actor/ppo_kl` specifically, should track near-zero here since `ppo_epochs=1` and mini-batch=full batch means only one update ever happens per rollout batch (ratio starts at ~1).

### 4.4 Entropy

| Metric | Formula | Normal range |
|---|---|---|
| `actor/entropy` | token entropy, old-log-prob recompute pass | `[source: PRIME-RL entropy-collapse paper, arXiv:2505.22617, general finding, not a VERL-stated threshold]` — a sharp, sustained downward trend early in training, without recovery, is the entropy-collapse signature; absolute "normal" value is model/task-dependent |
| `actor/entropy_loss` | token entropy, training-update pass (unscaled) — [`losses.py:129`](verl/verl/workers/utils/losses.py#L129) | same as above; compare trend against `actor/entropy` — large divergence suggests the two forward passes see meaningfully different mini-batch splits/precision |

### 4.5 KL (actor vs. reference)

| Metric | Formula | Normal range |
|---|---|---|
| `actor/reward_kl_penalty` | mean KL folded into reward (only if `use_kl_in_reward`) — [`ray_trainer.py:115`](verl/verl/trainer/ppo/ray_trainer.py#L115) | `[no documented threshold]`; only relevant if enabled — Cascade has it off |
| `actor/reward_kl_penalty_coeff` | current adaptive/fixed β | n/a, diagnostic only |
| `actor/kl_loss`, `actor/kl_coef` | in-loss KL penalty and coefficient (only if `use_kl_loss`) — [`losses.py:141-142`](verl/verl/workers/utils/losses.py#L141-L142) | `[no documented threshold]`; Cascade has this off too, per the paper's stated removal of the KL term |

### 4.6 Gradient / optimizer

| Metric | Formula | Normal range |
|---|---|---|
| `actor/grad_norm` | **pre-clip** total gradient norm | `[general RL knowledge]` — values persistently several× the `clip_grad` threshold (1.0 here) aren't necessarily broken (clipping still bounds the actual update) but indicate large raw gradients worth explaining; `nan`/`inf` means the update was silently skipped this step (§5) |
| `actor/lr` | current learning rate | should match your configured schedule exactly; divergence indicates a scheduler misconfiguration |

### 4.7 Response-length / truncation

[`metric_utils.py:441-508, 554-574`](verl/verl/trainer/ppo/metric_utils.py#L441-L574).

| Metric | Formula | Normal range |
|---|---|---|
| `response_length/{mean,max,min}` | full-batch response token-length stats | task-dependent |
| `response_length/clip_ratio` | `mean(response_length == max_response_length)` | `[no documented threshold]` — mechanically, any nonzero value means some responses hit the length ceiling (truncated); a high value alongside a flat reward for a multi-step task is a strong truncation signal (see §5) |
| `response_length_non_aborted/*` | same stats, excluding zero-length ("aborted") samples | comment: *"excludes aborted samples to avoid skew from zeros"* |
| `num_turns/{min,max,mean}` | multi-turn agent-loop turn counts | task-dependent; compare against `multi_turn.max_assistant_turns` — mean approaching the cap suggests turns are being cut off, not completed naturally |
| `tool_call_counts/{min,max,mean}` | tool-calling frequency | task-dependent |

### 4.8 Throughput / performance

[`metric_utils.py:593-670`](verl/verl/trainer/ppo/metric_utils.py#L593-L670).

| Metric | Formula | Normal range |
|---|---|---|
| `timing_s/{stage}` | raw wall-clock seconds per pipeline stage (gen, update_actor, update_weights, etc.) | use to find the dominant stage (commonly "gen" — generation phase, per the async-motivation docs in §1.6) |
| `perf/mfu/actor`, `perf/mfu/critic`, `perf/mfu/actor_infer` | `estimated_flops / promised_flops / world_size` (forward-only passes divided by an extra 3.0, encoding the standard fwd:fwd+bwd ≈ 1:3 heuristic `[general deep-learning-systems knowledge, not a code comment]`) | higher = better hardware utilization; no VERL-stated "good" threshold, compare against your hardware's known achievable MFU |
| `perf/throughput` | tokens/sec/GPU | use for capacity planning, not correctness |
| `global_seqlen/{min,max,minmax_diff,balanced_min,balanced_max,mean}` | DP-rank token-workload stats before/after rebalancing | large `minmax_diff` before balancing, small after, is expected; large after balancing suggests `use_dynamic_bsz`/packing isn't helping |

### 4.9 Rollout-correction / off-policy diagnostics

All prefixed `rollout_corr/`, computed in [`verl/verl/trainer/ppo/rollout_corr_helper.py`](verl/verl/trainer/ppo/rollout_corr_helper.py). **Explicit code guidance found:** [line 210](verl/verl/trainer/ppo/rollout_corr_helper.py#L210) states for the KL-divergence trust-region modes: **"ideal = 0.0 unless noted"** — large values flag policy staleness / training-inference mismatch. No further numeric bands are given in code.

| Metric | Meaning | Normal range |
|---|---|---|
| `training/rollout_probs_diff_{mean,max,std}` | `mean(\|exp(actor_logprob) − exp(rollout_logprob)\|)` — **raw probability difference**, not log-space, not KL ([`verl/verl/utils/debug/metrics.py:63-121`](verl/verl/utils/debug/metrics.py#L63-L121)) | `[source: this project's prior notes/supervisor conversation, NOT found in VERL source or docs]` — mean < ~0.005 "generally acceptable", 0.005–0.01 "borderline", persistently > 0.01 "likely meaningful mismatch". **Treat as an unverified heuristic, not a validated VERL-documented threshold.** |
| `rollout_corr/k3_kl` | K3 estimator: `exp(log_ratio) − log_ratio − 1` — [lines 959-964](verl/verl/trainer/ppo/rollout_corr_helper.py#L959-L964) | ideal ≈ 0 `[source: code comment, line 210]` |
| `rollout_corr/rollout_is_mean` | mean applied (post-clip) IS weight — [`compute_is_metrics`, lines 658-781](verl/verl/trainer/ppo/rollout_corr_helper.py#L658-L781) | should be near 1 if train/rollout policies match closely; **not sufficient alone** — large and small ratios can cancel in the mean |
| `rollout_corr/rollout_is_std` | std of applied IS weight | low = consistent correction magnitude across tokens; high = some tokens getting corrected much more than others |
| `rollout_corr/rollout_is_eff_sample_size` | classical ESS: `(Σw)²/Σw²` via normalized weights — [lines 751-757](verl/verl/trainer/ppo/rollout_corr_helper.py#L751-L757) | closer to the batch size = better (little effective data loss from reweighting); much smaller = most of the batch's effective signal comes from a few tokens |
| `rollout_corr/rollout_is_ratio_fraction_high`, `_fraction_low` | fraction of ratios clipped at the upper/lower threshold | high fraction = mismatch is large enough to be routinely truncated, not a rare tail event |
| `rollout_corr/chi2_token`, `chi2_seq` | chi-squared divergence diagnostics — [lines 947-1006](verl/verl/trainer/ppo/rollout_corr_helper.py#L947-L1006) | `[no documented threshold — trend matters]` |
| `rollout_corr/log_ppl_abs_diff`, `ppl_ratio` | training vs. rollout perplexity mismatch | `[no documented threshold]` — large values corroborate a `k3_kl`/`rollout_probs_diff` signal from a different angle |

### 4.10 Validation

**Correction: the actual prefixes are `val-core/` and `val-aux/`, not `val/`.** `val-core/{data_source}/{var}/{mean@N,maj@N,best@N}` for the "core" variable (`acc` if present, else `reward`); everything else under `val-aux/`. `val-aux/num_turns/{min,max,mean}` also present for multi-turn validation. `maj@N` (majority-vote bootstrap) only appears if a `"pred"` field exists in your data. `[legacy]` [`ray_trainer.py:721-744`](verl/verl/trainer/ppo/ray_trainer.py#L721-L744), formulas in [`metric_utils.py:878-1026`](verl/verl/trainer/ppo/metric_utils.py#L878-L1026).

---

## 5. Problem → Symptom → Metric → Fix

Only failure modes with direct evidence from this codebase or this project's actual runs — not a generic RL troubleshooting list.

| Problem | Symptoms | Metrics to check | Likely cause / fix |
|---|---|---|---|
| **Advantage collapse from a degenerate (all-same-reward) group** | A specific domain/task's reward never moves, even though training "runs fine" | `critic/advantages/{mean,std}` conditioned per-domain if you log it separately; per-domain reward counts (your `DAPORewardManagerNemotron` already logs `"workplace assistant reward"` etc. — [`multi-domain-RL/other/dapo_overlong_penalty.py`](multi-domain-RL/other/dapo_overlong_penalty.py)) | If every rollout in a GRPO group scores identically (often 0), `A_i=0` for all of them by construction (§2 step 6, §3.5) — no gradient signal, regardless of `filter_groups` (which doesn't run anyway, §3.6). Fix the underlying reward-reachability problem (see next row), not the advantage math. |
| **Truncation masquerading as task failure** | A multi-turn/tool-use domain's reward plateaus; response length near the cap | `response_length/clip_ratio`, `num_turns/mean` vs. `multi_turn.max_assistant_turns`, `response/aborted_ratio` | If the agent structurally can't finish within `max_response_length`/`max_assistant_turns`, an all-or-nothing reward function ([`workplace_reward_fn`, multi-domain-RL/reward.py:593](multi-domain-RL/reward.py)) will always score 0 — raise the length/turn budget (Cascade 2's own stage-2 hyperparameter table specifies max response length **49K**, far above typical smoke-test configs of 4000-8000) or give partial credit in the reward function. |
| **Train/inference (rollout vs. trainer) numerical mismatch** | Reward increases in training but downstream eval is worse than expected; "on-policy" assumptions don't hold in practice | `training/rollout_probs_diff_mean/max/std`, `rollout_corr/k3_kl`, `chi2_token/seq`, `rollout_corr/rollout_is_eff_sample_size` if IS is enabled | Enable `calculate_log_probs=True` with `bypass_mode=False`/`rollout_is=null` first (diagnostics-only, §3.9) and read these metrics *before* enabling any correction. If confirmed large, correct via decoupled `rollout_is` (`bypass_mode` stays `False`) — an engineering correction for vLLM/FSDP mismatch, a separate question from GRPO's `ppo_epochs`/mini-batch on-policy guarantee (§2 step 7-9). |
| **Entropy collapse** | Policy becomes deterministic early, reward plateaus at a mediocre level, little further exploration | `actor/entropy` / `actor/entropy_loss` trend (sustained monotonic decline without recovery) | Per PRIME-RL (arXiv:2505.22617, §6), consider `clip_ratio_high > clip_ratio_low` (Clip-Higher, already true in Cascade's config: 0.28 vs 0.2) or the `clip_cov`/`kl_cov` loss types if collapse persists — see §6. |
| **Non-finite gradient silently skipping the update** | Training "runs" but a step contributes nothing, hard to notice without inspecting logs closely | `actor/grad_norm == nan` or `inf` | `optimizer_step()` zeroes grads and skips `optimizer.step()` on non-finite norm (§3.7) with no separate flag — you must watch `grad_norm` directly for `nan`/`inf`, not assume "no crash" means "update happened." |
| **`filter_groups` flag doing nothing** | Expecting DAPO-style dynamic resampling of degenerate groups; batch composition looks unaffected by the flag | No metric exists for this (confirmed — no "groups dropped"/"num_gen_batches" metric anywhere) | Confirmed by source: not wired into this checkout at all (§3.6). Either obtain the `recipe` submodule if you need real dynamic sampling, or stop relying on it — currently a pure no-op regardless of the value you set. |
| **`token-mean` letting one domain's long sequences dominate a mixed-domain batch** | Gradient updates seem to track the longest-response domain's behavior disproportionately | Per-domain `response_length/mean` compared against overall `actor/pg_loss` trend | Expected given `loss_agg_mode=token-mean` (§3.2) — not a bug, but worth knowing when interpreting why one domain's dynamics seem to dominate a shared update. Switch to `seq-mean-token-mean` if you want equal per-example weighting across domains (changes the loss semantics, not a free lunch). |
| **Singleton-group edge case in GRPO** | An advantage of exactly the raw reward value appears for an isolated sample | `critic/advantages` distribution has an outlier equal to a raw score, not group-relative | Only possible if a prompt ever ends up with just 1 rollout in its group ([`core_algos.py:315-317`](verl/verl/trainer/ppo/core_algos.py#L315-L317), mean=0/std=1 fallback) — not expected here since `rollout.n=16` is fixed and `filter_groups` doesn't shrink groups, but check if you ever change grouping logic. |
| **Sparse binary reward saturation (general)** | Reward stuck near 0 or 1 with little gradient signal either way | Per-domain reward mean/variance | All three Cascade domain reward functions here are strictly binary/all-or-nothing (§2 step 5) — inherent to the design, not a bug per se, but the least forgiving of the three (Workplace Assistant, requiring exact 5-way dataframe match) is the most exposed to this. |
| **Overlapped/async training staleness (if you ever switch trainer modes)** | Not currently applicable — default `trainer_mode=sync` has zero staleness by construction (§1.4-1.5) | `ReplayBuffer`'s `trajectory_staleness` (only relevant in `colocate_async`/`separate_async` modes) | If you ever deliberately move to an overlapped trainer mode for speed, re-check `rollout_corr`/`training_probs_diff` metrics — staleness reintroduces exactly the train/inference mismatch this section already covers, at a magnitude controlled by `max_off_policy_threshold`/`parameter_sync_step`. |

---

## 6. Research Papers Behind VERL's Algorithms

Each entry: the paper → what problem it solves → the exact VERL knob it maps to. Two entries have deliberately-flagged confidence caveats — read those notes before citing them as certain.

| Paper | Core idea | VERL mapping |
|---|---|---|
| **PPO** — Schulman et al. (OpenAI), "Proximal Policy Optimization Algorithms," 2017 (arXiv:1707.06347) | Replaces TRPO's expensive KL-constrained trust region with a first-order clipped-surrogate objective: clip the probability ratio to `[1−ε,1+ε]`, take the min of clipped/unclipped objectives. | `clip_ratio_low`/`clip_ratio_high`, [`compute_policy_loss_vanilla`](verl/verl/trainer/ppo/core_algos.py#L1278-L1369) (§3.1) |
| **GRPO** — Shao et al. (DeepSeek-AI), "DeepSeekMath," 2024 (arXiv:2402.03300) | Removes the learned value/critic network (expensive, ~2× memory/compute); replaces it with a group-relative baseline: `(r_i − mean_g)/std_g` over G samples per prompt. | `adv_estimator=grpo`, `norm_adv_by_std_in_grpo=True` (§3.5, [`core_algos.py:267-331`](verl/verl/trainer/ppo/core_algos.py#L267-L331)) |
| **Dr. GRPO** — Liu et al. (Sea AI Lab/NUS), "Understanding R1-Zero-Like Training," 2025 (arXiv:2503.20783) | Argues GRPO's std-normalization biases the gradient toward low-variance (easy/already-mastered) prompts and inflates length growth for wrong answers; removes the std term. | `norm_adv_by_std_in_grpo=False` — VERL's own docstring at [`core_algos.py:296`](verl/verl/trainer/ppo/core_algos.py#L296) cites this paper by name |
| **DAPO** — Yu et al. (ByteDance Seed/Tsinghua AIR), 2025 (arXiv:2503.14476) | Co-developed with VERL. Four mechanisms: **Clip-Higher** (`clip_ratio_high > low`, prevents suppressing rare-but-good tokens' upward growth — motivated by entropy collapse), **Dynamic Sampling** (drop/resample all-same-reward groups — VERL's `filter_groups`, confirmed **not wired up in this checkout**, §3.6), **token-level loss** (`loss_agg_mode=token-mean`), **overlong reward shaping** (soft length penalty, §3.10). | `clip_ratio_low/high`, `filter_groups` (unimplemented here), `loss_agg_mode=token-mean`, [built-in `dapo` reward manager](verl/verl/workers/reward_manager/dapo.py) |
| **GSPO** — Qwen Team, "Group Sequence Policy Optimization," 2025 (arXiv:2507.18071) | Token-level importance ratios accumulate independent noise across a long sequence, destabilizing training (esp. for MoE routing noise). Defines one length-normalized sequence-level ratio instead, clips/rewards at that granularity. | `loss_type=gspo`, `rollout_is="sequence"` mode — [`core_algos.py:1538-1611`](verl/verl/trainer/ppo/core_algos.py#L1538-L1611) cites this arXiv ID directly |
| **CISPO** — MiniMax, "MiniMax-M1," 2025 (arXiv:2506.13585) `[expansion "Clipped Importance Sampling Policy Optimization" corroborated by third-party docs, not the primary source itself — flagged as secondary]` | Standard clipping fully zeroes gradient for a clipped token, silently dropping signal from rare high-entropy "pivot" tokens (e.g. "wait", "however") that matter for chain-of-thought quality. Clips the IS *weight* under stop-gradient instead, leaving `log π_θ` always differentiable — no token fully masked out. | `loss_type=cispo` ([`core_algos.py`](verl/verl/trainer/ppo/core_algos.py) — structurally close to `reinforce` with a clipped, rather than unclipped, detached IS weight) |
| **IMPALA / V-trace** — Espeholt et al. (DeepMind), ICML 2018 | Classic precedent for off-policy correction in decoupled actor-learner systems: the behavior policy can lag the learner by several updates; corrects via two truncated IS ratios (ρ̄ for the policy gradient, c̄ for multi-step value bootstrapping). `[Note: LLM-RL with terminal-only rewards only inherits the ρ̄-style correction, not the c̄ bootstrapping machinery, since there's no value-bootstrap chain.]` | Conceptual ancestor of `rollout_is`'s truncation mechanism (§3.9) |
| **"When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"** — Liu et al., 2025 (richardli.xyz/rl-collapse) | Even "on-policy" RL can silently go off-policy because the inference engine (vLLM, low-precision/speculative kernels) computes slightly different token probabilities than the training engine for the *same* weights — a numerical mismatch, not staleness. Proposes sequence-level Truncated IS and Masked IS. | **Directly cited in [`rollout_corr_helper.py:59`](verl/verl/trainer/ppo/rollout_corr_helper.py#L59)** — the whole `rollout_is`/`rollout_rs` mechanism (§3.9) implements this paper's corrections |
| **IcePop** — Ling Team/Ant Group, "Ring-1T," 2025 (arXiv:2510.18855) | For MoE RL specifically: routing-induced train/inference discrepancies compound over long CoT, producing high-variance gradients. Two-sided token-level masking (drop tokens whose ratio is too large *or* too small) bounds gradient norm through training where unmasked baselines diverge after a few dozen steps. | The `"lower_upper"` threshold string mode of `rollout_is_threshold` (§3.9) — this is the mechanism behind Cascade 2 MOPD's ε_low=0.5/ε_high=2.0 |
| **Asynchronous RLHF** — Noukhovitch et al., ICLR 2025 (arXiv:2410.18252) | Studies how much off-policy staleness RLHF algorithms tolerate when generation and learning are decoupled for speed; finds robustness improves with policy scale. | Background for VERL's `colocate_async`/`separate_async` trainer modes (§1.4) |
| **AReaL** — Fu et al. (Tsinghua IIIS/Ant Group), 2025 (arXiv:2505.24298) | Fully decouples rollout from training; controls staleness with an explicit budget (their setup tolerates ~8 steps of lag) plus a staleness-aware PPO variant; reports 2.77× speedup at matched final performance. | Conceptual precedent for `max_off_policy_threshold` (default 8, [`ppo_trainer.yaml:231`](verl/verl/trainer/config/ppo_trainer.yaml#L231)) — same number, worth noting as likely not a coincidence |
| **AsyncFlow** — Han et al., 2025 (arXiv:2507.01663) | Systems paper (not an algorithm paper): a distributed transfer-queue + producer-consumer scheduler removing GPU idle bubbles when rollout/training are pipelined. | Architecture background for VERL's TransferQueue-based replay buffer (§1.4, [`v1/replay_buffer.py`](verl/verl/trainer/ppo/v1/replay_buffer.py)) |
| **The Entropy Mechanism of RL for Reasoning LMs** — Cui et al. (PRIME-RL team), 2025 (arXiv:2505.22617) | Policy entropy drops sharply early in RLVR training across essentially all runs; fits `R = −a·exp(H)+b` (accuracy gains are "purchased" from entropy, hitting a ceiling). Driven by positive covariance between a token's probability and its advantage-correlated logit update. Proposes Clip-Cov (drop the loss term for a random slice of high-covariance tokens) and KL-Cov (KL-penalize them instead). | `loss_type=clip_cov`/`kl_cov` ([`core_algos.py:1822`](verl/verl/trainer/ppo/core_algos.py#L1822), [`core_algos.py:1882-1915`](verl/verl/trainer/ppo/core_algos.py#L1882-L1915) — docstrings cite this repo/paper directly) |
| **"Clip-Low Increases Entropy and Clip-High Decreases Entropy..."**, 2025 (arXiv:2509.26114) | Isolates the independent effect of each clip bound on entropy — useful empirical companion when tuning `clip_ratio_low`/`clip_ratio_high` against entropy-collapse symptoms. | `clip_ratio_low`/`clip_ratio_high` (§3.1) |
| **DPPO ("Rethinking the Trust Region in LLM RL")**, 2026 (arXiv:2602.04879) `[very recent — flagged as bleeding-edge, less battle-tested than DAPO/GSPO]` | Argues PPO's ratio-clip is structurally ill-suited to LLMs' huge vocabularies (over-penalizes low-probability/high-exploration-value tokens, under-penalizes high-probability ones). Replaces the ratio-clip trust region with a direct TV-distance (`dppo_tv`) or KL (`dppo_kl`) divergence estimate, approximated as a binary token-level mask. | `loss_type=dppo_tv`/`dppo_kl` ([`core_algos.py:1372-1535`](verl/verl/trainer/ppo/core_algos.py#L1372-L1535) — cites this arXiv ID and paper section directly) |
| **GPG** — AMAP-ML, "GPG: A Simple and Strong RL Baseline for Model Reasoning," 2025 (arXiv:2504.02546) `[note: a different, unrelated paper — arXiv:2510.03679, "Group Policy Gradient" — shares the acronym; don't conflate them]` | Plain policy-gradient baseline without PPO clipping, without a separate value function. | `loss_type=gpg` ([`core_algos.py:1700-1732`](verl/verl/trainer/ppo/core_algos.py#L1700-L1732) — cites `github.com/AMAP-ML/GPG` directly) |
| **REINFORCE** — Williams, "Simple statistical gradient-following algorithms...," Machine Learning, 1992 `[classical attribution, not a citation VERL's own docstring includes]` | The textbook policy-gradient estimator this whole family descends from. | `loss_type=reinforce` ([`core_algos.py:2271-2348`](verl/verl/trainer/ppo/core_algos.py#L2271-L2348), with optional detached IS weight) |

**Important schema note repeated from the top:** an online VERL doc page describes a `rollout_is_level`/`rollout_is_mode` schema that does **not** match this vendored copy. Always write config examples against `algorithm.rollout_correction.{rollout_is, rollout_is_threshold, rollout_rs, rollout_rs_threshold, bypass_mode}` as confirmed in [`verl/verl/trainer/config/algorithm/rollout_correction.yaml`](verl/verl/trainer/config/algorithm/rollout_correction.yaml) and your own launch scripts.

---

## 7. Off-Policy Sources — Complete Catalog

"Off-policy" gets used loosely. In this codebase there are **four distinct policy-mismatch axes**, each with its own causes, parameters, and metrics — none requires the others to be present. This section exists to answer one question precisely: *for any given run, which axis (if any) is active, and how would you know?*

### 7.1 Axis 1 — π_θ vs π_old (within-batch optimizer drift)

**What it is:** the policy being updated (π_θ, evolving mid-optimization) drifts away from π_old (the frozen anchor the PPO/GRPO ratio is computed against), because more than one gradient step is taken against the same rollout batch before π_old is refreshed.

**Cause / parameters:** `actor_rollout_ref.actor.ppo_epochs` > 1, and/or `actor_rollout_ref.actor.ppo_mini_batch_size` < the full rollout batch (`data.train_batch_size × actor_rollout_ref.rollout.n`) → together these set `total_num_iterations = (rollout_batch / mini_batch) × ppo_epochs` (§1.3/§2) — every iteration beyond the first optimizes against a π_θ that already moved.

**This is exactly what PPO/GRPO clipping exists to bound** — `clip_ratio_low`/`clip_ratio_high`/`clip_ratio_c` (§3.1) — so a well-clipped run tolerates some drift by design; the failure mode is clipping *saturating* (most tokens hit the clip boundary every step), not the drift's mere existence.

**Metrics:** `actor/ppo_kl` (rises across mini-batches/epochs if this is happening — mechanically pinned at ≈0 in the Cascade config specifically because `ppo_mini_batch_size × rollout.n == full rollout batch` and `ppo_epochs=1`, so there's only ever one iteration, §2 step 8), `actor/pg_clipfrac`, `actor/pg_clipfrac_lower`.

**Not itself a bug** — nonzero, growing `actor/ppo_kl` and `pg_clipfrac` are *expected* in any run with `ppo_epochs>1`; concerning only once clipping saturates.

### 7.2 Axis 2 — π_old vs π_rollout (rollout-engine vs training-engine numerical mismatch)

**What it is:** even with `ppo_epochs=1` (axis 1 eliminated), the *same weights* can produce slightly different log-probs depending on which engine computes them — vLLM/SGLang generation vs. FSDP/Megatron training forward pass. Causes: precision (bf16 rollout vs. fp32 training, or vice versa), different fused attention kernels between engines, LoRA adapters or quantization used only at inference time, tokenizer/chat-template edge cases in multi-turn rollouts.

**Parameters:**
- `actor_rollout_ref.rollout.calculate_log_probs` — turns on capture of vLLM's own per-token log-probs (diagnostics only, §3.9)
- `algorithm.rollout_correction.{rollout_is, rollout_is_threshold, rollout_rs, rollout_rs_threshold, bypass_mode, loss_type, rollout_is_batch_normalize}` (§3.9) — the actual correction mechanism, off by default (`rollout_is=null`)
- `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode` — catches one specific sub-cause: re-tokenization drift in multi-turn rollouts (§1.6)

**Metrics** (logged automatically once `calculate_log_probs=True` — no correction needs to be enabled to see these): `training/rollout_probs_diff_{mean,max,std}`, `training/rollout_actor_probs_pearson_corr` (§4.9), `rollout_corr/k3_kl`, `rollout_corr/kl`, `rollout_corr/chi2_token`, `rollout_corr/chi2_seq`, `rollout_corr/log_ppl_diff`, `rollout_corr/log_ppl_abs_diff`, `rollout_corr/training_ppl`, `rollout_corr/rollout_ppl`, `rollout_corr/ppl_ratio`; if `rollout_is` is enabled, additionally `rollout_corr/rollout_is_{mean,std,eff_sample_size,ratio_fraction_high,ratio_fraction_low,oob_ratio,batch_norm_factor}`.

**NOT a source of this mismatch, despite sounding related:** speculative decoding in the rollout engine (`spec_drafts`/`spec_accepts`/`spec_verifies`, logged via `compute_spec_decode_metrics`, `verl/trainer/ppo/ray_trainer.py:138-163`) — its verification step guarantees accepted tokens match the target model's own distribution exactly, so it's a throughput optimization, not an approximation; it doesn't introduce mismatch.

### 7.3 Axis 3 — π_rollout(generation time) vs π_train(consumption time): temporal staleness from async trainer modes

**What it is:** genuine calendar-time staleness — a trajectory was generated against weights that are no longer current by the time it's consumed for training, because generation and training are allowed to overlap.

**Parameters:**
- `trainer.v1.trainer_mode` — `sync` (default, zero staleness by construction, §1.4-1.5) vs. `colocate_async` vs. `separate_async`
- `parameter_sync_step` (default 4, [`ppo_trainer.yaml:225`](verl/verl/trainer/config/ppo_trainer.yaml#L225)) — how often weights actually sync in the async modes
- `trainer.v1.colocate_async.num_warmup_batches` (default 1) — how many generation batches are pre-seeded before training starts
- `max_off_policy_threshold` (default 8) and `max_off_policy_strategy` (`drop` default, or `wait`) — the replay buffer's bound on tolerable staleness (§1.4)
- Older experimental path: `verl.experimental.one_step_off_policy` / `fully_async_policy`, with `async_training.staleness_threshold` (0 = sync, >0 = async) and `async_training.partial_rollout`

**Metrics — exact formulas confirmed from source** ([`verl/verl/trainer/ppo/v1/trainer_base.py:1547-1571`](verl/verl/trainer/ppo/v1/trainer_base.py#L1547-L1571), all read exactly 0 in default `sync` mode):
- `training/off_policy/trajectory_staleness/{mean,max,min}` = `((global_steps−1) − max_global_steps) / parameter_sync_step` — how many parameter-sync steps the *freshest* weights in this trajectory lag behind the current policy.
- `training/off_policy/trajectory_staleness_worst/{mean,max,min}` = `((global_steps−1) − min_global_steps) / parameter_sync_step` — same, using the *oldest* weights the trajectory touched (worst case within one trajectory).
- `training/off_policy/trajectory_spans/{mean,max,min}` = `(max_global_steps − min_global_steps + 1) / parameter_sync_step` — how many **distinct model versions a single trajectory was generated across** (1 = generated entirely on one version). This is the direct metric for axis 4 below.
- `training/off_policy/dropped_samples` (count) and `training/off_policy/dropped_samples_staleness/{mean,max,min}` ([`verl/verl/trainer/ppo/v1/replay_buffer.py:154-181`](verl/verl/trainer/ppo/v1/replay_buffer.py#L154-L181)) — when `max_off_policy_strategy="drop"`, how many trajectories got discarded for exceeding `max_off_policy_threshold`, and how stale they were when dropped — useful for tuning the threshold (frequent drops right at the threshold suggest it's biting as intended; drops far past it suggest a slow consumer/generation imbalance worth investigating separately).

### 7.4 Axis 4 — Partial rollout: a single trajectory spans multiple policy versions

**What it is:** a special case of axis 3 — a weight sync happens *while a request is mid-generation*; the in-flight request is aborted and resumed (`abort_replicas`/`release_kv_cache_replicas`/`resume_kv_cache_replicas`, `checkpoint_engine/base.py`, §1.6), and if resumed on the new weights, the tokens generated before vs. after the swap came from two different policy versions **within the same trajectory** — a within-*sequence*, not just within-*batch*, mismatch.

**Parameters:** `async_training.partial_rollout` (experimental `fully_async_policy`); implicitly active whenever `colocate_async`/`separate_async` trainer modes use their abort/resume mechanism.

**Metric:** `training/off_policy/trajectory_spans` (above) — a value >1 directly confirms this happened for at least one trajectory in the batch.

### 7.5 Intentional off-policy training (not a bug to fix — MOPD-style)

Sometimes off-policyness is deliberate: a policy π_inf generates while a different π_train is optimized (e.g. distillation from a teacher, or a deliberately decoupled/streaming setup for throughput). The goal isn't to eliminate axis 2/3 mismatch but to **correct for it explicitly**:

- `rollout_is="token"` combined with `rollout_is_threshold` as a `"lower_upper"` string (e.g. `"0.5_2.0"`) — the two-sided masking mode (§3.9), the same mechanism NeMo-RL calls `icepop` and Cascade 2's MOPD stage uses with ε_low=0.5/ε_high=2.0 (see `NEMO_RL_GYM_REFERENCE.md` §6).
- Here, nonzero `rollout_corr/rollout_is_ratio_fraction_high/low` is *expected*, not a red flag — monitor `rollout_corr/rollout_is_eff_sample_size` instead, to confirm the correction isn't discarding so much of the batch that the effective sample size collapses.

### 7.6 `bypass_mode` — redefines what "old policy" means, doesn't itself create off-policyness

`bypass_mode=True` sets `old_log_probs := rollout_log_probs` (§3.9), collapsing axis 1's anchor (π_old) into axis 2's rollout policy (π_rollout). This doesn't introduce new off-policyness, but it **removes the buffer** that would otherwise keep axis 2 mismatch purely diagnostic — with `bypass_mode=True`, any rollout/train numerical mismatch now shows up *directly* inside the PPO/GRPO ratio and clip mechanism, rather than staying a side channel you can monitor and ignore. Combine with `loss_type=reinforce` + `rollout_is` if you want that mismatch actively corrected rather than let through uncorrected.

### 7.7 Quick reference — which metric to check for which suspicion

| Suspicion | Check first |
|---|---|
| "Is `ppo_epochs`/mini-batching letting the ratio drift mid-batch?" | `actor/ppo_kl`, `actor/pg_clipfrac` |
| "Is vLLM/FSDP disagreeing about the same weights?" | `training/rollout_probs_diff_mean`, `rollout_corr/k3_kl` |
| "Am I accidentally running an async/stale setup?" | `training/off_policy/trajectory_staleness/mean` (should be exactly 0 in default `sync` mode — nonzero means you're not where you think you are) |
| "Did any trajectory get generated across a weight swap?" | `training/off_policy/trajectory_spans/max` (>1 = yes) |
| "Is my `max_off_policy_threshold` well-tuned?" | `training/off_policy/dropped_samples`, `training/off_policy/dropped_samples_staleness/mean` |
| "Is my deliberate IS correction (MOPD-style) discarding too much data?" | `rollout_corr/rollout_is_eff_sample_size` |

---

## 8. File Map

Quick-reference index of every VERL-internal file cited above, for direct navigation.

**Core algorithm / loss math** (shared across both trainer generations):
- [`verl/verl/trainer/ppo/core_algos.py`](verl/verl/trainer/ppo/core_algos.py) — all policy losses, advantage estimators, KL estimators, clipping
- [`verl/verl/workers/utils/losses.py`](verl/verl/workers/utils/losses.py) — `ppo_loss()`, entropy/KL loss wiring
- [`verl/verl/trainer/ppo/metric_utils.py`](verl/verl/trainer/ppo/metric_utils.py) — reward/advantage/length/timing/throughput metric computation
- [`verl/verl/utils/debug/metrics.py`](verl/verl/utils/debug/metrics.py) — `rollout_probs_diff_*` diagnostics
- [`verl/verl/trainer/ppo/rollout_corr_helper.py`](verl/verl/trainer/ppo/rollout_corr_helper.py) — IS/rejection-sampling correction, off-policy diagnostics

**Trainer orchestration:**
- [`verl/verl/trainer/ppo/ray_trainer.py`](verl/verl/trainer/ppo/ray_trainer.py) — legacy trainer (`@deprecated`)
- [`verl/verl/trainer/ppo/v1/trainer_base.py`](verl/verl/trainer/ppo/v1/trainer_base.py) — v1 base trainer (live default)
- [`verl/verl/trainer/ppo/v1/trainer_sync.py`](verl/verl/trainer/ppo/v1/trainer_sync.py) — default `sync` mode
- [`verl/verl/trainer/ppo/v1/trainer_colocate_async.py`](verl/verl/trainer/ppo/v1/trainer_colocate_async.py) / [`trainer_separate_async.py`](verl/verl/trainer/ppo/v1/trainer_separate_async.py) — overlapped modes
- [`verl/verl/trainer/ppo/v1/replay_buffer.py`](verl/verl/trainer/ppo/v1/replay_buffer.py) / [`agent_loop_tq.py`](verl/verl/trainer/ppo/v1/agent_loop_tq.py) — TransferQueue staleness machinery

**Config schemas:**
- [`verl/verl/trainer/config/ppo_trainer.yaml`](verl/verl/trainer/config/ppo_trainer.yaml) — top-level trainer defaults
- [`verl/verl/trainer/config/algorithm.py`](verl/verl/trainer/config/algorithm.py) / [`algorithm/rollout_correction.yaml`](verl/verl/trainer/config/algorithm/rollout_correction.yaml) — `rollout_is`/`bypass_mode`/`filter_groups` schemas
- [`verl/verl/trainer/config/actor/dp_actor.yaml`](verl/verl/trainer/config/actor/dp_actor.yaml) — actor defaults incl. `grad_clip`
- [`verl/verl/workers/config/actor.py`](verl/verl/workers/config/actor.py) / [`rollout.py`](verl/verl/workers/config/rollout.py) / [`optimizer.py`](verl/verl/workers/config/optimizer.py) — worker-level config dataclasses

**Rollout / generation:**
- [`verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py`](verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py) / [`vllm_async_server.py`](verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py) — vLLM async engine integration
- [`verl/verl/workers/rollout/schemas.py`](verl/verl/workers/rollout/schemas.py) — conversation state, `loss_mask` construction, tokenization sanity check
- [`verl/verl/experimental/agent_loop/agent_loop.py`](verl/verl/experimental/agent_loop/agent_loop.py) — async per-sample dispatch, `response_mask` assembly

**Training engine:**
- [`verl/verl/workers/engine/base.py`](verl/verl/workers/engine/base.py) — `train_batch` (zero_grad → forward_backward → optimizer_step)
- [`verl/verl/workers/engine/fsdp/transformer_impl.py`](verl/verl/workers/engine/fsdp/transformer_impl.py) — FSDP forward/backward, gradient clipping, temperature scaling
- [`verl/verl/workers/engine_workers.py`](verl/verl/workers/engine_workers.py) — `train_mini_batch` (ppo_epochs × mini-batch loop)
- [`verl/verl/utils/seqlen_balancing.py`](verl/verl/utils/seqlen_balancing.py) — dynamic-bsz Karmarkar-Karp packing, `global_seqlen` metrics

**Weight sync / staleness:**
- [`verl/verl/checkpoint_engine/base.py`](verl/verl/checkpoint_engine/base.py) — `CheckpointEngine` abstraction, `update_weights`
- [`verl/verl/checkpoint_engine/nccl_checkpoint_engine.py`](verl/verl/checkpoint_engine/nccl_checkpoint_engine.py) — network broadcast backend

**Ray / placement:**
- [`verl/verl/single_controller/ray/base.py`](verl/verl/single_controller/ray/base.py) — `RayWorkerGroup`, `RayResourcePool`, blocking dispatch
- [`verl/verl/trainer/ppo/utils.py`](verl/verl/trainer/ppo/utils.py) — worker `Role` enum

**Reward:**
- [`verl/verl/workers/reward_manager/dapo.py`](verl/verl/workers/reward_manager/dapo.py) — built-in core-VERL DAPO overlong penalty
- [`verl/verl/experimental/reward_loop/reward_manager/dapo_overlong_penalty.py`](verl/verl/experimental/reward_loop/reward_manager/dapo_overlong_penalty.py) — the registered custom manager your launch scripts actually use
- [`verl/verl/utils/torch_functional.py`](verl/verl/utils/torch_functional.py) — entropy formula (note: `compute_grad_norm()` here has zero callers, disregard)

**Docs shipped with VERL:**
- [`verl/docs/advance/one_step_off.md`](verl/docs/advance/one_step_off.md) — one-batch-ahead async recipe
- [`verl/docs/advance/fully_async.md`](verl/docs/advance/fully_async.md) — fully decoupled async recipe

**Project files (this Cascade 2 reproduction, not VERL itself):**
- [`multi-domain-RL/launch.sh`](multi-domain-RL/launch.sh) — the actual config traced throughout §2
- [`multi-domain-RL/reward.py`](multi-domain-RL/reward.py) — `multi_domain_reward_fn` and per-domain reward functions
- [`multi-domain-RL/tools.yaml`](multi-domain-RL/tools.yaml) — Workplace Assistant tool schemas
- [`multi-domain-RL/other/dapo_overlong_penalty.py`](multi-domain-RL/other/dapo_overlong_penalty.py) — `DAPORewardManagerNemotron`, per-domain reward logging
- [`multi-domain-RL/AGENTS.md`](multi-domain-RL/AGENTS.md) / [`MOPD/AGENTS.md`](MOPD/AGENTS.md) / [`RLHF/AGENTS.md`](RLHF/AGENTS.md) — per-stage debugging status and TODOs
- [`Cascade2.pdf`](Cascade2.pdf) — the Nemotron Cascade 2 paper
