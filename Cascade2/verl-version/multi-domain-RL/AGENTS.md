# AGENTS.md — multi-domain-RL

## 1. Purpose

This is Stage 2 of the Nemotron Cascade 2 reproduction (see `Cascade2/Cascade2.pdf`, §4.3): a single GRPO RL stage blending three capabilities in one training run —

- **MCQA** (STEM multi-choice) — ~55% of the paper's mixture
- **Agentic tool calling** (Workplace Assistant) — ~30%
- **Structured output** (instruction-following JSON schema) — ~15%

The paper blends domains within a batch rather than training them sequentially, citing (a) no cross-domain evaluation degradation and (b) similar response length / verification time across domains, which avoids one domain stalling the others. Stage 1 (IF-RL) is fixed (was a `model_dtype=bfloat16` precision issue mirroring TRL's `bf16=True`; fixed by setting `fsdp_config.model_dtype=fp32` — both `launch.sh` here and `MOPD/launch.sh` already carry this fix).

Current status: the combined run executes and shows healthy MCQA/structured-outputs learning, but **Workplace Assistant reward is stuck at a low plateau**, and a domain-isolated MCQA run (see `MOPD/AGENTS.md`) shows anomalous `grad_norm` spikes to ~4. As of 2026-08-04, real wandb logs for these runs are now available locally (`multi-domain-RL/wandb/`) and have been parsed (see §4 below) — most of what follows was previously hypothesis, now confirmed or refuted against actual training curves. See also `multi-domain-RL/DOMAINS.md` for the dataset/reward-mechanism deep dive per domain (schema, MCQA regex catalog, tool list, capability mapping).

## 2. Current mechanism

### Reward (`reward.py`, dispatch via `extra_info["agent_ref"]` in `multi_domain_reward_fn`, line 28)

All three domain reward functions are **strictly binary, no partial credit**:

- `mcqa_reward_fn` (line 201): regex-extracts the answer letter (several fallback patterns), compares to ground truth. 1.0 / 0.0.
- `structured_reward_fn` (line 339): parses model output as JSON, validates against the ground-truth JSON schema via `openapi_schema_validator`. Any parse/validation failure → 0.0.
- `workplace_reward_fn` (line 593): replays `predicted_actions` (populated by `WorkplaceTool.execute()` during the tool-agent rollout) against ground-truth actions in fresh tool environments, then requires an **exact pandas `.equals()` match across all 5 toolkit dataframes simultaneously** (calendar, email, analytics, project_management, CRM). A mismatch in *any one* domain zeroes the whole reward. This is the most demanding of the three reward functions by a wide margin — a single incomplete sub-task (e.g. from running out of turns) fails the entire reward, whereas MCQA/structured only need to get one thing right.

Reward manager: `reward.reward_manager.name=dapo_overlong_penalty` → `DAPORewardManagerNemotron` (`other/dapo_overlong_penalty.py`). If `overlong_penalty.enable=True` and a response hits `max_resp_len` without an EOS token, it force-zeros the reward regardless of task score, and logs per-domain reward/count breakdowns keyed off `agent_ref` (`"workplace assistant reward"`, `"mcqa reward"`, etc.). **Currently explicitly disabled** in this `launch.sh` (`+reward.reward_kwargs.overlong_penalty.enable=False`), so truncated responses are not being force-zeroed by this mechanism.

### Masking (VERL core, `verl/workers/rollout/schemas.py` + `verl/experimental/agent_loop/agent_loop.py`)

Confirmed from source: assistant-generated tokens get `loss_mask=True` (`add_assistant_message`, schemas.py:428-451); tool-observation tokens get `loss_mask=False` (`add_tool_response_messages`, schemas.py:453-514). The final `response_mask` built in `agent_loop.py:727-778` is this loss_mask × attention_mask, and it's used uniformly for the policy-gradient loss, entropy/KL aggregation, and every IS-ratio/rejection computation in `rollout_corr_helper.py`. **Tool-observation tokens are correctly excluded from both the loss and any IS averaging** — this is not a suspect for the reward-plateau issue.

### Batch / optimizer semantics

`train_batch_size=128`, `rollout.n=16` → rollout batch = 2048 trajectories. `ppo_mini_batch_size=128` (multiplied internally by `rollout.n` → 2048, i.e. mini-batch == full rollout batch) with `ppo_epochs=1` → confirmed in VERL source (`ray_trainer.py:1290-1338`, `engine_workers.py:233-378`, `engine/base.py:113-132`) that this yields **exactly one `optimizer_step()` per rollout batch**. `use_dynamic_bsz=True` / microbatching only affects gradient-accumulation grouping for memory, not optimizer step count.

`+algorithm.filter_groups.enable=True` (metric=`acc`, `max_num_gen_batches=10`) — **CORRECTION, confirmed by direct source grep**: `FilterGroupsConfig` is defined in `verl/trainer/config/algorithm.py` but is not consumed anywhere else in this VERL checkout — no code in `ray_trainer.py`'s `fit()` reads `algorithm.filter_groups.enable`, drops degenerate groups, or triggers extra generation rounds, and no metric reports groups-filtered/extra-gen-batches. The actual DAPO dynamic-sampling loop lives in an external `recipe` git submodule (`verl-recipe.git`) that is **not checked out** in this repo (`ls verl/recipe/` is empty). **This means `+algorithm.filter_groups.enable=True` in this `launch.sh` is currently a silent no-op** — it is not dropping or resampling any groups, and is not the mechanism behind the reward plateau. (Two independent source-tracing passes confirmed this identically.)

### Rollout correction / `rollout_is`

Current: `algorithm.rollout_correction.{rollout_is,rollout_rs}=null`, `bypass_mode=False`, `actor_rollout_ref.rollout.calculate_log_probs=True`. Traced precisely against the pulled VERL fork (`Cascade2/verl-version/verl/`, commit `9481350e`, this rollout-correction code is unmodified upstream):

| Setting | What it does | File:line |
|---|---|---|
| `rollout_is: null` / `"token"` / `"sequence"` | off / per-token ratio π_θ/π_rollout / per-sequence product ratio, multiplies the pg-loss term, always detached before use | `rollout_corr_helper.py:520-655` |
| `bypass_mode` | `True`: sets `old_log_probs := rollout_log_probs`, identifying π_old with π_rollout and skipping the extra actor forward pass. `False` (current): 3 distinct policies (π_rollout, π_old, π_θ) | `ray_trainer.py:1533-1546`, `rollout_corr_helper.py:1107-1143` |
| `loss_type: ppo_clip` vs `reinforce` | Only meaningful under `bypass_mode=True`. `ppo_clip` explicitly does **not** also apply `rollout_is_weights` (comment in source: would double-count the correction already done by clipping). `reinforce` does apply it, detached. | `core_algos.py:2351-2486` |
| `calculate_log_probs` | Pure compute-and-expose switch (records vLLM's own per-token log-probs). Zero effect on gradients **as long as** `bypass_mode=False` and `rollout_is=null` (current defaults) — in that state it only feeds diagnostic metrics below. | `verl/workers/config/rollout.py:218` |

Because `calculate_log_probs=True` and `bypass_mode=False` are already set here, VERL **automatically logs, every step, with no extra experiment needed**:
`training/rollout_probs_diff_mean/max/std`, `training/rollout_actor_probs_pearson_corr` (`verl/utils/debug/metrics.py:63-121` — note: despite the name, this is `|exp(actor_logprob) − exp(rollout_logprob)|`, i.e. raw probability difference, not log-space or KL), and `rollout_corr/k3_kl` (`rollout_corr_helper.py:902-1008`, K3 KL estimator `exp(log_ratio) − log_ratio − 1`).

**Cascade 2's own claim** (paper §4.1.2): the GRPO stages (IF-RL, this multi-domain stage, RLHF) are **strictly on-policy by construction** — one gradient update per rollout batch, `π_θ/π_old` ratio exactly 1, KL term removed entirely (reduces the objective to plain REINFORCE with group-normalized rewards). This claim is about `π_θ/π_old`, which the VERL trace above confirms VERL delivers mechanically (single optimizer step). It says **nothing** about `π_rollout` (vLLM) vs. the FSDP trainer's own numerics — that's a separate, purely engineering question, answered by the metrics above, not by the paper.

Contrast: **MOPD (§4.4) is the one stage that is genuinely off-policy by design** (π_inf generates, π_train optimizes) and the paper explicitly reintroduces **truncated importance weighting** there (ε_low=0.5, ε_high=2.0, Eq. 3) — a real precedent for when `rollout_is` is needed, but it doesn't apply to this strictly-on-policy stage.

A `bypass_mode=True` + `loss_type=reinforce` + `rollout_is=token` configuration was considered as "the most coherent implementation" of Cascade's REINFORCE objective. It is **confirmed valid and reachable in the pulled VERL fork** — no assertion blocks it, and there's a named preset doing almost exactly this (`RolloutCorrectionConfig.bypass_pg_token_icepop()`, `verl/trainer/config/algorithm.py:369-391`), with the IS weight correctly detached (`rollout_corr_helper.py:623`, plus `torch.no_grad()` in `core_algos.py:2434`). **However, this is parked, not adopted**: `bypass_mode=True` substitutes what "old policy" means (`old_log_probs := rollout_log_probs`), which is a first-order deviation from Cascade's setup (which keeps π_old distinct via the decoupled path with `rollout_is=null`), not a reproduction of it. Revisit only if the diagnostic metrics above come back showing meaningful vLLM/FSDP mismatch.

## 3. Training run analysis (confirmed from real wandb logs, 2026-08-04)

Parsed directly from the binary `.wandb` datastore files in `multi-domain-RL/wandb/` (parsed offline via a throwaway venv + `wandb.sdk.internal.datastore` — no network sync, see run catalog below). This replaces the untested hypotheses previously in this section.

**Run catalog** (identity confirmed via checkpoint path in `output.log`, not the generic slurm `job_name`):

| Run(s) | Identity | Progress |
|---|---|---|
| `dxqyxpo4` | MCQA-only isolation | 9/10 steps |
| `fkimhgx3` → `mzv745vs` | Workplace-only isolation (standard 4000-token response length) | 9 + 48 steps |
| `65kn4j2p` → `nb5ctv2w` | Structured-outputs-only isolation | 4 + 8 steps |
| `z46psk6k` | Combined multi-domain (the "correct" bugfixed run), first with per-domain reward breakdown | 69/70 steps |
| `dj0i9zrv` → `75r9icmc` → `zvyidgbc` (= `latest-run`) | Workplace response-length experiment (max_response_length raised to 8000) | 0 + 2 + 48 steps |

### 3.1 MCQA-only `grad_norm` spike — confirmed cause, and confirmed NOT present in the blended run

Real per-step numbers from `dxqyxpo4`:

| step | grad_norm | response_length/mean | critic/score/mean (acc) | global_seqlen/mean |
|---|---|---|---|---|
| 1 | 0.66 | 69.0 | 0.335 | 200,850 |
| 4 | 1.14 | 29.5 | 0.481 | 178,799 |
| 6 | 2.49 | 14.3 | 0.459 | 172,442 |
| 8 | 4.11 | 8.9 | 0.438 | 167,882 |

**Confirmed: `grad_norm` climbs monotonically (0.66→4.11) in lockstep with `response_length/mean` collapsing (69→8.9 tokens) over just 8 steps**, while accuracy actually *improves* (0.335→0.48) — this is not noise, it's a real, sustained trend. Mechanism: `loss_agg_mode=token-mean` divides the total policy-gradient loss by the **global valid-token count** across the mini-batch (`VERL_MECHANICS.md` §3.2) — as responses collapse toward terse boilerplate (the model rapidly learns it doesn't need "reasoning" tokens to satisfy the regex-based reward, only the final answer pattern), the same total loss signal gets divided by a shrinking denominator, mechanically inflating the per-token — and hence total — gradient norm. `global_seqlen/mean` falling in step with `response_length/mean` confirms the token-count collapse directly.

**Critically, this same collapse does NOT happen when MCQA is trained inside the blended `multi_domain` run.** In `z46psk6k`, MCQA's own validation reward climbs smoothly (0.243→0.277 over 30 steps, then plateaus ~0.26-0.27) while `actor/grad_norm` for the whole run stays noisy but bounded (0.13–1.4, no sustained climb) across all 69 steps. **The instability is specific to the isolated MCQA-only ablation setup — likely because it's a very short (10-step), narrow, low-diversity training run that lets the policy overfit/sharpen rapidly on a small prompt distribution — not a property of the MCQA domain itself or of the multi-domain recipe.** Structured-outputs-only (`65kn4j2p`/`nb5ctv2w`, a similarly short single-domain ablation) shows no such instability either (`grad_norm` stays in 0.08–0.14, accuracy climbs 0.20→0.44 cleanly) — ruling out "any short single-domain isolation run is inherently unstable" as the explanation.

### 3.2 Workplace Assistant reward plateau — response length was NOT the binding constraint

This overturns what was previously "Hypothesis A" in this doc. Real data from both the standard-length workplace-only run (`fkimhgx3`/`mzv745vs`) and the response-length-extended run (`dj0i9zrv`→`75r9icmc`→`zvyidgbc`):

**`response_length/clip_ratio` stays at the observation floor (~0.0005, i.e. essentially zero responses ever hit the length cap) throughout every workplace run, standard or extended.** Truncation from hitting `max_response_length` was never actually happening at meaningful rates — the length-budget hypothesis is refuted as the primary driver.

What actually happens instead, and it differs between the two ablations:

- **Standard length (4000 tokens, `fkimhgx3`→`mzv745vs`):** `num_turns/mean` and `response_length/mean` **grow steadily** over training (turns 3.7→5.0, length 250→450 tokens) while `actor/entropy` **declines steadily and cleanly** (0.60→0.22-0.28 by step 40+) — a textbook entropy-collapse shape, not a truncation signature.
- **Extended length (8000 tokens, `zvyidgbc`):** `num_turns/mean` and `response_length/mean` **rise for the first ~10 steps** (turns 3.7→4.3, length 257→467 — this is the "num_turns looked better in the beginning" the user observed) **then collapse** for the rest of training (turns down to ~2.5-3.2, length down to ~110-165 by step 48; `num_turns/max` falls from a ceiling of 12 down to a floor of mostly 4). **And yet `workplace assistant reward` climbs slowly and steadily throughout this collapse** (0.110→0.125→0.129→0.133→0.137) — reward is *not* getting worse as completions shrink, it's improving. This means the model isn't failing due to truncation; it's converging toward shorter, more efficient completions that solve *some* tasks well enough, while entropy in this run stays noisier/higher in later steps (frequently 0.5-0.7+) than the standard-length run's clean decay.

**Best-supported explanation, tying this to the dataset:** the workplace ground-truth action-count distribution is heavily skewed toward trivial tasks (0 actions: 513/4968 rows, 1 action: 3084/4968 — i.e. **~72% of tasks need 0-1 tool calls**; only 9/4968 need as many as 8). A policy converging toward fewer, shorter turns is arguably fitting the dominant mode of the task distribution correctly — it doesn't need to fully use the length/turn budget to solve most examples. This explains why extending `max_response_length` produced a real but modest reward gain (workplace reward ~0.045-0.062 in the standard combined run vs. ~0.11-0.14 in the extended-length run — roughly **2x higher**, a genuine effect, just not the dramatic unlock the length-budget hypothesis alone would predict) — a longer budget helps the rare long-tail multi-action examples, but those are too infrequent to move the aggregate reward much, and don't require the policy to *keep* using long completions once it has learned the easy majority.

**GRPO advantage-collapse mechanism (previously "Hypothesis B") still holds, just for a different underlying reason than truncation:** within the blended `z46psk6k` run, workplace assistant reward is stuck at a genuinely low, barely-moving mean (0.046→0.062 over 65 steps, the flattest of the three domains against MCQA's 0.24→0.27 and structured-output's 0.086→0.135) — consistent with most 16-rollout groups landing on all-zero reward (mean_g=0 → advantage=0 for the whole group, per `core_algos.py:267-331`) and only a minority of "easy" (0-1 action) groups contributing any gradient at all. This is a low-signal-density problem, not a truncation problem.

### 3.3 `rollout_is` question — resolved, empirically, from real data

Pulled directly from `z46psk6k`'s 69 logged steps: `training/rollout_probs_diff_mean` = **0.0015–0.0030** (mean 0.00198), comfortably under the 0.005 heuristic threshold noted earlier in this doc; `rollout_corr/k3_kl` = **0.0002–0.0014** (mean 0.00036), consistent with the code's own "ideal ≈ 0" comment (`rollout_corr_helper.py:210`); `training/off_policy/trajectory_staleness/mean` = **exactly 0.0** every step, confirming the default `sync` trainer mode has zero staleness as predicted in `VERL_MECHANICS.md` §1.5.

**Conclusion: `rollout_is: null` is empirically justified, not just the literal Cascade reproduction.** The vLLM/FSDP mismatch is negligible in this setup. No further action needed on the rollout_is question — the previously-parked `bypass_mode + reinforce + token IS` idea can stay parked; there is no measured mismatch to correct.

## 4. Open issues / TODOs

1. **~~Workplace Assistant reward not increasing~~ — resolved, see §3.2.** Not a truncation problem (`response_length/clip_ratio` stays ~0 throughout); it's a low-signal-density problem from a heavily 0-1-action-skewed task distribution combined with GRPO's all-zero-group advantage collapse. Remaining open question: **is there a way to raise the density of non-degenerate groups** (e.g. curriculum toward harder multi-action examples, per-domain advantage weighting via `gdpo`, or partial-credit reward shaping for near-miss DataFrame states) without abandoning the strict-correctness reward design. Not yet attempted.
2. **~~MCQA-only `grad_norm` spike~~ — resolved, see §3.1.** Confirmed as an isolated-ablation artifact (response-length collapse under `token-mean` loss aggregation), absent when MCQA trains inside the blended batch. No action needed on the real training recipe; if the MCQA-only smoke test in `MOPD/` is still useful going forward, consider training it for more steps on a larger/more diverse prompt slice to avoid the same rapid-overfit dynamics, or just treat short single-domain smoke tests as expected to look unstable and not diagnostic of the real recipe.
3. **~~`rollout_is` question~~ — resolved, see §3.3.** `rollout_is: null` is empirically justified; no correction needed.
4. **Overlong filtering divergence from paper**: paper specifies stage-2 overlong filtering = `True` (Table 8); current `launch.sh` explicitly disables the custom overlong penalty (`overlong_penalty.enable=False`). Still open — worth reconciling, but not yet investigated as a cause of the reward plateau (truncation isn't being force-zeroed currently, so it isn't hiding the workplace signal, consistent with §3.2's finding that truncation isn't the driver anyway).
5. **Stale/inconsistent draft files**: `Cascade2/verl-version/config_if_rl.yaml`, `config_multi_domain_rl.yaml`, `launch_if_rl.sh`, `launch_multi_domain_rl.sh`, `tools.py` (at `verl-version/` root, untracked) do not match this working `launch.sh` — e.g. they set `ppo_mini_batch_size: 2048`, which VERL multiplies by `rollout.n` internally (→ 32768 against a 2048-trajectory rollout batch, `total_num_iterations` would floor-divide toward zero); `launch_if_rl.sh` sets `train_batch_size=2048` where its own paired yaml says 128; `launch_multi_domain_rl.sh` is empty; `tools.py` is an incomplete stub with a dangling import. **Do not treat these as reference for this stage** until fixed or removed — the real config is this `launch.sh`.
6. Do not edit anything under `Cascade2/verl-version/verl/` — it's a separate nested repo (upstream VERL fork, commit `9481350e`) with its own `AGENTS.md`/`CLAUDE.md`/contribution policy. Only cite it as source-of-truth reference.
