# Multi-turn math + calculator-tool RL (VeRL)

A from-scratch VeRL pipeline that trains Qwen2.5-1.5B-Instruct with GRPO to
solve hard math problems, using VeRL's *native* multi-turn tool-calling
agent loop (`agent.default_agent_loop=tool_agent`) -- the model can actually
invoke a calculator tool mid-rollout and see its real output, rather than
faking tool use in a single generation pass (contrast with
`Cascade2/verl-version/multi-domain-RL`, which does the latter).

Train set: `hendrycks/competition_math` (7,500 problems). Eval set:
`HuggingFaceH4/MATH-500`, held out and never trained on, wired in as VeRL's
native `data.val_files` so every checkpoint gets an in-loop MATH-500
accuracy number with no merge step required.

## Files

- `prepare_data.py` -- builds `math_calculator_train.parquet` /
  `math_calculator_val.parquet` in VeRL's expected schema. Run this first,
  on a machine with real Hugging Face Hub access.
- `tools/calculator_tool.py` -- the `calculator` tool, registered via VeRL's
  `@function_tool` decorator. Uses an `ast`-based whitelist evaluator, not
  `eval()`.
- `reward.py` -- `math_tool_reward_fn`: extracts the boxed final answer and
  grades it against ground truth with `math-verify` (handles
  algebraic/fraction/decimal equivalence, not just string match).
- `chat_template.j2` -- reference copy of the Hermes-style tool-calling
  template Qwen2.5-Instruct already ships by default. Not wired into the
  launch script (no override needed for this model); kept in case this
  pipeline is ever pointed at a model without a native tool-calling template.
- `launch_math_tool_rl.sh` -- the VeRL launch command. Multi-GPU FSDP,
  `actor_rollout_ref.rollout.multi_turn.enable=True`, `function_tool_path`
  pointed at the calculator tool, `+actor_rollout_ref.model.override_config.
  tie_word_embeddings=false` (see "Known risks" below).
- `check_tied_embeddings.py` -- run against any merged checkpoint to confirm
  its `embed_tokens`/`lm_head` weren't silently split/diverged by FSDP.

## Setup

```bash
python prepare_data.py
pip install math-verify[antlr4_13_2]
```

VeRL's tool config also needs the calculator tool importable -- either run
from this directory, or make sure `Multi-turn/` is on `PYTHONPATH`.

## Recommended run order

1. **Tool-firing smoke test** -- confirm the calculator is actually being
   invoked mid-rollout before spending real GPU time:
   ```bash
   TOTAL_STEPS=3 ./launch_math_tool_rl.sh
   ```
   Watch for the `agent_loop/tool_calls` metric being > 0, and/or add a
   `print()` inside `calculator_tool.py`'s `calculator()` to confirm it's
   hit. There's an open, unresolved VeRL issue
   ([volcengine/verl#2986](https://github.com/volcengine/verl/issues/2986))
   where multi-turn tool calls silently fail to fire under vLLM/async mode
   (reported on an older verl+vLLM combo; unconfirmed whether it affects
   this install's newer vLLM). If tool calls never fire here, that's this
   issue manifesting -- reconsider SGLang as the rollout engine before going
   further.

2. **Calculator correctness check** -- sanity check `safe_eval()` in
   `tools/calculator_tool.py` against a handful of hand-computed expressions.

3. **Short trial run** (~20-30 steps):
   ```bash
   TOTAL_STEPS=30 ./launch_math_tool_rl.sh
   ```
   Watch the in-loop MATH-500 validation accuracy, `actor/grad_norm`, and
   `actor/entropy`. A flat/non-moving reward curve here is the earliest and
   cheapest signal of a deeper problem -- see "Known risks" below for what
   that turned out to mean on a sibling VeRL pipeline in this repo.

4. Full run: `./launch_math_tool_rl.sh` (defaults to 200 steps; override
   with `TOTAL_STEPS=<n>`).

## Known risks (carried over from the Cascade2 IF-RL investigation)

Qwen2.5-1.5B-Instruct ties `embed_tokens`/`lm_head`. Under multi-GPU FSDP,
this pairing has a documented failure class where the two can silently
desync during training and/or a checkpoint merge can save them wrong (see
[huggingface/accelerate#3870](https://github.com/huggingface/accelerate/issues/3870),
[volcengine/verl#2262](https://github.com/volcengine/verl/issues/2262)).
`launch_math_tool_rl.sh` proactively disables the tie
(`override_config.tie_word_embeddings=false`) as cheap insurance; run
`check_tied_embeddings.py` against any merged checkpoint as a second line of
defense. There's also an open, unresolved VeRL issue
([verl-project/verl#5308](https://github.com/verl-project/verl/issues/5308))
suggesting full-parameter FSDP training may need a different effective
learning rate than an equivalent single-GPU run to actually learn -- if the
reward curve looks flat/weak in step 3 above despite the tied-embedding
mitigation, an LR sweep is the next thing to try.
