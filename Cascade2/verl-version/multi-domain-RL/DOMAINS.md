# DOMAINS.md — Multi-Domain RL: Dataset, Reward, and Capability Reference

Detailed reference for the three domains blended in this stage's training data, per Nemotron Cascade 2 §4.3 (`Cascade2.pdf`): **MCQA** (STEM, ~55%), **Workplace Assistant** agentic tool calling (~30%), **Structured Outputs** (~15%). See `AGENTS.md` for training status/TODOs — this doc is the "what is this data and why" reference, kept separate so `AGENTS.md` stays focused on status.

Source of the mixture: `nvidia/Nemotron-Cascade-2-RL-data`, HuggingFace config `"multi-domain-RL"`, built by `prepare_data.py` in this folder. Verified empirically against the materialized `data/multi-domain-RL-train.parquet` (17,239 rows): **mcqa**=9,576 (55.5%), **workplace_assistant**=4,968 (28.8%), **structured_outputs**=2,695 (15.6%) — matches the paper's stated 55/30/15 split closely.

---

## 1. Shared dataset envelope

Every row in the parquet has the same 7 top-level columns, populated differently per domain by `format_data_multi_domain()` (`prepare_data.py:221-233`), which dispatches on the raw HuggingFace column `data["agent_ref"]["name"]`:

| Column | Meaning |
|---|---|
| `agent_name` | `"single_turn_agent"` (MCQA, structured outputs) or `"tool_agent"` (workplace assistant) — controls which VERL agent-loop handler runs the rollout |
| `data_source` | always `"nvidia/Nemotron-Cascade-2-RL-data"` |
| `prompt` | list of chat messages (system + user); 2 messages for MCQA/structured, occasionally more for workplace |
| `tool_selection` | `[]` for MCQA/structured; list of tool-name strings for workplace (constant length 27 — every row exposes the full toolset) |
| `ability` | `"mcqa"` / `"structured_outputs"` / `"workplace_assistant"` — the actual per-row domain label used everywhere downstream (dataset filtering, this doc's stats) |
| `reward_model` | `{"style": "rule", "ground_truth": ...}` — the verification target, shape differs completely per domain (see below) |
| `extra_info` | `{split, index, agent_ref, reward_mode, template_metadata, options}` — `agent_ref` (e.g. `"mcqa_simple_agent"`, `"workplace_assistant_simple_agent"`, `"structured_outputs_simple_agent"`) is what `multi_domain_reward_fn` (`reward.py:28-41`) actually dispatches reward computation on |

---

## 2. MCQA

### Dataset shape

- `prompt`: system prompt (`SYSTEM_PROMPT_MCQA`) + one user message — a STEM question with lettered options (sample: a biomedical physiology question on allostatic load/CHD, options A–J).
- `reward_model.ground_truth`: a bare single letter, e.g. `"C"` (stripped at build time, `prepare_data.py:161`).
- `extra_info.options`: JSON-encoded list of `{"letter": "text"}` dicts — used at reward time to validate/disambiguate extracted answers.
- `extra_info.template_metadata`: 
  ```python
  {"template_id": "mcqa_generated_009",
   "template_prompt": "At the conclusion of your response, select exactly one option and state it in the format: Option Selected: X\n\n{problem}\n\nSelect the most appropriate answer.",
   "output_regex": "Option Selected:\\s*([A-Za-z0-9])\\s*",
   "weight": 0.005952381, "prompt_type": "generated", "format_type": "mcqa"}
  ```
  **This is present on every single MCQA row (100% of 9,576)**, with **29 distinct `output_regex` templates** across the dataset (frequency-weighted, e.g. `\boxed{\s*([A-Za-z0-9])\s*}` — 1,970 rows; `Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])\s*` — 1,332 rows; `\*\*([A-Za-z0-9])\*\*` — 497 rows; `Selected Option\s*->\s*([A-Za-z0-9])\s*` — 492 rows; `Option Selected:\s*([A-Za-z0-9])\s*` — 474 rows; `<final_answer>\s*([A-Za-z0-9])\s*</final_answer>` — 216 rows; and 23 more).
- `extra_info.reward_mode`: **always the literal string `"strict_single_letter_boxed"`, hardcoded at dataset-build time** (`prepare_data.py:168`) — this is only a *fallback* label, not reflective of the actual answer format used in the prompt.

### Reward computation (`reward.py::mcqa_reward_fn`, lines 201-282)

Answer extraction, in priority order:

1. **Per-template `output_regex` (dominant path, fires for ~100% of rows since every row has one):** `_parse_answer_with_custom_regex` (`reward.py:60-105`) runs `re.findall(regex, text, re.IGNORECASE)` and takes the **last** match (to skip past any reasoning that appears before the final answer). The captured string passes through `_normalize_extracted_answer` (maps Arabic/Bengali/fullwidth digit-letter glyphs to Latin, `reward.py:120-135`), then is either accepted directly as a single letter, or matched against option text via `_normalize_for_match`.
2. **Fallback (`extra_info["reward_mode"]`), only if step 1 finds nothing** — four named modes, each with its exact regex:

   | Mode | Regex (literal) | What it targets |
   |---|---|---|
   | `strict_single_letter_boxed` | `r"\\boxed\{\s*[^A-Za-z]*([A-Z])[^A-Za-z]*\s*\}"` | `\boxed{...}` containing exactly one uppercase letter, tolerating surrounding punctuation/whitespace |
   | `lenient_boxed` | tries the above first; else `r"\\boxed\{\s*(.*?)\s*\}"` (DOTALL) | grabs the raw `\boxed{...}` content, strips LaTeX `\text{...}` wrappers, matches as a **substring** against option text — only accepted if exactly one option matches (ambiguity → reject) |
   | `lenient_answer_colon` | `r"(?i)answer\s*:\s*(.+)"` | text after `Answer:` (case-insensitive), matched as bare letter or **exact** normalized option text |
   | `lenient_answer_colon_md` | `r"(?i)[*_]{0,2}Answer[*_]{0,2}\s*:[*_\s]{0,2}\s*([A-Z])(?![a-zA-Z0-9])"` | markdown-decorated `**Answer:** A` style, single letter only, negative lookahead prevents matching inside a longer word/number |

3. **Comparison**: `pred == gold` where `gold = ground_truth.strip().upper()` — exact, case-insensitive, single-letter match. **Binary reward: 1.0 or 0.0, no partial credit.**

### What capability this trains

Two things simultaneously: (1) **STEM domain knowledge/reasoning** under multiple-choice constraint — the paper credits this domain's improvement to **MMLU-Pro**; (2) **rigid output-format discipline** — the model must reliably terminate its answer in one of 29 templated formats, itself an instruction-following skill reinforced by the all-or-nothing reward. The regex-cascade design (dataset-specific template first, generic fallbacks second) is what makes this domain tolerant of format variety while still requiring exactness.

---

## 3. Structured Outputs

### Dataset shape

- `prompt`: system prompt (`SYSTEM_PROMPT_STRUCTURED_OUTPUTS`, explicitly instructing *"Output only the JSON object, with no surrounding prose, explanation, or markdown code fences"*) + 1-2 user messages containing a JSON Schema spec to conform to (sample: a nested concert-event schema with `eventDetails`/`attendeeProfile`/`logistics`, enums, nested objects, date/time formats).
- `reward_model.ground_truth`: the raw JSON-schema text itself (`schema_str` from the raw dataset, copied verbatim, `prepare_data.py:207`) — i.e. the "ground truth" here is a *schema to validate against*, not a target output.
- `extra_info`: `reward_mode`/`template_metadata`/`options` are always `None` — unused for this domain.

### Reward computation (`reward.py::structured_reward_fn`, lines 339-383)

1. Empty completion → 0.0.
2. Parse `ground_truth` as JSON → the schema dict. Parse failure → 0.0.
3. **`strictify_schema()`** (`reward.py:329-336`) mutates the schema recursively: for every object node with a `"properties"` key, forces `required = list(properties.keys())` and `additionalProperties = False` — i.e., the dataset's own `required`/`additionalProperties` values are **overwritten** to make every declared property mandatory and forbid any extra key, regardless of what the raw schema specified.
4. Parse the **model's completion** as JSON (no markdown-fence stripping, no salvage — must be pure JSON per the system prompt). Parse failure → 0.0.
5. Validate the parsed completion against the strictified schema via **`openapi_schema_validator.validate()`**. Success → 1.0; any exception → 0.0.

Binary reward, no partial credit — pipeline is, in the code's own words: *"Prompt + schema → one model generation → validate against schema_str → binary reward"* (references NVIDIA-NeMo/Gym's `structured_outputs/app.py::evaluate_structured_output_response` as source of truth).

### What capability this trains

Precise **instruction-following for machine-consumable output formatting**: produce a single, syntactically valid JSON object matching an arbitrary (often deeply nested, enum-constrained) schema, with zero prose/fences and every declared property present, none extra. This is directly the skill needed for **tool-calling/function-calling and API-integration** downstream tasks — reliably emitting the exact shape an API contract expects. The paper credits this domain's improvement to **IF-Bench** (general instruction-following).

---

## 4. Workplace Assistant

### Dataset shape

- Dispatch key: `agent_ref.name == "workplace_assistant_simple_agent"` → `workplace_assistant_data()` (`prepare_data.py:26-127`).
- `prompt`: system message (date/time context + business rule, e.g. *"Meetings must not start before 9am or end after 6pm"*) + one user task (e.g. *"Employee Akira is taking over customer Casey Jackson. Can you reassign this customer in the CRM?"*).
- `tool_selection`: all 27 tool names, on every row (the model always has the full toolset available, must select the right ones itself).
- `reward_model.ground_truth`: a JSON-encoded list of 0-8 `{"name": str, "arguments": <JSON-string>}` dicts — the reference action trace.
- **Ground-truth action-count distribution** (empirical, 4,968 rows): 0 actions → 513 rows, 1 → 3,084, 2 → 885, 3 → 243, 4 → 157, 5 → 49, 6 → 25, 7 → 3, 8 → 9. **~72% of tasks need 0-1 tool calls**; only 9 rows (0.2%) need as many as 8 — the distribution is heavily skewed toward trivial tasks, with a long, thin tail of complex multi-step ones. (This distribution is directly relevant to the training-dynamics analysis in `AGENTS.md` §3.2 — a policy converging toward short completions is arguably fitting the dominant mode of this distribution.)

### The 27 tools (`tools.yaml`, all `class_name: reward.WorkplaceTool`), by toolkit

- **Analytics (6, read-mostly)**: `get_visitor_information_by_id`, `create_plot` (the only mutator — appends to a `_plots_data` frame), `total_visits_count`, `engaged_users_count`, `traffic_source_count`, `get_average_session_duration`.
- **Calendar (5, full CRUD)**: `get_event_information_by_id`, `search_events` (paginated), `create_event`, `delete_event`, `update_event` (single-field).
- **Company Directory (1, read-only)**: `find_email_address` (case-insensitive name→email lookup) — the "resolve a name before acting" tool most multi-step tasks need first.
- **CRM / Customer Relationship Manager (4, full CRUD)**: `search_customers` (multi-filter), `update_customer`, `add_customer`, `delete_customer`.
- **Email (6)**: `get_email_information_by_id`, `search_emails` (full-text, all query words must match), `send_email`, `delete_email`, `forward_email`, `reply_email`.
- **Project Management (5, full CRUD)**: `get_task_information_by_id`, `search_tasks`, `create_task`, `delete_task`, `update_task`.

Underlying data: each toolkit class (`tools/*.py`) loads its state from `csv_data/processed/*.csv` (`dtype=str`) into an in-memory pandas DataFrame on `reset_state()`. Columns:

| CSV | Columns |
|---|---|
| `analytics_data.csv` | `date_of_visit, visitor_id, page_views, session_duration_seconds, traffic_source, user_engaged` |
| `calendar_events.csv` | `event_id, event_name, participant_email, event_start, duration` |
| `customer_relationship_manager_data.csv` | `customer_id, assigned_to_email, customer_name, customer_email, customer_phone, last_contact_date, product_interest, status, follow_up_by, notes` |
| `emails.csv` | `email_id, inbox/outbox, sender/recipient, subject, sent_datetime, body` |
| `project_tasks.csv` | `task_id, task_name, assigned_to_email, list_name, due_date, board` |

(`csv_data/raw/email_addresses.csv` — headerless, one address per line — backs the company-directory lookup; `csv_data/raw/events.csv` looks like a generation seed list, not read by any tool at runtime.) All data is fictitious ("Atlas" company, `@atlas.com` domain).

Operation pattern is CRUD-like but consistently: one ID-based read, one paginated multi-filter search, and (except analytics/directory) create/update/delete mutators — all single-call, stateless lookups against the in-memory frame. There's no cross-toolkit joining inside one call (e.g. resolving a name to an email requires a separate `company_directory_find_email_address` call before `calendar_create_event`) — this is exactly where the "multi-step planning" requirement comes from.

### Reward computation (`reward.py::WorkplaceTool.execute()` + `workplace_reward_fn`, lines 388-464, 593-648)

1. **Every tool call the model makes during rollout — read or mutating, successful or not — is recorded** into `extra_info["predicted_actions"]` unconditionally, *before* the call is even attempted (`reward.py:429-440`). So `predicted_actions` is the complete, ordered trace of every tool invocation, not just the mutating ones.
2. At verification time, `predicted_actions` and the ground-truth action list are each replayed **sequentially, in exact recorded order**, into two **freshly-constructed** tool environments (`execute_actions_and_reset_state`, `reward.py:467-484`) — exceptions during replay are swallowed (`continue`), and there is **no reordering/permutation tolerance**: if two calls' side effects interact, order matters and both traces must match exactly for the outcome to be equal.
3. All string columns are lowercased before comparison **except** `status`, `list_name`, `board` (case-sensitive) — this is the *only* tolerance built into the comparison.
4. Final reward = **strict boolean AND of five separate pandas `.equals()` comparisons**, one per toolkit DataFrame (calendar, email, analytics-plots, project-tasks, CRM). **A mismatch in any single toolkit zeroes the entire reward** — this is the most punishing reward design of the three domains, since a single incomplete or slightly-wrong sub-action fails everything.

### What capability this trains

Agentic tool-use / function-calling, specifically: (1) **tool selection** among 27 candidate functions across 5 business toolkits plus a directory lookup; (2) **multi-step planning** when a task needs an intermediate lookup first (name→email before scheduling, etc.); (3) **exact-match state-mutation correctness** — not just calling the right tool, but supplying exactly the right arguments so the resulting state is byte-identical (case-insensitively, with 3 exceptions) to a human-authored reference; (4) **robustness across a highly skewed action-count distribution** (recognizing 0-1-action tasks as such, while occasionally handling repeated bulk actions like updating 8 CRM rows); and (5) operating within a bounded multi-turn budget (`max_assistant_turns=6`). The paper credits this domain's improvement to **τ²-Bench** (a tool-use/agent benchmark), citing "the Workplace Assistant setup (Blakeman et al., 2025)" as the external source of the task design — it gives no further architectural detail beyond that one-line attribution.

---

## 5. Why these three domains are trained together (paper's own rationale, §4.3)

> "We group these domains into a single multi-domain RL stage for two main reasons. First, we do not observe performance degradation across evaluation benchmarks when training on the blended domains. Instead, the model exhibits consistent improvements on benchmarks including MMLU-Pro, τ²-Bench, and IF-Bench. Second, the response lengths and verification times of these datasets are similar, which minimizes training inefficiencies caused by waiting for longer generations or slower environment verification."

Capability-to-benchmark mapping, per the paper's own citations:

| Domain | Capability | Benchmark cited |
|---|---|---|
| MCQA | STEM reasoning + output-format discipline | MMLU-Pro |
| Workplace Assistant | Agentic tool-use / multi-step planning | τ²-Bench |
| Structured Outputs | Instruction-following formatting fidelity | IF-Bench |

Hyperparameters shared across all three in the blended stage (Table 8, Appendix B; also see `AGENTS.md`): batch 128, `rollout.n=16`, LR 3e-6 (AdamW), temperature/top-p 1.0, entropy/KL coefficients zero, ~70 steps, max response length 49K (paper) vs. 4000-8000 in this reproduction's current configs.
