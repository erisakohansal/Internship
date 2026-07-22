"""Builds VeRL-schema train/val parquet files for the math+calculator-tool pipeline.

Train: hendrycks/competition_math (7,500 problems) -- falls back to the
       EleutherAI/hendrycks_math mirror if the primary repo fails to load
       (it ships an old-style loading script that some `datasets` versions
       refuse to run without trust_remote_code=True).
Val:   HuggingFaceH4/MATH-500 (500 problems), held out for evaluation only.

Run this on a machine with real HF Hub access (this repo's own dev sandbox
has no outbound access to huggingface.co) -- and sanity-check the printed
column names against what's assumed below (PROBLEM_KEY/SOLUTION_KEY/
ANSWER_KEY), since dataset column names can drift between mirrors/versions.
"""

import argparse
import os
import re

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "You are a careful math problem solver. You may call the `calculator` "
    "tool for arithmetic you are not fully confident in. Reason step by "
    "step, then give your final answer as \\boxed{answer}."
)

PROBLEM_KEY = "problem"
SOLUTION_KEY = "solution"
ANSWER_KEY = "answer"  # present on MATH-500; absent on the MATH train set


def extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in text (balanced braces)."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    chars = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(c)
        i += 1
    if depth != 0:
        return None
    return "".join(chars)


def to_ground_truth(example) -> str | None:
    if ANSWER_KEY in example and example[ANSWER_KEY]:
        answer = str(example[ANSWER_KEY])
    else:
        answer = extract_boxed(example.get(SOLUTION_KEY, "") or "")
    if answer is None:
        return None
    answer = answer.strip()
    if not answer:
        return None
    return f"${answer}$"


def format_row(example, idx, data_source):
    ground_truth = to_ground_truth(example)
    if ground_truth is None:
        return None
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example[PROBLEM_KEY]},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": "train" if data_source.endswith("train") else "val",
            "index": idx,
            "need_tools_kwargs": False,
        },
    }


def build_split(dataset, data_source):
    rows = []
    for idx, example in enumerate(dataset):
        row = format_row(example, idx, data_source)
        if row is not None:
            rows.append(row)
    dropped = len(dataset) - len(rows)
    if dropped:
        print(f"[{data_source}] dropped {dropped}/{len(dataset)} rows with no extractable answer")
    return Dataset.from_list(rows)


def load_math_train():
    try:
        return load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Falling back to EleutherAI/hendrycks_math mirror ({e})")
        return load_dataset("EleutherAI/hendrycks_math", split="train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    train_raw = load_math_train()
    print("Train columns:", train_raw.column_names)
    print("Train example:", train_raw[0])
    train = build_split(train_raw, "math_calculator/train")
    train.to_parquet(os.path.join(args.output_dir, "math_calculator_train.parquet"))
    print(f"Wrote {len(train)} train rows")

    val_raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print("Val columns:", val_raw.column_names)
    print("Val example:", val_raw[0])
    val = build_split(val_raw, "math_calculator/val")
    val.to_parquet(os.path.join(args.output_dir, "math_calculator_val.parquet"))
    print(f"Wrote {len(val)} val rows")


if __name__ == "__main__":
    main()
