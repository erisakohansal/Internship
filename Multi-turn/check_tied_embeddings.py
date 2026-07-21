"""Sanity-checks a merged VeRL/HF checkpoint for a broken tied-embedding pair.

Qwen2.5-1.5B(-Instruct) ties `model.embed_tokens.weight` and `lm_head.weight`.
FSDP shards parameters by identity, and if a checkpoint-merge step doesn't
special-case tied weights, the two can silently end up saved as two
different (and possibly diverged) tensors, or one of them missing entirely
and reconstructed wrong at load time. This script flags that before you
spend an evaluation run on a checkpoint that's quietly broken.

Usage: python check_tied_embeddings.py --checkpoint_dir /path/to/merged/global_step_X
"""

import argparse
import glob
import json
import os

from safetensors import safe_open


def find_weight(checkpoint_dir, name):
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        if name not in weight_map:
            return None
        shard_path = os.path.join(checkpoint_dir, weight_map[name])
        with safe_open(shard_path, framework="pt") as f:
            return f.get_tensor(name)

    # Single-file checkpoint (no index.json).
    for path in glob.glob(os.path.join(checkpoint_dir, "*.safetensors")):
        with safe_open(path, framework="pt") as f:
            if name in f.keys():
                return f.get_tensor(name)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    args = parser.parse_args()

    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    tie_word_embeddings = config.get("tie_word_embeddings")
    print(f"config.json tie_word_embeddings: {tie_word_embeddings}")

    embed_tokens = find_weight(args.checkpoint_dir, "model.embed_tokens.weight")
    lm_head = find_weight(args.checkpoint_dir, "lm_head.weight")

    if embed_tokens is None:
        print("FAIL: model.embed_tokens.weight not found in checkpoint")
        return
    print(f"model.embed_tokens.weight: shape={tuple(embed_tokens.shape)}")

    if lm_head is None:
        if tie_word_embeddings:
            print("OK: lm_head.weight absent, tie_word_embeddings=true "
                  "-- HF/vLLM will reconstruct lm_head from embed_tokens at load time.")
        else:
            print("FAIL: lm_head.weight absent but tie_word_embeddings is not true "
                  "-- the loaded model's output projection will be randomly initialized.")
        return

    print(f"lm_head.weight: shape={tuple(lm_head.shape)}")
    if embed_tokens.shape != lm_head.shape:
        print("FAIL: embed_tokens and lm_head have different shapes -- not tied, not comparable.")
        return

    max_abs_diff = (embed_tokens.float() - lm_head.float()).abs().max().item()
    identical = max_abs_diff == 0.0
    print(f"max |embed_tokens - lm_head| = {max_abs_diff}")
    if identical:
        print("OK: embed_tokens and lm_head are identical -- tie intact.")
    else:
        print("FAIL: embed_tokens and lm_head differ -- the tie broke during FSDP training "
              "and/or the merge saved two diverged copies. This checkpoint's output "
              "projection does not match what was actually optimized.")


if __name__ == "__main__":
    main()
