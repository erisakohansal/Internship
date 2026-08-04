from datasets import load_dataset
import os
from transformers import AutoTokenizer
import numpy as np


def format_rlhf(data, idx):
    assert len(data["prompt"]) == 1
    return {
        "data_source": data["data_source"],
        "prompt": data["prompt"],
        "ability": data["category"],
        "extra_info": {
            "split": "train",
            "index": idx,
        }
    }


def compute_prompt_length(row):

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    template_kwargs = {
        "conversation": row["prompt"],
        "tokenize": True,
        "add_generation_prompt": True,
    }

    input_ids = tokenizer.apply_chat_template(**template_kwargs)["input_ids"]

    return {
        "prompt_tokens": len(input_ids),
    }


def format_cascade2_rlhf():
    data = load_dataset(
        "nvidia/Nemotron-Cascade-RL-RLHF",
        split="train",
    )

    dataset = data.map(
        format_rlhf,
        remove_columns=data.column_names,
    )

    data = dataset.map(
        compute_prompt_length,
        load_from_cache_file=False,
        desc="Computing rendered prompt lengths",
    )

    lengths = np.asarray(data["prompt_tokens"])
    
    print("\nPrompt-length statistics:")
    print("Minimum:", lengths.min())
    print("Mean:", lengths.mean())
    print("Median:", np.median(lengths))
    print("p90:", np.percentile(lengths, 90))
    print("p95:", np.percentile(lengths, 95))
    print("p99:", np.percentile(lengths, 99))
    print("Maximum:", lengths.max())


    splits = dataset.train_test_split(
        test_size=0.05,
        seed=42, 
        shuffle=True,
    )

    train_set = splits["train"]
    test_set = splits["test"]
    local_dir = os.getcwd()

    print("\tSize of the train split : ", len(train_set))
    print("\tSize of the test split : ", len(test_set))

    train_set.to_parquet(os.path.join(local_dir, "rlhf-train.parquet"))
    test_set.to_parquet(os.path.join(local_dir, "rlhf-test.parquet"))

if __name__=="__main__":
    format_cascade2_rlhf()
