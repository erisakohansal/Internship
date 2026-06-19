from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
    truncation_side="left",
)

# IFEval dataset
ds = load_dataset("google/IFEval", split="train")

lengths = []

for ex in ds:
    # Usually IFEval has a "prompt" field
    messages = [
        {"role": "user", "content": ex["prompt"]}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    lengths.append(len(ids))

lengths = np.array(lengths)

print("num prompts:", len(lengths))
print("max prompt tokens:", lengths.max())
print("mean prompt tokens:", lengths.mean())
print("p95 prompt tokens:", np.percentile(lengths, 95))
print("p99 prompt tokens:", np.percentile(lengths, 99))