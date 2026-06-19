from datasets import load_dataset
from transformers import AutoTokenizer

path = "./grpo_Qwen25_15B_gsm8k_iterations2_checkpoints/checkpoint-1000"

tok1 = AutoTokenizer.from_pretrained(path)
tok2 = AutoTokenizer.from_pretrained(path, fix_mistral_regex=True)

dataset = load_dataset("openai/gsm8k", "main")["test"]

diffs = 0

for i, ex in enumerate(dataset):
    text = f"Question: {ex['question']}\nAnswer:"

    ids1 = tok1.encode(text)
    ids2 = tok2.encode(text)

    if ids1 != ids2:
        print("Example", i)
        print(text)
        print(len(ids1), len(ids2))
        diffs += 1

print(f"Different tokenizations: {diffs}/{len(dataset)}")