import weave
import os

"""
uv run doesn't see CUDA_VISIBLE_DEVICES have to put os.environ["CUDA_VISIBLE_DEVICES"] = "1"  before trl transformers and torch
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Must be before torch, transformers, trl

from trainer import IF_RL, multi_domain_RL
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)


if __name__=="__main__":
    
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("device_count =", torch.cuda.device_count())
    print("current_device =", torch.cuda.current_device())
    print("device_name =", torch.cuda.get_device_name(0))


    instruct_model = "Qwen/Qwen2.5-1.5B-Instruct" # "LiquidAI/LFM2.5-1.2B-Instruct"
    IF_RL(model=instruct_model, max_completion_length=4_000, reward_mode="binary", name="IF-RL-Binary-systemprompt")
    # multi_domain_RL(model="./IF-RL")

    # checkpoint_path = "IF-RL-Binary_checkpoints/checkpoint-180"  # e.g. ./checkpoints/checkpoint-180

    # tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    # model = AutoModelForCausalLM.from_pretrained(
    #     checkpoint_path,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    # if isinstance(tokenizer, ProcessorMixin):
    #     print("Tokenizer is a ProcessorMixin, using processing_class")
    # elif isinstance(tokenizer, PreTrainedTokenizerBase):
    #     print("Tokenizer is a PreTrainedTokenizerBase, using tokenizer_class")

    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Hello! Can you explain what reinforcement learning is in simple terms?"},
    # ]

    # prompt = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    # )

    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=4000,
    #         temperature=0.7,
    #         top_p=0.9,
    #         do_sample=True,
    #         pad_token_id=tokenizer.eos_token_id,
    #     )

    # response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    # print(response)
        


