import weave
import os

"""
uv run doesn't see CUDA_VISIBLE_DEVICES have to put os.environ["CUDA_VISIBLE_DEVICES"] = "1"  before trl transformers and torch
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Must be before torch, transformers, trl

from trainers import rloo_, grpo_
import torch

# why oom errors? DONE
# save checkpoints every 500 steps? DONE
# evaluate on benchmarks before (DONE) and after rl DONE
# ablations on different weights for rewards => 
# https://git.corp.linguacustodia.com/gcaillaut/rewards-lib/-/blob/main/rewards/code/openenv.py?ref_type=heads

if __name__=="__main__":
    
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("device_count =", torch.cuda.device_count())
    print("current_device =", torch.cuda.current_device())
    print("device_name =", torch.cuda.get_device_name(0))

    # To not save the wanb log
    # os.environ["WANDB_MODE"] = "online"
    # os.environ["WANDB_DIR"] = "/tmp/wandb"
    # os.environ["WANDB_CACHE_DIR"] = "/tmp/wandb_cache"
    # os.environ["WANDB_CONFIG_DIR"] = "/tmp/wandb_config"

    # To load from previous checkpoint
    # os.environ["WANDB_RUN_ID"]= "ba5z30hz"
    # os.environ["WANDB_RESUME"]= "must"

    instruct_model = "Qwen/Qwen2.5-1.5B-Instruct" # "LiquidAI/LFM2.5-1.2B-Instruct"
    rloo_(model=instruct_model, dataset_name="gsm8k", model_name="Qwen25_15B_gsm8k_iterations4")
