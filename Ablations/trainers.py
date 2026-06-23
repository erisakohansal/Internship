from trl import RLOOConfig, RLOOTrainer, GRPOConfig, GRPOTrainer
from dataset import format_dataset_dolci_math, format_dataset_gsm8k
from reward import reward_tmp, format_reward


def rloo_(model, dataset_name, model_name):

    train_dataset = None
    if dataset_name=="gsm8k":
        train_dataset = format_dataset_gsm8k(split="train")
    else:
        train_dataset = format_dataset_dolci_math() 
    print(train_dataset[0])


    # for resuming a checkpoint
    # import json
    # with open("rloo_Qwen25_15B_gsm8k_K8_checkpoints/checkpoint-29000/trainer_state.json") as f:
    #     state = json.load(f)

    # max_steps = state["max_steps"]  # reads 22419 from the checkpoint


    training_args = RLOOConfig(      
        use_vllm=True,
        vllm_mode="colocate",             # vllm runs inside the same training process, vllm run separately (on different gpus): CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct
        vllm_gpu_memory_utilization=0.25, # 0.3 by default
        vllm_max_model_length=8_000,
        # vllm_enable_sleep_mode=True,    # offloads vLLM's weights + KV cache to CPU during the optimizer step (the peak-memory phase), then reloads them for generation
        
        per_device_train_batch_size=1,    # defaults to 8
        gradient_accumulation_steps=4,   # defaults to 1
        steps_per_generation=4,           # default set to gradient_accumulation_steps
        num_generations=4,                # default to 2 generations per prompt, generation batch size must be divisible by num_generations
        max_completion_length=4_000,      # 2000,
        num_train_epochs=3,               # or max_steps?
        # max_steps=max_steps,
        
        # model_init_kwargs={
        #     "attn_implementation": "sdpa",
        #     "dtype": torch.float32,
        #     "torch_dtype": "bfloat16",
        #     "device_map": "cuda:0",
        #     "low_cpu_mem_usage": False,
        #     "attn_implementation": "flash_attention_4",
        # },
        # chat_template_kwargs={"enable_thinking": False},
        # fp16=False,                     # not an argument of RLOOConfig, it's an argument of TrainingArguments from transformers library (its parent)
        bf16=True,
        # beta=0.5,
        num_iterations=4, 
        
        report_to="wandb", 
        run_name="RLOO "+model_name,
        
        # save_strategy="no",
        # save_total_limit=3,             # keeps only last 3 checkpoints
        save_strategy="steps",
        save_steps=1000,
        output_dir="./rloo_"+model_name+"_checkpoints",
    )

    trainer = RLOOTrainer(
        model=model, 
        args=training_args,
        reward_funcs=[format_reward, reward_tmp],
        train_dataset=train_dataset,
    )

    # trainer.train(resume_from_checkpoint=True)
    trainer.train() 
   
    trainer.save_model("rloo_"+model_name)

def grpo_(model, dataset_name, model_name):
    
    train_dataset = None
    if dataset_name=="gsm8k":
        train_dataset = format_dataset_gsm8k(split="train")
    else:
        train_dataset = format_dataset_dolci_math() 

    training_args = GRPOConfig(      
        use_vllm=True,
        vllm_mode="colocate",             
        vllm_max_model_length=8_000,
   
        per_device_train_batch_size=1,    
        gradient_accumulation_steps=4, 
        steps_per_generation=4,   # mutually exclusive with generation batch size        
        num_generations=4,               
        max_completion_length=4_000,      
        num_train_epochs=3,               

        bf16=True,
        # important GRPO-specific params
        # beta=0.0,              # TRL default for GRPO
        # epsilon=0.2,           # PPO/GRPO clipping default
        # num_iterations=1,      # keep simple for first comparison
        
        report_to="wandb", 
        run_name="GRPO "+model_name,
        
        save_strategy="steps",
        save_steps=1000,
        output_dir="./grpo_"+model_name+"_checkpoints",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[format_reward, reward_tmp],
        train_dataset=train_dataset,
    )

    trainer.train() 
    trainer.save_model("grpo_"+model_name)
