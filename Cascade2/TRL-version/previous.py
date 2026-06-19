from trl import GRPOConfig, GRPOTrainer
from trl.rewards import get_soft_overlong_punishment

from dataset import format_dataset_RL_Cascade2
from reward import make_if_reward_fn
from transformers import AutoTokenizer, ProcessorMixin, PreTrainedTokenizerBase

def make_if_reward_fn(
    tokenizer,
    max_completion_length,
    reward_mode,
    counter=0,
    debug_path="if_reward_binary_systemprompt.txt",
    debug_every=20480*3,
    print_to_terminal=False,
):
    # closure state
    eos_id = tokenizer.eos_token_id

    def if_reward_fn(completion_ids, completions, instruction_id_list, kwargs, **extra):
        nonlocal counter, print_to_terminal

        def log(*args, **print_kwargs):
            """Print to terminal and append the same message to an external file."""
            if print_to_terminal:
                print(*args, **print_kwargs)

            with open(debug_path, "a", encoding="utf-8") as f:
                print(*args, **print_kwargs, file=f)


        # log("\t\tNumber of completions received:", len(completions))

        rewards = []

        for ids, completion, instr_list, kwarg_list in zip(
            completion_ids, completions, instruction_id_list, kwargs
        ):
            should_debug = counter % debug_every == 0

            if should_debug:
                log("#" * 100)
                log(f"Counter: {counter}")
                log(f"Instruction IDs: {instr_list}")
                log(f"Kwargs: {kwarg_list}")
                log(f"\nCompletion:\n{completion[0]['content']}")
                log("-" * 100)

            # A completion is "overlong" only if it hit the limit AND never emitted EOS
            # i.e. it was forcibly truncated, not a natural finish at exactly max length
            is_overlong = (
                len(ids) >= max_completion_length
                and eos_id not in ids
            )

            if is_overlong:
                rewards.append(0.0)

                if should_debug:
                    log(f"[overlong] length={len(ids)}, no EOS -> reward=0.0")

            else:
                is_following_list = []

                for instruction_id, kw in zip(instr_list, kwarg_list):
                    try:
                        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
                        instruction = instruction_cls(instruction_id)

                        if kw is None:
                            kw = {}

                        filtered_kwargs = {
                            k: v for k, v in kw.items()
                            if v is not None
                        }

                        # Important: build_description sets internal fields used by check_following
                        instruction.build_description(**filtered_kwargs)

                        followed = instruction.check_following(completion[0]["content"])

                        # Protect against buggy check_following functions returning None
                        followed = bool(followed)

                        is_following_list.append(followed)

                        if should_debug:
                            log("Instruction:", instruction_id)
                            log("kwargs:", filtered_kwargs)
                            log("Followed:", followed)
                            log("Reward contribution:", int(followed))

                    except Exception as e:
                        log(f"Error processing instruction {instruction_id} with kwargs {kw}: {e}")
                        log("The corresponding completion was:")
                        log(completion[0]["content"]) # here do a print
                        is_following_list.append(False)

                if reward_mode == "binary":
                    reward = float(all(is_following_list))
                else:
                    reward = float(
                        sum(is_following_list) / len(is_following_list)
                        if is_following_list else 0.0
                    )

                rewards.append(reward)

                if should_debug:
                    log("is_following_list:", is_following_list)
                    log("Final reward:", reward)

            if should_debug:
                log("-" * 100)

            counter += 1

        return rewards

    return if_reward_fn


def check_prompt_lengths(dataset, tokenizer):
    lengths = []

    for ex in dataset:
        text = tokenizer.apply_chat_template(
            ex["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))

    print("max prompt tokens:", max(lengths))
    print("mean prompt tokens:", sum(lengths) / len(lengths))



def IF_RL(model, max_completion_length, reward_mode, name): # check the params in the article
    config = "IF-RL"

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        padding_side="left",
        truncation_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = format_dataset_RL_Cascade2(config) 
    check_prompt_lengths(train_dataset, tokenizer)
    print(train_dataset[0])

    training_args = GRPOConfig( 
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.05,             
        vllm_max_model_length=5_000+max_completion_length, # should be equal to max prompt length(4094) + max_completion_lentgh(4000) 
        vllm_importance_sampling_correction=False, # defaults to true
    
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2, # 128*16
        # steps_per_generation=16,           
        num_generations=2,              
        max_completion_length=max_completion_length,      
        max_steps=180,       
        bf16=True,

        beta=.0, # KL loss set to 0
        temperature=1., # default
        top_p=1., # default
        num_iterations=1,
        loss_type="dapo", 
        epsilon_high=0.28,
        epsilon=0.2, 
        mask_truncated_completions=False,

        # TrainingArguments from transformers
        learning_rate=3e-6,
        optim="adamw_torch_fused",  # AdamW as per the paper stated
        adam_beta1=0.9,             
        adam_beta2=0.95,            
        
        report_to="wandb", 
        run_name=name,
        
        save_strategy="steps",
        save_steps=10,
        output_dir=name+"_checkpoints",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[
            make_if_reward_fn(
                tokenizer=tokenizer, 
                max_completion_length=max_completion_length, 
                reward_mode=reward_mode
            ), 
        ],
        train_dataset=train_dataset,
    )

    print("-" * 70)

    print("tokenizer bos", trainer.processing_class.bos_token_id)
    print("tokenizer eos", trainer.processing_class.eos_token_id)
    print("tokenizer pad", trainer.processing_class.pad_token_id, )

    print("-" * 70)

    print("model bos", trainer.model.config.bos_token_id)
    print("model eos", trainer.model.config.eos_token_id)
    print("model pad", trainer.model.config.pad_token_id)

    print("-" * 70)

    print("gen bos", trainer.model.generation_config.bos_token_id)
    print("gen eos", trainer.model.generation_config.eos_token_id)
    print("gen pad", trainer.model.generation_config.pad_token_id)

    print("-" * 70)

    trainer.train() 
    trainer.save_model(config)


def multi_domain_RL(model):
    config = "multi_domain_RL"

    train_dataset = format_dataset_RL_Cascade2(config) 
    print(train_dataset[0])

    training_args = GRPOConfig( 
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,             
        vllm_max_model_length=9_000, # should be equal to max prompt length(4565) + max_completion_lentgh(4000) 
        vllm_importance_sampling_correction=False, # defaults to true
    
        per_device_train_batch_size=1,   
        gradient_accumulation_steps=16, 
        # steps_per_generation=16,           
        num_generations=16,              
        max_completion_length=max_completion_length,      
        max_steps=560,       
        bf16=True,

        beta=.0, # KL loss set to 0
        temperature=1., # default
        top_p=1., # default
        num_iterations=1, # on policy
        loss_type="dapo", # default, dynamic sampling + KL=0 + ???
        epsilon=0.2, # => with importance sampling set to 1 this clipping mechanism never activates

        # TrainingArguments from transformers
        learning_rate=3e-6,
        optim="adamw_torch_fused",  # AdamW as per the paper stated
        adam_beta1=0.9,             
        adam_beta2=0.95,            
        
        report_to="wandb", 
        run_name=config,
        
        save_strategy="steps",
        save_steps=5,
        output_dir=config+"_checkpoints",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[
            if_reward_fn, 
            get_soft_overlong_punishment(
                max_completion_len=MAX_COMPLETION_LENGTH,
                soft_punish_cache=0,
            )
        ],
        train_dataset=train_dataset,
    )


    trainer.train() 
    trainer.save_model(config)