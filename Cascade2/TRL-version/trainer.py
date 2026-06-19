from trl import GRPOConfig, GRPOTrainer
from trl.rewards import get_soft_overlong_punishment

from dataset import format_dataset_RL_Cascade2
from reward import make_if_reward_fn, IFReward
from transformers import AutoTokenizer, ProcessorMixin, PreTrainedTokenizerBase



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
        vllm_gpu_memory_utilization=0.25,             
        vllm_max_model_length=5_000+max_completion_length, # should be equal to max prompt length(4094) + max_completion_lentgh(4000) 
        vllm_importance_sampling_correction=False, # defaults to true
    
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 1024, # 128*16
        # steps_per_generation=16,           
        num_generations=16,              
        max_completion_length=max_completion_length,      
        max_steps=180,       
        bf16=True,

        beta=.0, # KL loss set to 0
        temperature=1., # default
        top_p=1., # default
        num_iterations=1, # on policy
        loss_type="dapo", # default
        epsilon_high=0.28,
        epsilon=0.2, # => with importance sampling set to 1 this clipping mechanism never activates
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

    """
    if_reward_fn = IFReward(
        max_completion_length=max_completion_length,
        reward_mode=reward_mode,
        debug_path="if_reward_binary_systemprompt.txt",
        debug_every=20480/2*3,
        print_to_terminal=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[
            if_reward_fn, 
        ],
        train_dataset=train_dataset,
    )

    if_reward_fn.set_eos_token(trainer.model.generation_config.eos_token_id)

    
    Verify the tokenizer alignment issue


    print("Before alignment")
    print("\ttokenizer bos:", tokenizer.bos_token_id)
    print("\ttokenizer eos:", tokenizer.eos_token_id)
    print("\ttokenizer pad:", tokenizer.pad_token_id)

    print("\tmodel bos:", trainer.model.config.bos_token_id)
    print("\tmodel eos:", trainer.model.config.eos_token_id)
    print("\tmodel pad:", trainer.model.config.pad_token_id)

    print("\tgen bos:", trainer.model.generation_config.bos_token_id)
    print("\tgen eos:", trainer.model.generation_config.eos_token_id)
    print("\tgen pad:", trainer.model.generation_config.pad_token_id)
    print("-"*100)
    before_tokenizer_bos_token_id = tokenizer.bos_token_id
    before_tokenizer_eos_token_id = tokenizer.eos_token_id
    before_tokenizer_pad_token_id = tokenizer.pad_token_id

    before_trainer_model_config_bos_token_id = trainer.model.config.bos_token_id
    before_trainer_model_config_eos_token_id = trainer.model.config.eos_token_id
    before_trainer_model_config_pad_token_id = trainer.model.config.pad_token_id

    before_trainer_model_generation_config_bos_token_id = trainer.model.generation_config.bos_token_id
    before_trainer_model_generation_config_eos_token_id = trainer.model.generation_config.eos_token_id
    before_trainer_model_generation_config_pad_token_id = trainer.model.generation_config.pad_token_id

    if isinstance(tokenizer, ProcessorMixin):
        tokenizer: PreTrainedTokenizerBase = tokenizer.tokenizer
    else:
        print("1.")
        tokenizer = tokenizer
    model_has_generation_config = hasattr(trainer.model, "generation_config") and trainer.model.generation_config is not None # this is true
    updated_tokens = {}

    # 1 - Align EOS token. EOS is more complex than the others, as `generation_config` may hold more than one EOS
    # token.
    tokenizer_has_new_eos = tokenizer.eos_token_id != getattr(trainer.model.config, "eos_token_id", None)
    if model_has_generation_config:
        # `generation_config.eos_token_id` is None: direct comparison
        if trainer.model.generation_config.eos_token_id is None:
            tokenizer_has_new_eos |= tokenizer.eos_token_id != trainer.model.generation_config.eos_token_id
        else:
            print("2.")
            # `generation_config.eos_token_id` is an `int`: convert it to list (and continue below)
            if isinstance(trainer.model.generation_config.eos_token_id, int):
                trainer.model.generation_config.eos_token_id = [trainer.model.generation_config.eos_token_id]
            # `generation_config.eos_token_id` is a `list`: check if the tokenizer's EOS token is in the list
            tokenizer_has_new_eos |= tokenizer.eos_token_id not in trainer.model.generation_config.eos_token_id
            print("tokenizer_has_new_eos? :", tokenizer_has_new_eos) # False

    if tokenizer_has_new_eos:
        updated_tokens["eos_token_id"] = tokenizer.eos_token_id
        trainer.model.config.eos_token_id = tokenizer.eos_token_id
        # The generation config may hold more than one EOS token. We preserve the original EOS tokens: any of the
        # EOS tokens defined here will halt generation.
        if model_has_generation_config:
            all_eos_tokens = [tokenizer.eos_token_id]
            if trainer.model.generation_config.eos_token_id is not None:
                all_eos_tokens += list(trainer.model.generation_config.eos_token_id)
            trainer.model.generation_config.eos_token_id = [token for token in all_eos_tokens if token is not None]
            print("Updated eos of generation : ", trainer.model.generation_config.eos_token_id) # this is not printed




    # 2 - Align BOS
    tokenizer_has_new_bos = tokenizer.bos_token_id != getattr(trainer.model.config, "bos_token_id", None)
    if model_has_generation_config:
        tokenizer_has_new_bos |= tokenizer.bos_token_id != trainer.model.generation_config.bos_token_id
        print("3.")
        print("tokenizer_has_new_bos? :", tokenizer_has_new_bos)

    if tokenizer_has_new_bos:
        updated_tokens["bos_token_id"] = tokenizer.bos_token_id
        trainer.model.config.bos_token_id = tokenizer.bos_token_id
        if model_has_generation_config:
            trainer.model.generation_config.bos_token_id = tokenizer.bos_token_id
        print("bos updates")

    # 3 - Align PAD
    tokenizer_has_new_pad = tokenizer.pad_token_id != getattr(trainer.model.config, "pad_token_id", None)
    if model_has_generation_config:
        tokenizer_has_new_pad |= tokenizer.pad_token_id != trainer.model.generation_config.pad_token_id

    print("4.")
    print("tokenizer_has_new_pad? :", tokenizer_has_new_pad)
    if tokenizer_has_new_pad:
        updated_tokens["pad_token_id"] = tokenizer.pad_token_id
        trainer.model.config.pad_token_id = tokenizer.pad_token_id
        if model_has_generation_config:
            trainer.model.generation_config.pad_token_id = tokenizer.pad_token_id
        print("pad updated")

    # 4 - Warn users about the changes
    # if len(updated_tokens) > 0:
    #     logger.warning(
    #         "The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. "
    #         "The model config and generation config were aligned accordingly, being updated with the tokenizer's "
    #         f"values. Updated tokens: {updated_tokens}."
    #     )
    def fmt(x):
        return str(x)

    def row(name, after, before):
        print(f"{name:<18} | {fmt(after):<20} | {fmt(before):<20}")

    print("-" * 70)
    print(f"{'Token':<18} | {'Before alignment':<20} | {'After alignment':<20}")
    print("-" * 70)

    row("tokenizer bos", before_tokenizer_bos_token_id, tokenizer.bos_token_id)
    row("tokenizer eos", before_tokenizer_eos_token_id, tokenizer.eos_token_id)
    row("tokenizer pad", before_tokenizer_pad_token_id, tokenizer.pad_token_id, )

    print("-" * 70)

    row("model bos", before_trainer_model_config_bos_token_id, trainer.model.config.bos_token_id)
    row("model eos", before_trainer_model_config_eos_token_id, trainer.model.config.eos_token_id)
    row("model pad", before_trainer_model_config_pad_token_id, trainer.model.config.pad_token_id)

    print("-" * 70)

    row("gen bos", before_trainer_model_generation_config_bos_token_id, trainer.model.generation_config.bos_token_id)
    row("gen eos", before_trainer_model_generation_config_eos_token_id, trainer.model.generation_config.eos_token_id)
    row("gen pad", before_trainer_model_generation_config_pad_token_id, trainer.model.generation_config.pad_token_id)

    print("-" * 70)

    assert False

    print("-" * 70)

    print("tokenizer bos", trainer.processing_class.bos_token_id)
    print("tokenizer eos", trainer.processing_class.eos_token_id)
    print("tokenizer pad", trainer.processing_class.pad_token_id)

    print("-" * 70)

    print("model bos", trainer.model.config.bos_token_id)
    print("model eos", trainer.model.config.eos_token_id)
    print("model pad", trainer.model.config.pad_token_id)

    print("-" * 70)

    print("gen bos", trainer.model.generation_config.bos_token_id)
    print("gen eos", trainer.model.generation_config.eos_token_id)
    print("gen pad", trainer.model.generation_config.pad_token_id)

    print("-" * 70)
    """

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