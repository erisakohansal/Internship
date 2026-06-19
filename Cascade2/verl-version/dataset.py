import re
import datasets
from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset, Dataset

import os


SYSTEM_PROMPT_IF = """
You are a helpful and harmless assistant.
You are not allowed to use any tools.
""" 


def format_data_if(data, idx):
    # review the data structure, should i use .pop? 
    # https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
    res = {
            'data_source': 'nvidia/Nemotron-Cascade-2-RL-data',
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT_IF},
                data['responses_create_params']['input'][0]
            ],
            'ability': 'instruction_following',
            'reward_model': {
                'style': 'rule',
                'ground_truth': None
            },
            'extra_info': {
                'split': 'train',
                'index': idx,
                'reward_mode': 'binary',
                'instruction_id_list': data['instruction_id_list'],
                'kwargs': data['kwargs'],
                'print_to_terminal': False,
                'debug_path': 'if_reward_binary_verl.txt',
                'max_completion_length': 4000,
            }
        }
    return res


def format_dataset_RL_Cascade2(config="IF-RL") -> Dataset:
    data = load_dataset(
        "nvidia/Nemotron-Cascade-2-RL-data",  #"nvidia/Nemotron-RL-instruction_following",
        config,
        split="train",
    )
    print(data[0].keys())
    print(dataset[0])
    print(dataset[0]["prompt"])
    print(dataset[0]["extra_info"])

    dataset = data.map(format_data_if, remove_columns=data.column_names, load_from_cache_file=False, with_indices=True)
    local_dir = local_dir = os.getcwd()
    dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    return dataset