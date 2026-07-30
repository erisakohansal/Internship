import re
import datasets
#from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset, Dataset
from collections import defaultdict
import json

import os

# Total languages in the dataset : 30
# sw, ta, fr, en, th, bn, es, ja, de, ko,gu, ne, te, ur, ru, ml, kn, 
# bg, fa, pa, it, mr, pl, he, pt, hi, fi, vi, uk, ar

# . means supported by langdetect
SUPPORTED_LANGUAGES = { # https://deepwiki.com/QwenLM/Qwen2.5/5.3-multilingual-support
    "en", # english    (y).
    "fr", # french     (y).
    "es", # spanish    (y).
    "pt", # portuguese (y).
    "de", # german     (y).
    "it", # italian    (y).
    "ru", # russian    (y).
    "nl", # dutch
    "pl", # polish     (y).
    "cs", # czech
    "ro", # romanian
    "uk", # ukrainian  (y).
    # chinese is zh
    "zh-cn", # simplified  
    "zh-tw", # traditional 
    "ja", # japanese   (y).
    "ko", # korean     (y).
    "vi", # vietnamese (y).
    "th", # thai       (y).
    "id", # indonesian
    "ms", # malaysian
    "hi", # hindi      (y).
    "bn", # bengali    (y).
    "ur", # urdu       (y).
    "ar", # arabic     (y).
    "fa", # persian    (y).
    "tr", # turkish
    "sw", # swahili    (y).
    "am", # amharic
    "el", # greek
    "he", # hebrew     (y).
} 
# not in Qwen2.5: ta(tamil), te(telugu), gu(gujarati), ne(nepali), ml(malayalam), 
# kn(kannada), bg(bulgarian), pa(punjabi), mr(marathi), fi(finnish)

SYSTEM_PROMPT_IF = """
You are a helpful and harmless assistant.
You are not allowed to use any tools.
""" 


dataset_languages = []
filtered_languages = []


def is_supported_language(example):  
    global dataset_languages, filtered_languages
          
    for dict_kw in example['kwargs']:

        if not isinstance(dict_kw, dict):
            continue

        if "language" in dict_kw:
            value = dict_kw["language"]  
            assert type(value) == str and len(value) == 2              
            normalized = str(value).strip().lower()
            dataset_languages.append(normalized)
            if normalized not in SUPPORTED_LANGUAGES:
                return False
            
            filtered_languages.append(normalized)
        
    return True


def format_data_if(data, idx):
    # https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
    assert len(data['instruction_id_list']) > 0

    return {
        'data_source': 'nvidia/Nemotron-Cascade-2-RL-data',
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT_IF},
            data['responses_create_params']['input'][0],
        ],
        'ability': 'instruction_following',
        'reward_model': {
            'style': 'rule',
            'ground_truth': None,
        },
        'extra_info': {
            'split': 'train',
            'index': idx,
            'reward_mode': 'fraction',
            'instruction_id_list': data['instruction_id_list'],
            'kwargs': data['kwargs'],
        },
    }
        

def format_dataset_if_rl(config="IF-RL") -> Dataset:
    data = load_dataset(
        "nvidia/Nemotron-Cascade-2-RL-data",
        config,
        split="train",
    )

    print("Dataset columns : ", data.column_names)
    print("Raw dataset size : ", len(data))
    print(data[0]["agent_ref"])

    data = data.filter(is_supported_language, load_from_cache_file=False,)

    print("Filtered dataset size : ", len(data))

    print("All available languages in the dataset: ", len(set(dataset_languages)), set(dataset_languages))
    print("All supported languages by Qwen2.5-1.5B-Instruct: ", len(SUPPORTED_LANGUAGES),  SUPPORTED_LANGUAGES)
    print("All filtered languages : ", len(set(filtered_languages)), set(filtered_languages))

    dataset = data.map(
        format_data_if,
        remove_columns=data.column_names,
        load_from_cache_file=False,
        with_indices=True,
    )

    print(dataset[0])
    print(dataset[0]["prompt"])
    print(dataset[0]["extra_info"])


    splits = dataset.train_test_split(
        test_size=0.05,
        seed=42, 
        shuffle=True,
    )

    train_set = splits['train']
    test_set = splits['test']
    local_dir = os.getcwd()

    print("\tSize of the train split : ", len(train_set))
    print("\tSize of the test split : ", len(test_set))

    train_set.to_parquet(os.path.join(local_dir, config+'-train.parquet'))
    test_set.to_parquet(os.path.join(local_dir, config+'-test.parquet'))
    return train_set, test_set