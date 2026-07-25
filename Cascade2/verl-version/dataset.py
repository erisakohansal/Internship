import re
import datasets
#from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset, Dataset
from collections import defaultdict
import json

import os

"""
TODO
no sandbox => confirmation YUP
parallel_tool_call and strict, where to use? NO NEED, CONSTANTS
tool structure => openai compatible YUP
system_prompt in workplace agents!!!!!!
how to include tools? in the system prompt? some other way? https://deepwiki.com/chengminhua/verl/7.4-chat-templates-and-tool-configuration
Test the workplace setup :) DONE!, what's hermes style formatting? structured prompt list


response_create_params has different fields for each task !!!
IF => not allowed to use any tools???

"""

class FormatData:

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

    SYSTEM_PROMPT_MCQA = """
    You are solving a multiple-choice question. 
    Reason if needed, but your final answer must be exactly one option letter inside LaTeX boxed format, for example: \\boxed{A}.
    """

    SYSTEM_PROMPT_STRUCTURED_OUTPUTS = """
    Your response must be a single JSON object that conforms exactly to the provided JSON schema.
    Output only the JSON object, with no surrounding prose, explanation, or markdown code fences.
    """

    dataset_languages = []
    filtered_languages = []

    @staticmethod
    def is_supported_language(example):        
        for dict_kw in example['kwargs']:

            if not isinstance(dict_kw, dict):
                continue

            if "language" in dict_kw:
                value = dict_kw["language"]  
                assert type(value) == str and len(value) == 2              
                normalized = str(value).strip().lower()
                FormatData.dataset_languages.append(normalized)
                if normalized not in FormatData.SUPPORTED_LANGUAGES:
                    return False
                
                FormatData.filtered_languages.append(normalized)
            
        return True
    
    @staticmethod
    def format_data_if(data, idx):
        # https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
        assert len(data['instruction_id_list']) > 0

        return {
            'data_source': 'nvidia/Nemotron-Cascade-2-RL-data',
            'prompt': [
                {'role': 'system', 'content': FormatData.SYSTEM_PROMPT_IF},
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
                'print_to_terminal': False,
                'debug_path': 'if_reward_fraction_verl.txt',
                'max_completion_length': 4000,
            },
        }


    @staticmethod
    def transform_tool_format(tools):
        """
        the tools in the nemotron dataset are in a 
        different format than what Qwen2.5 expects, 
        so we need to transform them into the expected format.
        From Hermes style to OpenAI style 
        """
        # format of the dicts
        res = []
        for tool in tools:
            res.append(
                {
                    "type": tool["type"],
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    }
                }
            )
        # "strict" and "parallel_tool_calls" are dropped -> Qwen doesn't use them
        return res


    @staticmethod
    def workplace_assistant_data(data, idx):
        """
        Related columns : 
            - responses_create_params : contains system + user messages and list of tool schemas
            - ground_truth : the tool call
            - 
        """
        assert len(data['responses_create_params']['input']) == 2
        if data['ground_truth']:
            assert len(data['ground_truth']) == 1

        # Extract system and user prompts
        input_msgs = data['responses_create_params']['input']
        system_prompt = input_msgs[0] if input_msgs[0]['role'] == 'system' else input_msgs[1]
        user_prompt = input_msgs[1] if input_msgs[0]['role'] == 'system' else input_msgs[0]

        tools = FormatData.transform_tool_format(data['response_create_params']['tools'])

        print("debug the workplace_assistant_data")
        print("\tSystem prompt: ", system_prompt)
        print("\tUser prompt: ", user_prompt)
        print("\n\tTools (before):", data['response_create_params']['tools'])
        print("\n\tTools (After):", tools)

        return {
            'agent_name': 'multi_domain_agent',
            'data_source': 'nvidia/Nemotron-Cascade-2-RL-data',
            'prompt': [
                system_prompt, 
                user_prompt
            ],
            'tools': tools,
            'ability': data['agent_ref']['name'].strip(),
            'reward_model': {
                'style': 'rule',
                'ground_truth': data['ground_truth'], 
            },
            'extra_info': {
                'split': 'train',
                'index': idx,
                'print_to_terminal': False,
                'max_completion_length': 4000,
            },
        }
    

    @staticmethod
    def mcqa_data(data, idx):
        """
        multi choice question answering, the model has to choose
        the correct answer and provide it in \boxed{} format (or any
        format available in the list of the 4 provided regex patterns).

        related columns:
            - responses_create_params : contains user message
            - expected_answer : the correct option
            - template_metadata : contains the output regex, if
                                  not provided or problems encountered, 
                                  fallback on reward_mode provided regex
                                  pattern.
            - options : choice of answers to the questions
        """
        return {
            'data_source': 'nvidia/Nemotron-Cascade-2-RL-data',
            'prompt': [
                {'role': 'system', 'content': FormatData.SYSTEM_PROMPT_MCQA},
                data['responses_create_params']['input'][0],
            ],
            'ability': data['agent_ref']['name'].strip(),
            'reward_model': {
                'style': 'rule',
                'ground_truth': data['expected_answer'],
            },
            'extra_info': {
                'split': 'train',
                'index': idx,
                
                'reward_mode': 'strict_single_letter_boxed', # 4 options available, backup regex pattern
                'template_metadata': data['template_metadata'],
                'options': data['options'],

                'print_to_terminal': False,
                'debug_path': 'if_reward_binary_verl.txt',
                'max_completion_length': 4000,
            },
        }
    

    @staticmethod
    def structured_outputs_data(data, idx):
        """
        structured outputs in json format
        related columns : 
            - responses_create_params : contains user message
            - schema_str : contains the outline of what the output 
                           should look like in json format (json schema)
        """
        if data['schema_type']: assert data['schema_type'].strip() == 'json'

        return {
            'data_source': 'nvidia/Nemotron-Cascade-2-RL-data',
            'prompt': [
                {'role': 'system', 'content': FormatData.SYSTEM_PROMPT_STRUCTURED_OUTPUTS},
                data['responses_create_params']['input'][0],
            ],
            'ability': data['agent_ref']['name'].strip(),
            'reward_model': {
                'style': 'rule',
                'ground_truth': data['schema_str'],
            },
            'extra_info': {
                'split': 'train',
                'index': idx,
                'print_to_terminal': False,
                'debug_path': 'if_reward_binary_verl.txt',
                'max_completion_length': 4000,
            },
        }


    @staticmethod
    def format_data_multi_domain(data, idx):

        match data['agent_ref']['name']:
            case "workplace_assistant_simple_agent":
                return FormatData.workplace_assistant_data(data, idx)

            case "mcqa_simple_agent":
                return FormatData.mcqa_data(data, idx)
            
            case "structured_outputs_simple_agent":
                return FormatData.structured_outputs_data(data, idx)


    @staticmethod
    def format_dataset_RL_Cascade2(config="IF-RL") -> Dataset:
        data = load_dataset(
            "nvidia/Nemotron-Cascade-2-RL-data",
            config,
            split="train",
        )

        print("Dataset columns : ", data.column_names)
        print("Raw dataset size : ", len(data))
        print(data[0]["agent_ref"])


        match config:
            case "IF-RL":
                data = data.filter(FormatData.is_supported_language, load_from_cache_file=False,)

                print("Filtered dataset size : ", len(data))

                print("All available languages in the dataset: ", len(set(FormatData.dataset_languages)), set(FormatData.dataset_languages))
                print("All supported languages by Qwen2.5-1.5B-Instruct: ", len(FormatData.SUPPORTED_LANGUAGES),  FormatData.SUPPORTED_LANGUAGES)
                print("All filtered languages : ", len(set(FormatData.filtered_languages)), set(FormatData.filtered_languages))

                dataset = data.map(
                    FormatData.format_data_if,
                    remove_columns=data.column_names,
                    load_from_cache_file=False,
                    with_indices=True,
                )


            case "multi-domain-RL":
                dataset = data.map(
                    FormatData.format_data_multi_domain,
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

        # train_set.to_parquet(os.path.join(local_dir, config+'-fraction-train.parquet'))
        # test_set.to_parquet(os.path.join(local_dir, config+'-fraction-test.parquet'))
        return train_set, test_set

if __name__ == "__main__":
    # Uncomment one of the following:
    FormatData.format_dataset_RL_Cascade2(config="multi-domain-RL")
    # test_IF_RL_config()