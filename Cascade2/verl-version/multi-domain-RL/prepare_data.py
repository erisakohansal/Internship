import re
import datasets
#from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset, Dataset
from collections import defaultdict
import json

import os


SYSTEM_PROMPT_MCQA = """
You are solving a multiple-choice question. 
RYou may reason before answering but your final answer must be exactly one option letter inside LaTeX boxed format, for example: \\boxed{A}.
"""

SYSTEM_PROMPT_STRUCTURED_OUTPUTS = """
Your response must be a single JSON object that conforms exactly to the provided JSON schema.
Output only the JSON object, with no surrounding prose, explanation, or markdown code fences.
"""



def workplace_assistant_data(data, idx):
    """
    Relevant columns:
        - responses_create_params:
            system/user messages and tool schemas
        - ground_truth:
            list of reference tool actions; may contain 0-8 actions
        - category:
            Workplace subdomain, such as email or calendar
        - environment_name:
            Workplace environment identifier
        - agent_ref:
            original NeMo agent metadata
    The YAML registers tool implementations and schemas globally.
    This row selects the tools available for this particular prompt.
    https://github.com/verl-project/verl/blob/v0.5.0/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml#L6
    """
    params = data["responses_create_params"]

    # Validate and extract the prompt.
    input_msgs = params["input"]
    assert len(input_msgs) == 2

    system_msgs = [
        dict(message)
        for message in input_msgs
        if message["role"] == "system"
    ]
    user_msgs = [
        dict(message)
        for message in input_msgs
        if message["role"] == "user"
    ]

    assert len(system_msgs) == 1
    assert len(user_msgs) == 1

    system_prompt = system_msgs[0]
    user_prompt = user_msgs[0]

    tool_names = [
        tool["name"]
        for tool in params.get("tools", [])
    ]

    assert len(tool_names) == len(set(tool_names)), (
        f"Duplicate tool names for sample {idx}"
    )

    # Validate reference actions without assuming there is only one.
    # Do not use `or []`, because that could hide invalid falsy values.
    ground_truth = data.get("ground_truth")
    if ground_truth is None:
        ground_truth = []

    assert isinstance(ground_truth, list)

    ground_truth_names = set()

    for action in ground_truth:
        assert isinstance(action, dict)
        assert isinstance(action.get("name"), str)
        assert isinstance(action.get("arguments"), str)

        arguments = json.loads(action["arguments"])
        assert isinstance(arguments, dict)

        ground_truth_names.add(action["name"])

    # Every reference action must use a tool exposed to the model.
    unavailable_tools = ground_truth_names - set(tool_names)
    assert not unavailable_tools, (
        f"Ground-truth tools not exposed for sample {idx}: "
        f"{sorted(unavailable_tools)}"
    )

    return {
        'agent_name': 'tool_agent',
        'data_source': 'nvidia/Nemotron-Cascade-2-RL-data',
        'prompt': [
            system_prompt, 
            user_prompt
        ],
        'tool_selection': tool_names,
        'ability': 'workplace_assistant',
        'reward_model': {
            'style': 'rule',
            'ground_truth': json.dumps(data['ground_truth']), 
        },
        'extra_info': {
            'split': 'train',
            'index': idx,
            'agent_ref': data['agent_ref']['name'].strip(),

            'reward_mode': None,
            'template_metadata': None,
            'options': None,
        },
    }


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
    assert len(data['responses_create_params']['input']) == 1

    return {
        'agent_name': 'single_turn_agent',
        'data_source': 'nvidia/Nemotron-Cascade-2-RL-data',
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT_MCQA},
            data['responses_create_params']['input'][0],
        ],
        'tool_selection': None,
        'ability': 'mcqa',
        'reward_model': {
            'style': 'rule',
            'ground_truth': data['expected_answer'].strip(),
        },
        'extra_info': {
            'split': 'train',
            'index': idx,
            'agent_ref': data['agent_ref']['name'].strip(),
            
            'reward_mode': 'strict_single_letter_boxed', # 4 options available, backup regex pattern
            'template_metadata': data['template_metadata'],
            'options': data['options'],
        },
    }


def structured_outputs_data(data, idx):
    """
    structured outputs in json format
    related columns : 
        - responses_create_params : contains user message
        - schema_str : contains the outline of what the output 
                        should look like in json format (json schema)
    """
    if data['schema_type']: assert data['schema_type'].strip() == 'json'
    messages = data["responses_create_params"]["input"]

    assert len(messages) in {1, 2}
    assert all(message["role"] == "user" for message in messages)
    assert all(message.get("content", "").strip() for message in messages)

    schema_str = data["schema_str"]
    assert isinstance(schema_str, str) and schema_str.strip()

    return {
        'agent_name': 'single_turn_agent',
        'data_source': 'nvidia/Nemotron-Cascade-2-RL-data',
        'prompt': [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_STRUCTURED_OUTPUTS,
            },
            *messages,
        ],
        'tool_selection': None,
        'ability': 'structured_outputs',
        'reward_model': {
            'style': 'rule',
            'ground_truth': schema_str,
        },
        'extra_info': {
            'split': 'train',
            'index': idx,
            'agent_ref': data['agent_ref']['name'].strip(),

            'reward_mode': None,
            'template_metadata': None,
            'options': None,
        },
    }


def format_data_multi_domain(data, idx):

    match data['agent_ref']['name']:
        case "workplace_assistant_simple_agent":
            return workplace_assistant_data(data, idx)

        case "mcqa_simple_agent":
            return mcqa_data(data, idx)
        
        case "structured_outputs_simple_agent":
            return structured_outputs_data(data, idx)
        

def format_dataset_multi_domain(config="multi-domain-RL") -> Dataset:
    data = load_dataset(
        "nvidia/Nemotron-Cascade-2-RL-data",
        config,
        split="train",
    )

    print("Dataset columns : ", data.column_names)
    print("Raw dataset size : ", len(data))
    print(data[0]["agent_ref"])


    dataset = data.map(
        format_data_multi_domain,
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



if __name__ == "__main__":
    # FormatData.format_dataset_RL_Cascade2(config="multi-domain-RL")
    PWD="/project/scratch/p201382/erisa/Internship/Cascade2/verl-version/multi-domain-RL"
    TRAIN_FILE=f"{PWD}/multi-domain-RL-train.parquet"
    import pandas as pd
    df = pd.read_parquet(TRAIN_FILE)
    structured_df = df[
    df["extra_info"].apply(
            lambda extra_info: extra_info.get("agent_ref") == "structured_outputs"
        )
    ]

    row = structured_df.iloc[0]
    print(type(row["reward_model"]["ground_truth"]))
    print(row["reward_model"]["ground_truth"])