from datasets import load_dataset, Dataset
import re
import numpy as np
from collections import Counter, defaultdict
import json


prompt_length = []
type_agent_ref = []
agents = []
tools = []
parallel_tool_calls = []
temperature = []
strict_ = []
fields_in_tools = []

environment_name = []
category = []
schema_type = []
schema_str = []
schema_fields_count = []
template_metadata_output_regex = []
languages = []


SYSTEM_PROMPT_IF = """
You are a helpful and harmless assistant.
You are not allowed to use any tools.
""" 

# Always think step by step before answering inside <think></think> tags, then give your final answer.

def format_data_if(data):
    # some tests
    """
    global prompt_length
    prompt_length.append(len(data['prompt'].strip()))
    assert len(data['responses_create_params']['input']) == 1 and data['responses_create_params']['input'][0]['content'].strip() == data['prompt'].strip()

    global tools, parallel_tool_calls
    tools.append(data['responses_create_params']['tools'] if data['responses_create_params']['tools'] != [] else None)
    parallel_tool_calls.append(data['responses_create_params']['parallel_tool_calls'])

    global agents, type_agent_ref
    type_agent_ref.append(data['agent_ref']['type'])
    agents.append(data['agent_ref']['name'])

    global instruction_id_list
    for elem in data['instruction_id_list']:
        instruction_id_list.append(elem)
    """
    global languages
    for kw in data['kwargs']:
        if kw and "language" in kw.keys():
            languages.append(kw["language"])
    res = {
        'prompt' : [
            {'role': 'system', 'content': SYSTEM_PROMPT_IF},
            data['responses_create_params']['input'][0],
        ],
        'instruction_id_list': data['instruction_id_list'],
        'kwargs': data['kwargs'],
    }
    return res

def format_data_multi_domain(data): # specify the regex pattern here with hydra
    # some tests
    """
    global prompt_length
    prompt_length.append(len(data['responses_create_params']['input'][0]['content'].strip()))

    global agents, type_agent_ref, environment_name, category
    type_agent_ref.append(data['agent_ref']['type'])
    agents.append(data['agent_ref']['name'])
    environment_name.append(data['environment_name'].strip() if data['environment_name'] else data['environment_name'])
    category.append(data['category'].strip() if data['category'] else data['category'])

    global schema_type, schema_str, schema_fields_count
    schema_type.append(data['schema_type'].strip() if data['schema_type'] else data['schema_type'])
    schema_str.append(data['schema_str'].strip() if data['schema_str'] else data['schema_str'])
    schema_fields_count.append(data['schema_fields_count'].strip() if data['schema_fields_count'] else data['schema_fields_count'])

    # if data['schema_str']:
    #     print("schema_str:", type(data['schema_str']), len(json.loads(data['schema_str'])["required"]))
    #     print(json.loads(data['schema_str'])["required"])
    #     assert int(data['schema_fields_count']) == len(json.loads(data['schema_str'])["required"]), \
    #     f"schema_fields_count {data['schema_fields_count']} does not match number of fields in schema_str {len(json.loads(data['schema_str'])["required"])}"
    """

    if data['agent_ref']['name'].strip() == 'mcqa_simple_agent':  # verify everything related to the mcqa agent
        # seems that the output regex exists for the concerned samples
        assert "output_regex" in data['template_metadata'] and data['template_metadata']['output_regex'] is not None, \
            "output_regex is missing in template_metadata or is None"

        # verify options 
        for entry in data['options']:
            for k, v in entry.items():
                assert isinstance(k, str) and len(k) == 1 and k.isalpha() and k.isupper(), f"Option key '{k}' is not a single uppercase letter"
                assert isinstance(v, str) and v.strip() != "", f"Option value for key '{k}' is not a non-empty string"


    if data['agent_ref']['name'].strip() == 'workplace_assistant_simple_agent':
        global parallel_tool_calls, temperature, strict_, fields_in_tools
        parallel_tool_calls.append(data['responses_create_params'].get("parallel_tool_calls"))
        temperature.append(data['responses_create_params'].get("temperature"))
        assert len(data['responses_create_params']['input']) == 2
        # print(data['responses_create_params']['input'])
        if data['responses_create_params']['tools'] and len(data['responses_create_params']['input']) > 2:
            print("multi turn")
            return

        for tool in data['responses_create_params']['tools']:
            strict_.append(tool.get("strict"))
            tool_keys = list(tool.keys())
            # print(type(tool_keys))
            for elem in ["type", "name", "description", "parameters"]:
                tool_keys.remove(elem)

            fields_in_tools.extend(tool_keys)
        


    res = {
        'prompt' : [
            {'role': 'system', 'content': SYSTEM_PROMPT_IF},
            data['responses_create_params']['input'][0],
        ],
        'agent_ref': data['agent_ref']['name'].strip(), # 3 types of agents

        # mcqa
        'options': data['options'] if data['options'] else None,
        'expected_answer': data['expected_answer'].strip() if data['expected_answer'] else data['expected_answer'], 
        'template_regex': data['template_metadata']['output_regex'] if data['template_metadata'] else data['template_metadata'],
        
        # agentic
        'ground_truth': data['ground_truth'],
        'category': data['category'].strip() if data['category'] else data['category'], 

        # structured output
        'schema_fields_count': data['schema_fields_count'].strip() if data['schema_fields_count'] else data['schema_fields_count'], 
        'schema_str': data['schema_str'].strip() if data['schema_str'] else data['schema_str'],

    }
    return res


def format_dataset_RL_Cascade2(config="IF-RL") -> Dataset:

    # => filter out non supported languages
    data = load_dataset(
        "nvidia/Nemotron-Cascade-2-RL-data",  #"nvidia/Nemotron-RL-instruction_following",
        config,
        split="train",
    )
    print(data.column_names)

    match config:
        case "IF-RL": dataset = data.map(format_data_if, remove_columns=data.column_names, load_from_cache_file=False)
        case "multi-domain-RL": dataset = data.map(format_data_multi_domain, remove_columns=data.column_names, load_from_cache_file=False)
    
    """
    # Determines the maximum prompt length*
    global prompt_length
    tmp = np.array(prompt_length)
    print("max len:", tmp.max()) # 4565

    # Check if parallel tool call and tools values differ
    global tools, parallel_tool_calls
    print("=" * 80)
    print("TOOLS VALUES")
    print("=" * 80)
    print(set(tools))
    print("\n")
    print("=" * 80)
    print("PARALLEL TOOL CALLS VALUES")
    print("=" * 80)
    print(set(parallel_tool_calls))


    global agents, type_agent_ref, environment_name, category, schema_type, schema_str, schema_fields_count
    print("agent types:", set(agents)) 
    print("agent_ref types:", set(type_agent_ref)) 
    print("environment_name types:", set(environment_name)) 
    print("category types:", set(category))
    print("schema_type types:", set(schema_type)) 
    print("schema_str types:", len(set(schema_str)))
    print("schema_fields_count types:", set(schema_fields_count)) 
    """

    print("languages : ", set(languages))
    print("parallel_tool_calls : ", set(parallel_tool_calls))
    print("temperature : ", set(temperature))
    print("strict : ", set(strict_))
    print("remaining fields : ", set(fields_in_tools))

    return dataset


format_dataset_RL_Cascade2(config="multi-domain-RL")

"""
# Test for if they resolved the inconsistencies they talked about
cim_samples = [s for s in data
            if any("count_increment_word" in i 
                    for i in s["instruction_id_list"])]

# Look at the kwargs for these samples
for s in cim_samples:
    print(s["kwargs"])

# List of different reward function categories
instruction_counter = Counter()
kwargs_per_instruction = defaultdict(set)

for sample in data:
    for instruction_id, kw in zip(sample["instruction_id_list"], sample["kwargs"]):
        instruction_counter[instruction_id] += 1
        
        if kw is not None:
            # store the kwarg keys (not values) to understand the schema
            kwargs_per_instruction[instruction_id].add(
                tuple(sorted(kw.keys()))
            )
        else:
            kwargs_per_instruction[instruction_id].add(())

print(f"Total unique instruction types: {len(instruction_counter)}\n")
print(f"{'Instruction ID':<50} {'Count':>8} Kwarg keys ")
print("-" * 100)
for instruction_id, count in instruction_counter.most_common():
    kwarg_schemas = kwargs_per_instruction[instruction_id]
    print(f"{instruction_id:<50} {count:>8} {kwarg_schemas} ")
"""