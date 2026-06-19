from datasets import load_dataset, Dataset
import re
import numpy as np

SYSTEM_PROMPT_FORMAT = r"""Solve the problem.

Use the following structure:

<think>
reason here
</think>
<answer>
explain here
\boxed{final answer}
</answer>

Rules:
- Always put reasoning inside <think>...</think>.
- Always put the final answer inside <answer>...</answer>.
- The <answer> section may contain an explanation, but must contain exactly one \boxed{}.
- The \boxed{} must contain the final answer.
- Do not put the final answer inside <think>.
- Do not use more than one \boxed{}.
- Put only the final numerical answer inside \boxed{}.
"""

# SYSTEM_PROMPT = r"""Solve the problem.

# Return exactly one final numerical answer inside a single LaTeX boxed expression:

# \boxed{...}

# Rules:
# - Use exactly one \boxed{} expression.
# - Do not output multiple boxed expressions.
# - Put only the final numerical answer inside \boxed{}.

# Examples:
# \boxed{3}
# \boxed{-50}
# \boxed{1080}
# """

SYSTEM_PROMPT_BOXED = r"""Solve the problem.

Return exactly one final numerical answer inside a single LaTeX boxed expression:

\boxed{...}

Rules:
- Return only the final answer.
- Do not use more than one \boxed{}.
- Always put only the final numerical answer inside \boxed{}.
- Remember to provide your final answer clearly within the \boxed{} format.
"""



# DOLCI -------------------------------------------------------------------------------

def format_data_dolci_math(data):
    res = {
        "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_BOXED},
                data["messages"][0]
            ],
            "solution": r"\boxed{"+data["ground_truth"].strip()+"}", # need to do this for parsed in accuracy_reward
        }
    return res

def format_dataset_dolci_math() -> Dataset:
    """
    has latex answers => \fraq, \left, \sqrt, \text{No}, \log, \dfraq, \pi, \overline, \sin, etc
    """
    dataset = load_dataset("allenai/Dolci-RL-Zero-Math-7B", split="train")  
    formatted = dataset.map(format_data_dolci_math, remove_columns=dataset.column_names)

    idx = formatted.train_test_split(test_size=0.1) # create a test split that is 10% of the original dataset:
    train_dataset = idx["train"]
    test_dataset = idx["test"]
    return train_dataset, test_dataset

# GSM8K --------------------------------------------------------------------------------
# generation for this dataset needs less tokens than dolci (around 256)
# answers are always provided after ####
# doesn't have any latex format answers, only numbers with "," in them (ex: 1,080) and negative numbers that .isdigit() recognizes as False
def format_data_gsm8k(data):
    # assert "####" in data['answer']
    # assert data['answer'].split("####")[1].strip().isdigit()

    res = {
        'prompt' : [
            {'role': 'system', 'content': SYSTEM_PROMPT_FORMAT},
            {'role': 'user', 'content': data['question'].strip()}
        ],
        'solution': r"\boxed{"+data['answer'].split("####")[1].strip()+"}"
    }
    return res

def format_dataset_gsm8k(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(format_data_gsm8k, remove_columns=data.column_names, load_from_cache_file=False)
    return data 
