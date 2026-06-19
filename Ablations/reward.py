from math_verify import parse
from trl.rewards import accuracy_reward
import re
import numpy as np

counter = 0

def reward_func(completions, solution, **kwargs):  # simpler version of accuracy_reward
    """ 
    not only math answers, yes and no and mathematical expressions.
    for now the only problem caused by this is the max length being too short.
    """

    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion[0]["content"]) for completion in completions] 
    # contents = [match.group(1) if match else "" for match in matches]
    contents = []
    rewards = []

    for i, match in enumerate(matches):
        if match: 
            contents.append(match.group(1))
            if solution[i] == contents[-1]:
                rewards.append(1.)
            else:
                rewards.append(-1.)
        else: 
            print("prb, \ncompletions:", completions[i], "\nsol:", solution[i])
            rewards.append(-10.)


    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    # [1.0 if c == gt else -1. for c, gt in zip(contents, solution)]
    return rewards

def reward_func_reg(completions, solution, **kwargs): # from huggingface RLOOConfig docs

    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion[0]["content"]) for completion in completions] 
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else -1.0 for c, gt in zip(contents, solution)]

# def extract_answer(completions):
#     extracted = []
#     for c in completions:
#         content = c[0]["content"]
#         answer = content.split("<answer>")[-1].split("</answer>")[0].strip()
#         extracted.append([{
#             **c[0],
#             "content": answer
#         }])
#     return extracted

def extract_answer(completions):
    extracted = []
    for c in completions:
        content = c[0]["content"]
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        answer = ""   # no <answer> tag => no extracted answer
        if match:
            answer = match.group(1).strip()

        extracted.append([{
            **c[0],
            "content": answer
        }])
    return extracted


def reward_tmp(completions, solution, log_extra, **kwargs):
    extracted = extract_answer(completions)
    rewards = accuracy_reward(extracted, solution, log_extra, **kwargs)


    global counter
    if counter % 200 == 0:
        print("#"*100)
        print("solution", solution[0])
        for i, c in enumerate(completions):
            print(f"\n--- completion {i} ---")
            print(f"Solution: {solution[i]}, reward: {rewards[i]}, answer:{parse(completions[i][0]["content"])}")
            print(f"extracted : \n{extracted[i][0]["content"]}")
            print(f"completions : \n{c[0]["content"]}")
        print("#"*100)
    counter += 1

    return rewards 


def format_reward(completions, **kwargs): # DONE
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    rewards = [1. if elem else 0.0 for elem in matches]

    global counter
    if counter % 200 == 0:
        print("-"*50)
        for i, res in enumerate(responses):
            print("responses : ", res)
            print("matches : ", matches[i])
            print("rewards : ", rewards[i])
            print("\n\n")
        print("-"*50)
    return rewards