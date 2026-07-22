from verifiable_instructions import instructions_registry
import re
import json


# in verl the reward function is called once per generated response as opposed to trl where it's called on the whole batch
# https://verl.readthedocs.io/en/latest/preparation/reward_function.html

def if_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    solution_str : decoded model response
    extra_info : dataset metadata
    the reward manager detokenizes the response before calling the scoring function.
    """

    print_to_terminal = extra_info['print_to_terminal']
    # debug_path = extra_info['debug_path']
    debug_path = "/mnt/tier1/project/p201382/erisa/Internship/Cascade2/verl-version/IF-RL/logs.txt"

    instr_list = extra_info['instruction_id_list']
    kwarg_list = extra_info['kwargs']
    reward_mode = extra_info['reward_mode'] 

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

            instruction.build_description(**filtered_kwargs)
            followed = instruction.check_following(solution_str)
            followed = bool(followed)
            is_following_list.append(followed)

        except Exception as e:
            is_following_list.append(False)

    
    if reward_mode == "binary":
        reward = float(all(is_following_list))

    elif reward_mode == "fraction":
        reward = float(
            sum(is_following_list) / len(is_following_list)
            if is_following_list else 0.0
        )
    
    else:
        raise ValueError(f"Invalid reward mode: {reward_mode}")

    return reward  
