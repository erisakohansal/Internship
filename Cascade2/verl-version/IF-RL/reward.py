from verifiable_instructions import instructions_registry
import re
import json
from dataset import FormatData

# in verl the reward function is called once per generated response as opposed to trl where it's called on the whole batch
# https://verl.readthedocs.io/en/latest/preparation/reward_function.html

counter = 0
debug_every = 20480*3

def if_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    solution_str : decoded model response
    extra_info : dataset metadata
    the reward manager detokenizes the response before calling the scoring function.
    """
    global counter, debug_every
    should_debug = counter % debug_every == 0

    print_to_terminal = extra_info['print_to_terminal']
    debug_path = extra_info['debug_path']

    def log(*args, **print_kwargs):
        """Print to terminal and append the same message to an external file."""
        if print_to_terminal:
            print(*args, **print_kwargs)

        with open(debug_path, "a", encoding="utf-8") as f:
            print(*args, **print_kwargs, file=f)

    instr_list = extra_info['instruction_id_list']
    kwarg_list = extra_info['kwargs']

    is_following_list = []
    if should_debug:
        log("#" * 100)
        
    for instruction_id, kw in zip(instr_list, kwarg_list):
        try:
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            if kw is None:
                kw = {}

            if "language" in kw:
                assert kw["language"] in FormatData.SUPPORTED_LANGUAGES

            filtered_kwargs = {
                k: v for k, v in kw.items()
                if v is not None
            }

            instruction.build_description(**filtered_kwargs)
            followed = instruction.check_following(solution_str)
            followed = bool(followed)
            is_following_list.append(followed)

            if should_debug:
                log("-" * 100)
                log("Instruction:", instruction_id)
                log("kwargs:", filtered_kwargs)
                log("Followed:", followed)
                log("Reward contribution:", int(followed))
                log("-" * 100)

        except Exception as e:
            log("-" * 100)
            log(f"Error processing instruction {instruction_id} with kwargs {kw}: {e}")
            log("The corresponding completion was:")
            log(solution_str)
            is_following_list.append(False)
            log("-" * 100)

    reward_mode = extra_info['reward_mode'] 
    assert reward_mode == "fraction"
    if reward_mode == "binary":
        reward = float(all(is_following_list))

    elif reward_mode == "fraction":
        reward = float(
            sum(is_following_list) / len(is_following_list)
            if is_following_list else 0.0
        )
    
    else:
        raise ValueError(f"Invalid reward mode: {reward_mode}")

    if should_debug:
        log("is_following_list:", is_following_list)
        log("Final reward:", reward)
    
    counter += 1
    assert type(reward) is float
    return reward  # reward == acc

