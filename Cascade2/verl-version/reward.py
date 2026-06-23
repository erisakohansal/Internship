from verifiable_instructions import instructions_registry
import re
import json

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
    max_completion_length = extra_info['max_compeltion_length']

    def log(*args, **print_kwargs):
        """Print to terminal and append the same message to an external file."""
        if print_to_terminal:
            print(*args, **print_kwargs)

        with open(debug_path, "a", encoding="utf-8") as f:
            print(*args, **print_kwargs, file=f)

    instr_list = extra_info.get('instruction_id_list', [])
    kwarg_list = extra_info.get('kwargs', [])

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

            if should_debug:
                log("Instruction:", instruction_id)
                log("kwargs:", filtered_kwargs)
                log("Followed:", followed)
                log("Reward contribution:", int(followed))

        except Exception as e:
            log(f"Error processing instruction {instruction_id} with kwargs {kw}: {e}")
            log("The corresponding completion was:")
            log(solution_str)
            is_following_list.append(False)

    if extra_info.get('reward_mode') == "binary":
        reward = float(all(is_following_list))
    else:
        reward = float(
            sum(is_following_list) / len(is_following_list)
            if is_following_list else 0.0
        )

    if should_debug:
        log("is_following_list:", is_following_list)
        log("Final reward:", reward)
    
    counter += 1

    return {"reward": reward, "acc": reward}  # binary, so acc == reward here


def tool_call_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    # should extract whats inside <tool_call> tags as the provided answer and compare to the ground_truth
    pattern = re.compile(r"<tool_call\s*>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(solution_str)
    for match in matches:
        if match:
            answer = match.group(1).strip()
            try:
                payload = json.loads(answer)
            except json.JSONDecodeError:
                return None

            if not isinstance(payload, dict):
                return None

            if set(payload.keys()) != {"name", "arguments"}:
                return None

            if not isinstance(payload["name"], str) or not payload["name"].strip():
                return None

            if not isinstance(payload["arguments"], dict):
                return None

        else:
            answer = None
            print("answer couldn't be extracted from the <tool_call> tags : ", solution_str)
            return 0.0
        

"""
apparently there is what's called an in memory sandbox necessary here, look into it
finish the IF-RL stage and the report!!!!!!!!!!!!!!!

"""
    
    
    
    