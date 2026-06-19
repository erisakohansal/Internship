from verifiable_instructions import instructions_registry
import re

# in verl the reward function is called once per generated response as opposed to trl where it's called on the whole batch
# https://verl.readthedocs.io/en/latest/preparation/reward_function.html

def if_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    solution_str : decoded model response
    extra_info : dataset metadata
    the reward manager detokenizes the response before calling the scoring function.
    """
    print_to_terminal = extra_info['print_to_terminal']
    debug_path = extra_info['debug_path']
    max_completion_length = extra_info['max_compeltion_length']

    def log(*args, **print_kwargs):
        """Print to terminal and append the same message to an external file."""
        if print_to_terminal:
            print(*args, **print_kwargs)

        with open(debug_path, "a", encoding="utf-8") as f:
            print(*args, **print_kwargs, file=f)

    instr_list = extra_info['instruction_id_list']
    kwarg_list = extra_info['kwargs']

    # A completion is "overlong" only if it hit the limit AND never emitted EOS
    # i.e. it was forcibly truncated, not a natural finish at exactly max length
    # is_overlong = (
    #     len(ids) >= max_completion_length
    #     and eos_id not in ids
    # )

    # if is_overlong:
    #     if should_debug:
    #         log(f"[overlong] length={len(ids)}, no EOS -> reward=0.0")

    #     return 0.0

    else:
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

                # Important: build_description sets internal fields used by check_following
                instruction.build_description(**filtered_kwargs)

                followed = instruction.check_following(solution_str)

                # Protect against buggy check_following functions returning None
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

        if extra_info['reward_mode'] == "binary":
            reward = float(all(is_following_list))
        else:
            reward = float(
                sum(is_following_list) / len(is_following_list)
                if is_following_list else 0.0
            )


        if should_debug:
            log("is_following_list:", is_following_list)
            log("Final reward:", reward)

        return reward