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
    score = float(reward.item()) if hasattr(reward, "item") else float(reward)
    assert type(score) is float
    return score  # binary, so acc == reward here


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
"""
    
def mcqa_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    solution_str : decoded model response
    extra_info : dataset metadata
    the reward manager detokenizes the response before calling the scoring function.
    there are 4 different grading modes for mcqa: 
    - "strict_single_letter_boxed" : the answer must be a single letter (A, B, C, D) and it must be boxed (e.g., [A], (B), {C}, <D>)
    - "lenient_boxed" : the answer can be a single letter (A, B, C, D) or a word (e.g., "A", "B", "C", "D") and it must be boxed (e.g., [A], (B), {C}, <D>)
    - "lenient_answer_colon" : the answer can be a single letter (A, B, C, D) or a word (e.g., "A", "B", "C", "D") and it can be preceded by a colon (e.g., ": A", ": B", ": C", ": D")
    - "lenient_answer_colon_md" : the answer can be a single letter (A, B, C, D) or a word (e.g., "A", "B", "C", "D") and it can be preceded by a colon (e.g., ": A", ": B", ": C", ": D") and it can be in markdown format (e.g., "**A**", "*B*", "__C__", "_D_")
    """

    reward_mode = extra_info["reward_mode"]
    output_regex = extra_info["template_metadata"]["output_regex"]
    options = extra_info["options"]

    # allowed letters extracted from the options 
    letters: set[str] = set()
    if options:
        for entry in options:
            # Exclude null values
            for k, v in entry.items():
                if isinstance(k, str) and len(k) == 1 and k.isalpha() and v is not None:
                    letters.add(k.upper())
    return letters

    if isinstance(output_regex, str):
        """Parse answer using custom regex from template_metadata.

        Uses rightmost (last) match to handle reasoning before final answer.
        Case-insensitive matching to handle capitalization variations.

        When using template_metadata with custom regex, we trust the regex pattern
        and allow extracted letters even if options metadata is incomplete.
        """
        try:
            # Use IGNORECASE flag and findall to get all matches
            matches = re.findall(regex_pattern, text, re.IGNORECASE)
            if not matches:
                return None

            # Take the LAST match (rightmost)
            text = matches[-1].strip()
            captured = (
                text.replace("أ", " A")
                .replace("ب", " B")
                .replace("ج", " C")
                .replace("د", " D")
                .replace("অ", " A")
                .replace("ব", " B")
                .replace("ড", " C")
                .replace("ঢ", " D")
                .replace("Ａ", " A") # different A character ????
                .replace("Ｂ", " B")
                .replace("Ｃ", " C")
                .replace("Ｄ", " D")
                .strip()
            ).upper()
            
          

            # Try direct letter match first
            if len(captured) == 1 and captured.isalpha():
                # If we have options metadata, validate against it
                if allowed_letters and captured in allowed_letters:
                    return captured
                # If options metadata is missing/incomplete, trust the regex
                # This handles cases where template_metadata regex is used but options are incomplete
                elif not allowed_letters:
                    return captured
                # If captured letter is not in allowed_letters but allowed_letters exists,
                # it might be a data quality issue - still return it when using template_metadata
                else:
                    # Trust the regex when using template_metadata (this function is only called for template_metadata)
                    return captured

            # Try matching against option text (normalized)
            normalized_captured = " ".join(captured.lower().split()) # _normalize_for_match function from verl
            for entry in options or []:
                for k, v in entry.items():
                    if v is not None and k.upper() in allowed_letters and " ".join(v.lower().split()) == normalized_captured:
                        return k.upper()

            return None
        except re.error:
            # Invalid regex pattern, return None
            return None

    # if template metadata didn't work etc, it relies on the default grading mode   
    
    
