from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig

_ANSWER_EXTRACTION_CONFIG = [LatexExtractionConfig(boxed_match_priority=0)]

counter = 0
debug_every = 256


def calculator_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """Reward for the math+calculator-tool multi-turn task.

    solution_str: the decoded final assistant turn (VeRL detokenizes the
    full trajectory response before calling this function).
    ground_truth: the reference answer, stored at dataset-prep time already
    wrapped for math_verify (e.g. "$42$" or "\\boxed{1/2}").
    """
    global counter

    should_debug = counter % debug_every == 0
    counter += 1

    try:
        gold = parse(ground_truth, extraction_config=_ANSWER_EXTRACTION_CONFIG)
        answer = parse(solution_str, extraction_config=_ANSWER_EXTRACTION_CONFIG)
        correct = bool(gold) and bool(answer) and verify(gold, answer)
    except Exception as e:
        if should_debug:
            print(f"[math_tool_reward_fn] parse/verify error: {e}")
        correct = False

    reward = 1.0 if correct else 0.0

    if should_debug:
        print("-" * 100)
        print("Ground truth:", ground_truth)
        print("Solution (tail):", solution_str[-500:])
        print("Correct:", correct)
        print("-" * 100)

    return reward