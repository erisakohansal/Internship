from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig

"""
boxed_match_priority=0 tells Math-Verify to prioritize \boxed{...} matches.
If several matches have the same priority, it generally tries the last one first—useful because the final answer is normally near the end.
Important: this prioritizes boxed answers but does not necessarily require them. It can still extract other LaTeX environments such as $42$.
"""
_ANSWER_EXTRACTION_CONFIG = [LatexExtractionConfig(boxed_match_priority=0)]


def calculator_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    predicted_actions = extra_info.get("predicted_actions", [])

    try:
        gold_text = str(ground_truth).strip()

        # Also supports an original GSM8K answer such as "... #### 308".
        if "####" in gold_text:
            gold_text = gold_text.rsplit("####", 1)[-1].strip()

        # LatexExtractionConfig needs a latex environment.
        gold_text = f"${gold_text}$"
        gold = parse(gold_text, extraction_config=_ANSWER_EXTRACTION_CONFIG, parsing_timeout=None)
        answer = parse(solution_str, extraction_config=_ANSWER_EXTRACTION_CONFIG, parsing_timeout=None)
        correct = verify(gold, answer, timeout_seconds=None)
    except Exception as e:
        print(f"[math_tool_reward_fn] parse/verify error: {e}")
        correct = False

    reward = float(correct) * (1. if predicted_actions else .5)
    return reward


if __name__=="__main__":
    sample = {"input": "system\nYou are a careful math problem solver. You may call the `calculator` tool for arithmetic you are not fully confident in. Reason step by step, then give your final answer as \\boxed{answer}.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"calculator\", \"description\": \"Evaluate an arithmetic expression, e.g. '48 + 48/2'.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"expression\": {\"type\": \"string\", \"description\": \"The arithmetic expression to evaluate.\"}}, \"required\": [\"expression\"]}, \"strict\": false}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\nuser\nLeah just got a new kitten. Spaying her cost $200, her 3 vaccines costs $20 each, and the kitten broke 4 vases that cost $12 each. How much has the kitten cost so far? Let's think step by step and output the final answer after \"####\".\nassistant\n", "output": "To calculate the total cost of the kitten, we need to consider the cost of spaying, the cost of vaccines, and the cost of the broken vases. Let's break it down step by step:\n\n1. **Spaying Cost**: The spaying cost is $200.\n2. **Vaccines Cost**: The kitten received 3 vaccines, each costing $20. So, the total cost for the vaccines is \\(3 \\times 20 = $60\\).\n3. **Broken Vases Cost**: The kitten broke 4 vases, each costing $12. So, the total cost for the broken vases is \\(4 \\times 12 = $48\\).\n\nNow, let's add up all these costs:\n\n- Spaying cost: $200\n- Vaccine cost: $60\n- Broken vase cost: $48\n\nTotal cost = $200 + $60 + $48 = $308\n\nTherefore, the total cost of the kitten so far is $308.\n\n#### Final Answer: \\(\\boxed{308}\\)", "gts": "308", "score": 0.0, "step": 50, "reward": 0.0, "acc": 0.0, "uid": "007e8393-8967-45b2-a790-23984713c903_0_0"}

    reward = calculator_reward_fn(None, sample["output"], sample["gts"], None)
    print(reward)
    print(verify(parse("$4/48$", extraction_config=_ANSWER_EXTRACTION_CONFIG, parsing_timeout=None),parse("\\boxed{\\frac{1}{12}}", extraction_config=_ANSWER_EXTRACTION_CONFIG, parsing_timeout=None)))