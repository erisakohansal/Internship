from verifiable_instructions import instructions_registry
import re
from typing import Tuple, List, Dict, Optional, Any
import json
"""
run once
import nltk
nltk.download('punkt_tab')
"""


class IFReward:
    def __init__(self,
        max_completion_length,
        reward_mode,
        debug_path="if_reward_binary_systemprompt.txt",
        debug_every=20480*3,
        print_to_terminal=True,
    ):
        self.__name__ = "IFReward"
        self.max_completion_length = max_completion_length
        self.reward_mode = reward_mode
        self.debug_path = debug_path
        self.debug_every = debug_every
        self.print_to_terminal = print_to_terminal
        self.eos_token = []
        self.counter = 0
    
    def set_eos_token(self, eos_token):
        if eos_token is None:
            self.eos_token = []
        elif isinstance(eos_token, int):
            self.eos_token = [eos_token]
        else : 
            self.eos_token = list(eos_token)

    def log(self, *args, **print_kwargs):
        """Print to terminal and append the same message to an external file."""
        if self.print_to_terminal:
            print(*args, **print_kwargs)

        with open(self.debug_path, "a", encoding="utf-8") as f:
            print(*args, **print_kwargs, file=f)
    
    def __call__(self, completion_ids, completions, instruction_id_list, kwargs, **extra):
        self.log("\t\tNumber of completions received:", len(completions))

        rewards = []

        for ids, completion, instr_list, kwarg_list in zip(completion_ids, completions, instruction_id_list, kwargs):
            should_debug = self.counter % self.debug_every == 0

            if should_debug:
                self.log("#" * 100)
                self.log(f"Counter: {self.counter}")
                self.log(f"Instruction IDs: {instr_list}")
                self.log(f"Kwargs: {kwarg_list}")
                self.log(f"\nCompletion:\n{completion[0]['content']}")
                self.log("-" * 100)

            # A completion is "overlong" only if it hit the limit AND never emitted EOS
            # i.e. it was forcibly truncated, not a natural finish at exactly max length
            is_overlong = (
                len(ids) >= max_completion_length
                and not any(stop_token in ids for stop_token in self.eos_token)
            )

            if is_overlong:
                rewards.append(0.0)
                self.log(f"[overlong] length={len(ids)}, no EOS -> reward=0.0")

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

                        followed = instruction.check_following(completion[0]["content"])

                        # Protect against buggy check_following functions returning None
                        followed = bool(followed)

                        is_following_list.append(followed)

                        if should_debug:
                            self.log("Instruction:", instruction_id)
                            self.log("kwargs:", filtered_kwargs)
                            self.log("Followed:", followed)
                            self.log("Reward contribution:", int(followed))

                    except Exception as e:
                        self.log(f"Error processing instruction {instruction_id} with kwargs {kw}: {e}")
                        self.log("The corresponding completion was:")
                        self.log(completion[0]["content"]) # here do a print
                        is_following_list.append(False)

                if self.reward_mode == "binary":
                    reward = float(all(is_following_list))
                else:
                    reward = float(
                        sum(is_following_list) / len(is_following_list)
                        if is_following_list else 0.0
                    )

                rewards.append(reward)

                if should_debug:
                    self.log("is_following_list:", is_following_list)
                    self.log("Final reward:", reward)

            if should_debug:
                self.log("-" * 100)

            self.counter += 1

        return rewards


def make_if_reward_fn(
    tokenizer,
    max_completion_length,
    reward_mode,
    counter=0,
    debug_path="if_reward_binary_systemprompt.txt",
    debug_every=20480*3,
    print_to_terminal=True,
):
    eos_id = tokenizer.eos_token_id

    def if_reward_fn(completion_ids, completions, instruction_id_list, kwargs, **extra):
        nonlocal counter, print_to_terminal

        def log(*args, **print_kwargs):
            """Print to terminal and append the same message to an external file."""
            if print_to_terminal:
                print(*args, **print_kwargs)

            with open(debug_path, "a", encoding="utf-8") as f:
                print(*args, **print_kwargs, file=f)


        log("\t\tNumber of completions received:", len(completions))

        rewards = []

        for ids, completion, instr_list, kwarg_list in zip(
            completion_ids, completions, instruction_id_list, kwargs
        ):
            should_debug = counter % debug_every == 0

            if should_debug:
                log("#" * 100)
                log(f"Counter: {counter}")
                log(f"Instruction IDs: {instr_list}")
                log(f"Kwargs: {kwarg_list}")
                log(f"\nCompletion:\n{completion[0]['content']}")
                log("-" * 100)

            # A completion is "overlong" only if it hit the limit AND never emitted EOS
            # i.e. it was forcibly truncated, not a natural finish at exactly max length
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            is_overlong = (
                len(ids) >= max_completion_length
                and eos_id not in ids
            )

            if is_overlong:
                if not should_debug:
                    log("#" * 100)
                    log(f"Counter: {counter}")
                    log(f"Instruction IDs: {instr_list}")
                    log(f"Kwargs: {kwarg_list}")
                    log(f"\nCompletion:\n{completion[0]['content']}")
                    log("-" * 100)

                rewards.append(0.0)
                log(f"[overlong] length={len(ids)}, no EOS -> reward=0.0")
                

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

                        followed = instruction.check_following(completion[0]["content"])

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
                        log(completion[0]["content"]) # here do a print
                        is_following_list.append(False)

                if reward_mode == "binary":
                    reward = float(all(is_following_list))
                else:
                    reward = float(
                        sum(is_following_list) / len(is_following_list)
                        if is_following_list else 0.0
                    )

                rewards.append(reward)

                if should_debug:
                    log("is_following_list:", is_following_list)
                    log("Final reward:", reward)

            if should_debug:
                log("-" * 100)

            counter += 1

        return rewards

    return if_reward_fn



class MultiDomainRewardFn:
    """
    Reward functions for each domain in Cascade 2 multi-domain RL.
    Follows nemo-gym verification patterns.
    """
    
    def __call__(self, completions: List[str], formatted_batch: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Args:
            completions: Model outputs (one per sample in batch)
            formatted_batch: Batch of formatted data (includes domain info)
        
        Returns:
            rewards: List of reward values (0.0 to 1.0)
            info: Dict with detailed reward breakdown for logging
        """
        domain = formatted_batch['domain'][0]  # Assuming batched, take first
        
        if domain == 'mcqa':
            return self._mcqa_reward(completions, formatted_batch)
        elif domain == 'agentic':
            return self._agentic_reward(completions, formatted_batch)
        elif domain == 'structured_output':
            return self._structured_reward(completions, formatted_batch)
        else:
            raise ValueError(f"Unknown domain: {domain}")
    
    @staticmethod
    def _mcqa_reward(completions: List[str], batch: Dict[str, Any]) -> Tuple[List[float], Dict]:
        """
        MCQA Reward: Extract answer from completion and compare to expected_answer.
        Uses regex from template_metadata (from nemo-gym).
        """
        batch_size = len(completions)
        rewards = []
        extracted_answers = []
        
        for completion, expected, regex in zip(
            completions,
            batch['expected_answer'],
            batch['template_regex']
        ):
            # Extract answer using regex (e.g., "ANSWER IS ([A-D])")
            match = re.search(regex, completion, re.IGNORECASE)
            
            if match:
                predicted = match.group(1).upper()
                extracted_answers.append(predicted)
                reward = 1.0 if predicted == expected.upper() else 0.0
            else:
                # If regex fails, try fallback: extract boxed answer
                boxed_match = re.search(r'\\boxed\{([A-Z])\}', completion)
                if boxed_match:
                    predicted = boxed_match.group(1).upper()
                    extracted_answers.append(predicted)
                    reward = 1.0 if predicted == expected.upper() else 0.0
                else:
                    # Last resort: check if expected answer appears anywhere
                    extracted_answers.append("NONE")
                    reward = 0.0
            
            rewards.append(reward)
        
        return rewards, {
            'domain': 'mcqa',
            'extracted_answers': extracted_answers,
            'mean_reward': sum(rewards) / len(rewards),
        }
    
    @staticmethod
    def _agentic_reward(completions: List[str], batch: Dict[str, Any]) -> Tuple[List[float], Dict]:
        """
        Agentic Reward: Compare completion against ground_truth.
        Uses multiple similarity metrics for robustness.
        """
        from difflib import SequenceMatcher
        
        batch_size = len(completions)
        rewards = []
        similarity_scores = []
        
        for completion, ground_truth, category in zip(
            completions,
            batch['ground_truth'],
            batch.get('category', ['general'] * batch_size)
        ):
            # Normalize strings for comparison
            pred_norm = completion.lower().strip()
            truth_norm = ground_truth.lower().strip()
            
            # Compute multiple similarity metrics
            # 1. Exact match (hardest)
            exact_match = 1.0 if pred_norm == truth_norm else 0.0
            
            # 2. Sequence similarity (soft match)
            seq_sim = SequenceMatcher(None, pred_norm, truth_norm).ratio()
            
            # 3. Substring containment (lenient)
            contains_truth = 1.0 if truth_norm in pred_norm else 0.0
            
            # Combine metrics: prioritize exact match, then sequence sim
            if exact_match == 1.0:
                reward = 1.0
            else:
                # Weight: 60% sequence similarity, 40% substring
                reward = seq_sim * 0.6 + contains_truth * 0.4
            
            rewards.append(reward)
            similarity_scores.append({
                'exact_match': exact_match,
                'sequence_similarity': seq_sim,
                'substring_match': contains_truth,
                'final_reward': reward,
                'category': category,
            })
        
        return rewards, {
            'domain': 'agentic',
            'similarity_scores': similarity_scores,
            'mean_reward': sum(rewards) / len(rewards),
        }
    
    @staticmethod
    def _structured_reward(completions: List[str], batch: Dict[str, Any]) -> Tuple[List[float], Dict]:
        """
        Structured Output Reward: Validate JSON and schema compliance.
        Awards points for: valid JSON, required fields, correct types.
        """
        batch_size = len(completions)
        rewards = []
        validation_results = []
        
        for completion, schema_dict, expected_count in zip(
            completions,
            batch['schema_dict'],
            batch.get('schema_fields_count', [None] * batch_size)
        ):
            reward_components = {}
            
            # 1. Valid JSON (50% of reward)
            try:
                output = json.loads(completion)
                reward_components['valid_json'] = 0.5
            except json.JSONDecodeError:
                reward_components['valid_json'] = 0.0
                rewards.append(0.0)
                validation_results.append({**reward_components, 'error': 'invalid_json'})
                continue
            
            # 2. Required fields present (30% of reward)
            required_fields = schema_dict.get('required', [])
            fields_present = len([f for f in required_fields if f in output]) / max(len(required_fields), 1)
            reward_components['fields_present'] = min(fields_present, 1.0) * 0.3
            
            # 3. Field count matches expected (20% of reward)
            if expected_count:
                actual_count = len(output)
                count_match = 1.0 if actual_count == expected_count else max(0, 1.0 - abs(actual_count - expected_count) / expected_count)
                reward_components['field_count'] = count_match * 0.2
            else:
                reward_components['field_count'] = 0.2  # Give full credit if count not specified
            
            total_reward = sum(reward_components.values())
            rewards.append(total_reward)
            validation_results.append(reward_components)
        
        return rewards, {
            'domain': 'structured_output',
            'validation_results': validation_results,
            'mean_reward': sum(rewards) / len(rewards),
        }


if __name__=="__main__":
    # test overlong punishment
    
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    eos_id = tokenizer.eos_token_id
    max_completion_length = 10

    text = "While diying after his fight against the antimonitor, \
        Oliver Queen finds his soul sent back to the start of his journey. \
        Right back to Starling City hospital. First Episode of season 1. \
        This is Oliver Queen realising he has a second chance. Oliver & his \
        mom are in the car. Oliver asked his mom to take him to where he can \
        find Laurel Lance. “Hello Laurel” his voice breaking at the end. Oliver \
        knew that Laurel would be angry & demanding answer. Why had he cheated \
        on her with her sister ? Was she dead then, with her body resting at \
        the bottom of the ocean. He could not tell her she was dead. He would \
        not lye again to her. Please ensure your answer follows these guidelines: \
        There should be 4 paragraphs. Paragraphs and only paragraphs are separated \
        with each other by two new lines as if it was '\n\n' in python. Paragraph \
        3 must start with word crash and The last word of your response should be \
        the word contest"
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Simulate generation being cut at max_completion_length
    ids = ids[:max_completion_length]

    is_overlong = (
        len(ids) >= max_completion_length
        and eos_id not in ids
    )

    print("=" * 80)
    print("text:", text)
    print("ids:", ids)
    print("length:", len(ids))
    print("contains EOS:", eos_id in ids)
    print("is_overlong:", is_overlong)

    print("eos_token:", tokenizer.eos_token)
    print("eos_token_id:", eos_id)

    
    # if_reward_fn(
    #     completions=[ 
    #         [
    #             {
    #                 "content": "While diying after his fight against the antimonitor, \
    #                 Oliver Queen finds his soul sent back to the start of his journey. \
    #                 Right back to Starling City hospital. First Episode of season 1. \
    #                 This is Oliver Queen realising he has a second chance. Oliver & his \
    #                 mom are in the car. Oliver asked his mom to take him to where he can \
    #                 find Laurel Lance. “Hello Laurel” his voice breaking at the end. Oliver \
    #                 knew that Laurel would be angry & demanding answer. Why had he cheated \
    #                 on her with her sister ? Was she dead then, with her body resting at \
    #                 the bottom of the ocean. He could not tell her she was dead. He would \
    #                 not lye again to her. Please ensure your answer follows these guidelines: \
    #                 There should be 4 paragraphs. Paragraphs and only paragraphs are separated \
    #                 with each other by two new lines as if it was '\n\n' in python. Paragraph \
    #                 3 must start with word crash and The last word of your response should be \
    #                 the word contest"
    #             }
    #         ] 
    #     ], 
    #     instruction_id_list=[
    #         [ 
    #             "length_constraints:nth_paragraph_first_word",
    #             "last_word:last_word_answer" 
    #         ]
    #     ], 
    #     kwargs=[
    #         [
    #             {
    #                 "num_paragraphs": 4,
    #                 "nth_paragraph": 3,
    #                 "first_word": "crash"
    #             },
    #             {
    #                 "last_word": "contest"
    #             }
    #         ]
    #     ]
    # )