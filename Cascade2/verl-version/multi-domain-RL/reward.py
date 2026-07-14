from openapi_schema_validator import validate as validate_against_schema_openapi
# pip install --no-deps openapi-schema-validator 
       

"""
apparently there is what's called an in memory sandbox necessary here, look into it
"""

def multi_domain_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    solution_str : decoded model response
    extra_info : dataset metadata
    the reward manager detokenizes the response before calling the scoring function.
    """
    reward_type = extra_info["agent_ref"]
    # This is a placeholder for multi-domain reward logic.
    # Implement domain-specific reward calculations here.
    # For now, we return a default reward of 0.0.
    match reward_type:
        case "workplace_assistant_simple_agent":
            return tool_call_reward_fn(data_source, solution_str, ground_truth, extra_info)
        case "mcqa_simple_agent":
            return mcqa_reward_fn(data_source, solution_str, ground_truth, extra_info)
        case "structured_outputs_simple_agent":
            return structured_reward_fn(data_source, solution_str, ground_truth, extra_info)


# MCQA -------------------------------------------------------------------------------------
# source : https://github.com/NVIDIA-NeMo/Gym/blob/50af84a5e2a7142c7d496dd9ea76b1e9d64202bd/resources_servers/mcqa/app.py

def _get_allowed_letters_from_options(
    options: Optional[list[dict[str, str]]],
) -> set[str]:
    """Collect uppercase option letters from list of single-key dicts."""
    letters: set[str] = set()
    if options:
        for entry in options:
            # Exclude null values
            for k, v in entry.items():
                if isinstance(k, str) and len(k) == 1 and k.isalpha() and v is not None:
                    letters.add(k.upper())
    return letters

def _parse_answer_with_custom_regex(
    text: str, regex_pattern: str, allowed_letters: set[str], options: Optional[list[dict[str, str]]]
) -> Optional[str]:
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
        captured = _normalize_extracted_answer(matches[-1].strip()).upper()

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
        normalized_captured = _normalize_for_match(captured)
        for entry in options or []:
            for k, v in entry.items():
                if v is not None and k.upper() in allowed_letters and _normalize_for_match(v) == normalized_captured:
                    return k.upper()

        return None
    except re.error:
        # Invalid regex pattern, return None
        return None

def _parse_answer_with_custom_regexes(
    text: str, regex_patterns: str | list[str], allowed_letters: set[str], options: Optional[list[dict[str, str]]]
) -> Optional[str]:
    if isinstance(regex_patterns, str):
        return _parse_answer_with_custom_regex(text, regex_patterns, allowed_letters, options)

    for regex_pattern in regex_patterns:
        pred = _parse_answer_with_custom_regex(text, regex_pattern, allowed_letters, options)
        if pred is not None:
            return pred
    return None


def _normalize_extracted_answer(text: str) -> str:
    return (
        text.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )

def _normalize_for_match(s: str) -> str:
    """Lowercase and collapse whitespace for robust substring/equality checks."""
    return " ".join(s.lower().split())


def _parse_answer_letter_strict_boxed(text: str, allowed_letters: set[str]) -> tuple[Optional[str], str, bool]:
    # Strict boxed: capture a single UPPERCASE letter, allowing non-letter chars around it inside the box
    STRICT_BOXED_PATTERN = re.compile(r"\\boxed\{\s*[^A-Za-z]*([A-Z])[^A-Za-z]*\s*\}")
    parsed_text = text
    m = STRICT_BOXED_PATTERN.search(text)
    if not m:
        return None, parsed_text, True
    letter = m.group(1).upper()
    if letter not in allowed_letters:
        return None, parsed_text, True
    return letter, parsed_text, False


def _match_option_text(text: str, options: list[dict[str, str]], allowed_letters: set[str]) -> Optional[str]:
    """Match boxed content against option texts and return the option letter.

    - Looks ONLY inside the first \boxed{...} region; returns None if absent.
    - Normalizes (lowercase, collapse whitespace) both boxed content and option texts.
    - Treats a match as substring containment of an option's text in the boxed content.
    - Returns the option letter only if EXACTLY ONE option matches; otherwise returns None.
    """
    # Only match within boxed content; if no boxed content, return None
    BOXED_CONTENT_PATTERN = re.compile(r"\\boxed\{\s*(.*?)\s*\}", re.S)
    boxed = BOXED_CONTENT_PATTERN.search(text)
    if not boxed:
        return None
    inner = boxed.group(1)
    candidate_texts = [inner, _strip_latex_wrappers(inner)]
    normalized_candidates = [_normalize_for_match(t) for t in candidate_texts]

    # Build list of (letter, normalized_option_text)
    normalized_options: list[tuple[str, str]] = []
    for entry in options or []:
        for k, v in entry.items():
            # Skip null values and only include valid letter keys with string values
            if v is not None and isinstance(k, str) and len(k) == 1 and k.upper() in allowed_letters:
                normalized_options.append((k.upper(), _normalize_for_match(v)))

    matched_letters: set[str] = set()
    for cand in normalized_candidates:
        for letter, opt_norm in normalized_options:
            if opt_norm and opt_norm in cand:
                matched_letters.add(letter)
    if len(matched_letters) == 1:
        return next(iter(matched_letters))
    return None


def _strip_latex_wrappers(s: str) -> str:
    """Remove successive \\text{...} wrappers from a LaTeX string."""
    LATEX_TEXT_WRAP_PATTERN = re.compile(r"\\text\{\s*(.*?)\s*\}", re.S)
    while True:
        m = LATEX_TEXT_WRAP_PATTERN.fullmatch(s)
        if not m:
            break
        s = m.group(1)
    return s

    
def mcqa_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    solution_str : decoded model response
    extra_info : dataset metadata
    the reward manager detokenizes the response before calling the scoring function.
    there are 4 different grading modes for mcqa: 
    - strict_single_letter_boxed  -> accepts \boxed{A}
    - lenient_boxed               -> accepts \boxed{A} or boxed option text
    - lenient_answer_colon        -> accepts Answer: A or Answer: option text
    - lenient_answer_colon_md     -> accepts markdown-ish **Answer: A**
    """
    
    # Pull options/expected_answer from dataset-style metadata if available
    options, expected_answer = options = extra_info["options"], ground_truth
    gold = (expected_answer or "").strip().upper()
    # Derive allowed letters from option keys
    allowed_letters = _get_allowed_letters_from_options(options)

    grading_mode = extra_info["reward_mode"]

    pred: Optional[str] = None

    if not solution_str or not solution_str.strip():
        return 0.0  # Empty response gets zero reward

    # Check for template_metadata first (highest priority)
    template_metadata = extra_info["template_metadata"]
    if template_metadata and "output_regex" in template_metadata:
        regex_patterns = template_metadata["output_regex"]
        pred = _parse_answer_with_custom_regexes(solution_str, regex_patterns, allowed_letters, options)

    # Fallback to existing grading_mode logic if template_metadata didn't work
    if pred is None:
        if grading_mode == "strict_single_letter_boxed":
            pred, _, _ = _parse_answer_letter_strict_boxed(solution_str, allowed_letters)
        elif grading_mode == "lenient_boxed":
            # Try strict boxed first
            pred, _, _ = _parse_answer_letter_strict_boxed(solution_str, allowed_letters)
            if pred is None:
                # Then try to match option text inside boxed content only
                letter_from_text = _match_option_text(solution_str, options, allowed_letters)
                if letter_from_text is not None:
                    pred = letter_from_text
        elif grading_mode == "lenient_answer_colon":
            # Look for Answer: <...>
            ANSWER_COLON_PATTERN = re.compile(r"(?i)answer\s*:\s*(.+)")
            m = ANSWER_COLON_PATTERN.search(solution_str)
            if m:
                candidate = _strip_latex_wrappers(m.group(1)).strip()
                # Letter case
                if len(candidate) == 1 and candidate.isalpha():
                    letter_up = candidate.upper()
                    if letter_up in allowed_letters:
                        pred = letter_up
                # Option text equality (normalized)
                if pred is None:
                    cand_norm = _normalize_for_match(candidate)
                    for entry in options or []:
                        for k, v in entry.items():
                            k_up = k.upper()
                            if k_up in allowed_letters and _normalize_for_match(v) == cand_norm:
                                pred = k_up
                                break
                        if pred is not None:
                            break
        elif grading_mode == "lenient_answer_colon_md":
            # Markdown-aware Answer: extraction handles **Answer: B**, etc.
            # Markdown-aware variant: tolerates **Answer: B**, __Answer__: B, etc. Captures single letter only.
            ANSWER_COLON_MD_PATTERN = re.compile(r"(?i)[*_]{0,2}Answer[*_]{0,2}\s*:[*_\s]{0,2}\s*([A-Z])(?![a-zA-Z0-9])")
            md_match = ANSWER_COLON_MD_PATTERN.search(solution_str)
            if md_match:
                letter_up = md_match.group(1).strip().upper()
                if letter_up in allowed_letters:
                    pred = letter_up

    is_correct = (pred == gold) if (pred is not None and gold) else False
    reward = 1.0 if is_correct else 0.0

    assert type(reward) is float
    return reward


# STRUCTURED OUTPUTS -------------------------------------------------------------------

def strictify_schema(schema: Dict[str, Any]):
    """Make a schema strict as per OpenAPI guidelines"""
    if isinstance(schema, Dict):
        if "properties" in schema:
            schema["required"] = list(schema["properties"])
            schema["additionalProperties"] = False
        for k, v in schema.items():
            strictify_schema(v)


def structured_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    solution_str : decoded model response
    extra_info : dataset metadata
    the reward manager detokenizes the response before calling the scoring function.
    in this nemotron dataset there are only json schema types
    """
    schema_str = extra_info["schema_str"]
    # strict schemas and schemaless? 

    # only the evaluate_structured_output_response is useful in this case : https://github.com/NVIDIA-NeMo/Gym/blob/50af84a5e2a7142c7d496dd9ea76b1e9d64202bd/resources_servers/structured_outputs/app.py#L434
    """Returns (reward, error_type, error_message)."""
        if not solution_str or not solution_str.strip():
            return 0.0, "empty_response", "No assistant response text"

        try:
            schema = json.loads(schema_str)
        except Exception as e:
            return 0.0 # "schema_error", str(e)[:200]

        strictify_schema(schema)

        try:
            parsed = json.loads(solution_str)
        except Exception as e:
            return 0.0 # "parse_error", f"{type(e).__name__}: {str(e)[:200]}"

        try:

            validate_against_schema_openapi(parsed, schema)
            return 1.0 # None, None
        except Exception as e:
            return 0.0 # "validation_error", f"{type(e).__name__}: {str(e)[:200]}"


# TOOL CALLING ---------------------------------------------------------------------------

def try_parse_tool_calls(content: str):
    """Try parse the tool calls.
    source : https://colab.research.google.com/github/oliveirabruno01/unsloth-challenge/blob/main/Qwen2_5_1_5B_Tool_Calling.ipynb#scrollTo=AsF3E3RTes8w
    """
    tool_calls = []
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)): 
        # better version? 
        # re.finditer(
        #     r"<tool_call>\s*(.*?)\s*</tool_call>",
        #     content,
        #     re.DOTALL,
        # )
        try:
            func = json.loads(m.group(1).strip())
            if isinstance(func["arguments"], str): # sometimes the model may output a string containing json
                func["arguments"] = json.loads(func["arguments"])
                """
                Small subtlety: it appends func to tool_calls before modifying func["arguments"], but since dictionaries are stored by reference, the object inside tool_calls is also updated. So this still works.
                """
            tool_calls.append(func)
        except json.JSONDecodeError as e:
            print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
            pass # pass? no reward 0? just append []?
    return tool_calls

def execute_actions_and_reset_state(actions: List[Dict[str, str]]):
    toolkits = [
        "email",
        "calendar",
        "analytics",
        "project_management",
        "customer_relationship_manager",
    ]
    tool_env = get_tools(toolkits)

    # Execute the actions
    for action in actions:
        try:
            tool_env["functions"][action["name"]](**json.loads(action["arguments"]))
        except Exception as e:
            print("Error executing tool: ", e)
            continue
    return tool_env

def get_tools(toolkits):
    tool_env = {
        "containers": {},
        "functions": {},
        "schemas": [],
    }
    company_directory = CompanyDirectoryTool()
    tool_env["containers"]["company_directory"] = company_directory
    tool_env["functions"]["company_directory_find_email_address"] = company_directory.find_email_address
    tool_env["schemas"].extend(company_directory_tool_schemas)
    if "email" in toolkits:
        email = EmailTool()
        tool_env["containers"]["email"] = email
        tool_env["functions"]["email_get_email_information_by_id"] = email.get_email_information_by_id
        tool_env["functions"]["email_search_emails"] = email.search_emails
        tool_env["functions"]["email_send_email"] = email.send_email
        tool_env["functions"]["email_delete_email"] = email.delete_email
        tool_env["functions"]["email_forward_email"] = email.forward_email
        tool_env["functions"]["email_reply_email"] = email.reply_email
        tool_env["schemas"].extend(email_tool_schemas)
    if "calendar" in toolkits:
        calendar = CalendarTool()
        tool_env["containers"]["calendar"] = calendar
        tool_env["functions"]["calendar_get_event_information_by_id"] = calendar.get_event_information_by_id
        tool_env["functions"]["calendar_search_events"] = calendar.search_events
        tool_env["functions"]["calendar_create_event"] = calendar.create_event
        tool_env["functions"]["calendar_delete_event"] = calendar.delete_event
        tool_env["functions"]["calendar_update_event"] = calendar.update_event
        tool_env["schemas"].extend(calendar_tool_schemas)
    if "analytics" in toolkits:
        analytics = AnalyticsTool()
        tool_env["containers"]["analytics"] = analytics
        tool_env["functions"]["analytics_engaged_users_count"] = analytics.engaged_users_count
        tool_env["functions"]["analytics_get_visitor_information_by_id"] = analytics.get_visitor_information_by_id
        tool_env["functions"]["analytics_create_plot"] = analytics.create_plot
        tool_env["functions"]["analytics_traffic_source_count"] = analytics.traffic_source_count
        tool_env["functions"]["analytics_total_visits_count"] = analytics.total_visits_count
        tool_env["functions"]["analytics_get_average_session_duration"] = analytics.get_average_session_duration
        tool_env["schemas"].extend(analytics_tool_schemas)
    if "project_management" in toolkits:
        project_management = ProjectManagementTool()
        tool_env["containers"]["project_management"] = project_management
        tool_env["functions"]["project_management_get_task_information_by_id"] = (
            project_management.get_task_information_by_id
        )
        tool_env["functions"]["project_management_search_tasks"] = project_management.search_tasks
        tool_env["functions"]["project_management_create_task"] = project_management.create_task
        tool_env["functions"]["project_management_delete_task"] = project_management.delete_task
        tool_env["functions"]["project_management_update_task"] = project_management.update_task
        tool_env["schemas"].extend(project_management_tool_schemas)
    if "customer_relationship_manager" in toolkits:
        customer_relationship_manager = CustomerRelationshipManagerTool()
        tool_env["containers"]["customer_relationship_manager"] = customer_relationship_manager
        tool_env["functions"]["customer_relationship_manager_search_customers"] = (
            customer_relationship_manager.search_customers
        )
        tool_env["functions"]["customer_relationship_manager_update_customer"] = (
            customer_relationship_manager.update_customer
        )
        tool_env["functions"]["customer_relationship_manager_add_customer"] = (
            customer_relationship_manager.add_customer
        )
        tool_env["functions"]["customer_relationship_manager_delete_customer"] = (
            customer_relationship_manager.delete_customer
        )
        tool_env["schemas"].extend(customer_relationship_manager_tool_schemas)
    return tool_env

def tool_call_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """
    TODO:
    VERL rollout produces solution_str DONE
        ↓
    your parser extracts <tool_call>...</tool_call> DONE
        ↓
    your reward function creates two fresh in-memory workbench envs
        ↓
    execute predicted calls in env A
    execute ground-truth calls in env B
        ↓
    compare states
        ↓
    return 1.0 or 0.0
    """

    # should extract whats inside <tool_call> tags as the provided answer and compare to the ground_truth
    tools_calls = try_parse_tool_calls(solution_str)
    predict_env = execute_actions_and_reset_state(tool_calls)
    ground_truth_env = execute_actions_and_reset_state(ground_truth)

    def convert_strs_to_lowercase(df):
        # For some fields the case matters, so we don't convert them to lowercase
        fields_not_to_convert = ["status", "list_name", "board"]
        for col in df.columns:
            if col not in fields_not_to_convert:
                df[col] = df[col].str.lower()
        return df

    # We allow for case-insensitive comparison of strings for most fields
    predicted_calendar_state = convert_strs_to_lowercase(predict_env["containers"]["calendar"]._calendar_events)
    predicted_email_state = convert_strs_to_lowercase(predict_env["containers"]["email"]._emails)
    predicted_analytics_state = convert_strs_to_lowercase(predict_env["containers"]["analytics"]._plots_data)
    predicted_project_management_state = convert_strs_to_lowercase(
        predict_env["containers"]["project_management"]._project_tasks
    )
    predicted_customer_relationship_manager_state = convert_strs_to_lowercase(
        predict_env["containers"]["customer_relationship_manager"]._crm_data
    )

    ground_truth_calendar_state = convert_strs_to_lowercase(
        ground_truth_env["containers"]["calendar"]._calendar_events
    )
    ground_truth_email_state = convert_strs_to_lowercase(ground_truth_env["containers"]["email"]._emails)
    ground_truth_analytics_state = convert_strs_to_lowercase(ground_truth_env["containers"]["analytics"]._plots_data)
    ground_truth_project_management_state = convert_strs_to_lowercase(
        ground_truth_env["containers"]["project_management"]._project_tasks
    )
    ground_truth_customer_relationship_manager_state = convert_strs_to_lowercase(
        ground_truth_env["containers"]["customer_relationship_manager"]._crm_data
    )

    return (
        predicted_calendar_state.equals(ground_truth_calendar_state)
        and predicted_email_state.equals(ground_truth_email_state)
        and predicted_analytics_state.equals(ground_truth_analytics_state)
        and predicted_project_management_state.equals(ground_truth_project_management_state)
        and predicted_customer_relationship_manager_state.equals(ground_truth_customer_relationship_manager_state)
    )
