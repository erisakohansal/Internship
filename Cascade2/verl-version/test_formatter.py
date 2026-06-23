"""
Test script for FormatData class — validates tool integration and formatting logic.
"""

import json
from dataset import FormatData

# Mock tool data
mock_tools = [
    {
        "function": "get_weather",
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    },
    {
        "function": "search_web",
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]

# Mock workplace assistant data
mock_workplace_data = {
    'agent_ref': {'name': 'workplace_assistant_simple_agent'},
    'category': 'tool_calling',
    'ground_truth': None,
    'responses_create_params': {
        'input': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is the weather in Paris?'}
        ],
        'tools': mock_tools
    }
}

# Mock mcqa data (no tools)
mock_mcqa_data = {
    'agent_ref': {'name': 'mcqa_simple_agent'},
    'category': 'instruction_following',
    'ground_truth': None,
    'responses_create_params': {
        'input': [
            {'role': 'user', 'content': 'Which is the capital of France?\nA) London\nB) Paris\nC) Berlin'}
        ]
    },
    'instruction_id_list': ['id_123'],
    'kwargs': {}
}

def test_transform_tool_format():
    print("=" * 60)
    print("TEST 1: Transform tool format")
    print("=" * 60)
    
    transformed = FormatData.transform_tool_format(mock_tools)
    print(f"Input tools count: {len(mock_tools)}")
    print(f"Transformed tools count: {len(transformed)}")
    print(f"First transformed tool:\n{json.dumps(transformed[0], indent=2)}")
    assert len(transformed) == len(mock_tools), "Tool count mismatch"
    print("✅ PASSED\n")


def test_build_tool_prompt():
    print("=" * 60)
    print("TEST 2: Build tool prompt")
    print("=" * 60)
    
    prompt = FormatData.build_tool_prompt(mock_tools)
    print(f"Tool prompt length: {len(prompt)} chars")
    assert "# Tools" in prompt, "Missing '# Tools' header"
    assert "<tools>" in prompt, "Missing <tools> XML tag"
    assert "get_weather" in prompt, "Missing tool name in prompt"
    assert "<tool_call>" in prompt, "Missing <tool_call> instruction"
    print("Tool prompt preview:")
    print(prompt[:300] + "...")
    print("✅ PASSED\n")


def test_workplace_assistant_data():
    print("=" * 60)
    print("TEST 3: Workplace assistant data formatting")
    print("=" * 60)
    
    formatted = FormatData.workplace_assistant_data(mock_workplace_data, idx=0)
    
    print(f"Data source: {formatted['data_source']}")
    print(f"Ability: {formatted['ability']}")
    print(f"Prompt type: {type(formatted['prompt'])}")
    print(f"Prompt length: {len(formatted['prompt'])}")
    
    # Validate structure
    assert isinstance(formatted['prompt'], list), "Prompt should be a list"
    assert len(formatted['prompt']) == 2, "Should have system and user messages"
    assert formatted['prompt'][0]['role'] == 'system', "First message should be system"
    assert "# Tools" in formatted['prompt'][0]['content'], "System prompt should include tools"
    assert formatted['prompt'][1]['role'] == 'user', "Second message should be user"
    
    print("System prompt (first 400 chars):")
    print(formatted['prompt'][0]['content'][:400] + "...")
    print("\nUser prompt:")
    print(formatted['prompt'][1]['content'])
    print("✅ PASSED\n")


def test_mcqa_data():
    print("=" * 60)
    print("TEST 4: MCQA data formatting (no tools)")
    print("=" * 60)
    
    formatted = FormatData.mcqa_data(mock_mcqa_data, idx=1)
    
    print(f"Data source: {formatted['data_source']}")
    print(f"Ability: {formatted['ability']}")
    print(f"Prompt type: {type(formatted['prompt'])}")
    
    # Validate structure
    assert isinstance(formatted['prompt'], list), "Prompt should be a list"
    assert formatted['prompt'][0]['role'] == 'system', "First message should be system"
    assert "not allowed to use any tools" in formatted['prompt'][0]['content'], "IF system prompt should be present"
    
    print("System prompt:")
    print(formatted['prompt'][0]['content'])
    print("✅ PASSED\n")


def test_format_data_multi_domain():
    print("=" * 60)
    print("TEST 5: Multi-domain formatter (dispatch logic)")
    print("=" * 60)
    
    # Test workplace
    workplace_result = FormatData.format_data_multi_domain(mock_workplace_data, idx=0)
    assert "# Tools" in workplace_result['prompt'][0]['content'], "Should format as workplace with tools"
    print("✅ Workplace assistant formatted correctly")
    
    # Test mcqa
    mcqa_result = FormatData.format_data_multi_domain(mock_mcqa_data, idx=1)
    assert "not allowed to use any tools" in mcqa_result['prompt'][0]['content'], "Should format as mcqa without tools"
    print("✅ MCQA formatted correctly")
    print("✅ PASSED\n")


def main():
    print("\n🧪 Running FormatData unit tests...\n")
    
    try:
        test_transform_tool_format()
        test_build_tool_prompt()
        test_workplace_assistant_data()
        test_mcqa_data()
        test_format_data_multi_domain()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
