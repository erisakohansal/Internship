from datasets import load_dataset, Dataset
import re
import numpy as np
from collections import Counter, defaultdict
import json
from typing import Dict, Any, List, Optional


class Format:

    def __init__(self, stage="IF-RL"):

        self.stage = stage
        self.system_prompt = """
        You are a helpful and harmless assistant.
        You are not allowed to use any tools.
        """ 
        # => filter out non supported languages
        data = load_dataset(
            "nvidia/Nemotron-Cascade-2-RL-data",  #"nvidia/Nemotron-RL-instruction_following",
            stage,
            split="train",
        )
        print(data[0].keys())

        match stage:
            case "IF-RL": 
                self.dataset = data.map(self.data_if, remove_columns=data.column_names, load_from_cache_file=False)
            case "multi-domain-RL": 
                self.dataset = data.map(self.data_multi_domain, remove_columns=data.column_names, load_from_cache_file=False)
    

    def data_if(self, data):
            
        res = {
            'prompt' : [
                {'role': 'system', 'content': self.system_prompt},
                data['responses_create_params']['input'][0],
            ],
            'instruction_id_list': data['instruction_id_list'],
            'kwargs': data['kwargs'],
        }
        return res
    

    def _format_mcqa(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format MCQA-specific data"""
        # Validate required fields
        assert "output_regex" in data.get('template_metadata', {}) and \
            data['template_metadata']['output_regex'] is not None, \
            "MCQA: output_regex missing in template_metadata"
        
        # Validate options format
        options = data.get('options', [])
        assert isinstance(options, list) and len(options) > 0, "MCQA: options must be non-empty list"
        for entry in options:
            assert isinstance(entry, dict) and len(entry) == 1, \
                f"MCQA: each option must be dict with single key-value pair"
            key, val = list(entry.items())[0]
            assert isinstance(key, str) and len(key) == 1 and key.isalpha() and key.isupper(), \
                f"MCQA: option key '{key}' must be single uppercase letter"
            assert isinstance(val, str) and val.strip() != "", \
                f"MCQA: option value for key '{key}' must be non-empty string"
            

        expected_answer = data.get('expected_answer', '').strip()
        assert expected_answer in [list(opt.keys())[0] for opt in options], \
            f"MCQA: expected_answer '{expected_answer}' not in options"
        
        return {
            'domain': 'mcqa',
            'agent_type': 'mcqa_simple_agent',
            'prompt': [
                {'role': 'system', 'content': self.system_prompt},
                data['responses_create_params']['input'][0],
            ],
            'options': options,
            'expected_answer': expected_answer,
            'template_regex': data['template_metadata']['output_regex'],
            'template_id': data['template_metadata'].get('template_id', None),
            'uuid': data.get('uuid'),
        }
    
    def _format_agentic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format agentic task data"""
        ground_truth = data.get('ground_truth')
        category = data.get('category', '').strip()
        
        assert ground_truth is not None and ground_truth.strip() != "", \
            "AGENTIC: ground_truth cannot be empty"
        assert category != "", "AGENTIC: category cannot be empty"
        
        return {
            'domain': 'agentic',
            'agent_type': 'workplace_assistant_agent',
            'prompt': [
                {'role': 'system', 'content': self.system_prompt},
                data['responses_create_params']['input'][0],
            ],
            'ground_truth': ground_truth,
            'category': category,
            'uuid': data.get('uuid'),
        }

    def _format_structured_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format structured output task data"""
        schema_str = data.get('schema_str', '').strip()
        schema_fields_count = data.get('schema_fields_count', '').strip()
        
        assert schema_str != "", "STRUCTURED: schema_str cannot be empty"
        assert schema_fields_count != "", "STRUCTURED: schema_fields_count cannot be empty"
        
        # Validate JSON schema
        try:
            schema_dict = json.loads(schema_str)
        except json.JSONDecodeError:
            raise ValueError("STRUCTURED: schema_str is not valid JSON")
        
        return {
            'domain': 'structured_output',
            'agent_type': 'structured_outputs_agent',
            'prompt': [
                {'role': 'system', 'content': self.system_prompt},
                data['responses_create_params']['input'][0],
            ],
            'schema_str': schema_str,
            'schema_dict': schema_dict,  # Parsed for easier access
            'schema_fields_count': int(schema_fields_count),
            'uuid': data.get('uuid'),
        }
    
    
    def data_multi_domain(self, data): # specify the regex pattern here with hydra

        agent_name = data.get('agent_ref', {}).get('name', '').strip()
        # ============ DOMAIN 1: MCQA ============
        if agent_name == 'mcqa_simple_agent':
            return self._format_mcqa(data)
        
        # ============ DOMAIN 2: AGENTIC ============
        elif agent_name == 'workplace_assistant_agent':
            return self._format_agentic(data)
        
        # ============ DOMAIN 3: STRUCTURED OUTPUT ============
        elif agent_name == 'structured_outputs_agent':
            return self._format_structured_output(data)
        
        else:
            raise ValueError(f"Unknown agent type: {agent_name}")
