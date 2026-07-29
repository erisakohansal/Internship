from reward import transform_tool_format
import json

tools = [
    {
        "type": "function",
        "name": "company_directory_find_email_address",
        "description": "Finds all email addresses containing the given name (case-insensitive search).",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name or partial name to search for in email addresses"
                }
            },
            "required": [],
            "additionalProperties": False
        },
        "strict": False
    },
]

print(json.loads(transform_tool_format(tools)[0]))