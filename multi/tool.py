import ast
import operator as op
import json
from typing import Any
import asyncio

from verl.tools.function_tool import function_tool
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse, OpenAIFunctionToolSchema


CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '48 + 48/2'.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": False,
    },
}


_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval(node):
    """
    This receives one node from the parsed syntax tree and evaluates it recursively.
    """
    if isinstance(node, ast.Constant):
        """
        This handles literal values such as 42, 3.14, or -5’s inner 5.
        """
        # bool is a subclass of int, so reject it explicitly.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are supported")
        return node.value

    if isinstance(node, ast.BinOp):
        """
        This handles operations involving a left and right operand, such as:
        2 + 3
        8 / 4
        5 ** 2
        """
        operation = _OPS.get(type(node.op))
        if operation is None:
            raise ValueError("unsupported binary operator")
        return operation(_eval(node.left), _eval(node.right))

    if isinstance(node, ast.UnaryOp):
        """
        This handles a sign applied to one value:
        -5
        +5
        """
        operation = _OPS.get(type(node.op))
        if operation is None:
            raise ValueError("unsupported unary operator")
        return operation(_eval(node.operand))

    raise ValueError("unsupported expression")


@function_tool(schema=CALCULATOR)
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression.

    Args:
        expression: Arithmetic expression to evaluate.

    Returns:
        The result or an error message.
    """
    try:
        if len(expression) > 200:
            raise ValueError("expression is too long")

        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body)) # tree.body is the actual mathematical expression inside the outer Expression node.
    except Exception as error:
        # This observation is returned to the model.
        return f"error: {error}"


class CalculatorTool(BaseTool):

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the tool.

        Args:
            instance_id: The instance id of the tool.
            parameters: The json string of the parameters of the tool.

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing text, image, and/or video content.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.

        - We would need a per-rollout asyncio.Lock only if you allowed multiple tool calls 
        from the same assistant turn to execute concurrently against the same environment
        but with max_parallel_calls: 1, only one tool call accesses a rollout’s environment at a time
        - We would need a per-rollout asyncio.Lock only if you allowed multiple tool calls from the 
        same assistant turn to execute concurrently against the same environment
        - await asyncio.to_thread(...) waits for that tool function to finish before the rollout continues
        """
        agent_data = kwargs["agent_data"]

        # Preserve the action history for episodic verification.
        agent_data.extra_fields.setdefault(
            "predicted_actions",
            [],
        ).append(
            {
                "name": self.name,
                "arguments": parameters, 
            }
        )
        try:
            result = await asyncio.to_thread(
                calculator,
                **parameters,
            )

            observation = (
                result
                if isinstance(result, str) 
                else json.dumps(
                    result,
                    ensure_ascii=False, 
                    default=str,
                )
            )

        except Exception as e:
            print(f"Error executing tool '{self.name}': {e}")
            observation = f"Error executing tool '{self.name}': {e}"

        return ToolResponse(text=observation), 0.0, {} # model observation, step-level tool rewards, tool metrics
