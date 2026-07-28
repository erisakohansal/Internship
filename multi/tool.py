import ast
import operator as op

from verl.tools.function_tool import function_tool


SYSTEM = (
    "You are a math expert. Use the `calculator` tool when arithmetic is "
    "required. When done, put the final answer inside \\boxed{}."
)


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
    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so reject it explicitly.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are supported")
        return node.value

    if isinstance(node, ast.BinOp):
        operation = _OPS.get(type(node.op))
        if operation is None:
            raise ValueError("unsupported binary operator")
        return operation(_eval(node.left), _eval(node.right))

    if isinstance(node, ast.UnaryOp):
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
        return str(_eval(tree.body))
    except Exception as error:
        # This observation is returned to the model.
        return f"error: {error}"