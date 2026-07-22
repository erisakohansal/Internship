import ast
import math
import operator

from verl.tools.function_tool import function_tool

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "factorial": math.factorial,
}
_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}

_MAX_NUM_DIGITS = 4300  # matches CPython's default int-to-str conversion limit


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or abs(left) > 10**6):
            raise ValueError("Exponent too large")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("Only whitelisted functions are allowed: " + ", ".join(_FUNCS))
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        args = [_eval_node(a) for a in node.args]
        return _FUNCS[node.func.id](*args)
    if isinstance(node, ast.Name):
        if node.id not in _CONSTS:
            raise ValueError(f"Unknown identifier: {node.id}")
        return _CONSTS[node.id]
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Safely evaluate a restricted arithmetic expression.

    Only numeric literals, + - * / ** % //, parentheses, unary +/-, the
    constants pi/e, and the functions sqrt/abs/round/min/max/log/log10/exp/
    sin/cos/tan/factorial are permitted. Anything else raises ValueError.
    """
    if len(expression) > 300:
        raise ValueError("Expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from e
    result = _eval_node(tree.body)
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        raise ValueError("Result is not a finite number")
    if abs(result) >= 10 ** _MAX_NUM_DIGITS:
        raise ValueError("Result too large")
    return result


@function_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Args:
        expression: A Python-style arithmetic expression, e.g. "(3 + 4) * 5",
            "sqrt(2)", or "2**10". Supports + - * / ** % //, parentheses,
            the constants pi and e, and the functions sqrt, abs, round, min,
            max, log, log10, exp, sin, cos, tan, factorial.
    """
    try:
        result = safe_eval(expression)
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        return f"Error: {e}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)
