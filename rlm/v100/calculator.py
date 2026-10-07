"""Bounded decimal arithmetic; assumptions and market forecasts are not validated."""

import ast
import re
from decimal import Decimal, DecimalException, localcontext


def calculate(expression: str) -> dict:
    if not isinstance(expression, str) or not 1 <= len(expression) <= 512:
        raise ValueError("Expression must have 1-512 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid arithmetic syntax") from error
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression is too complex")

    def visit(node, depth=0):
        if depth > 24:
            raise ValueError("Expression nesting exceeds budget")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            raw = ast.get_source_segment(expression, node)
            if (
                raw is None
                or len(raw) > 64
                or not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,2})?", raw)
            ):
                raise ValueError("Use bounded decimal literals")
            value = Decimal(raw)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            else:
                value = left / right
        else:
            raise ValueError("Only decimal + - * / and parentheses are supported")
        if (
            not value.is_finite()
            or abs(value) > Decimal("1e30")
            or (value and value.adjusted() < -40)
        ):
            raise ValueError("Numeric result exceeds budget")
        return value

    try:
        with localcontext() as context:
            context.prec = 40
            value = visit(tree.body)
            return {
                "result": format(value, "f"),
                "precision": 40,
                "scope": "Explicit arithmetic; assumptions are not certified",
            }
    except DecimalException as error:
        raise ValueError("Invalid decimal arithmetic") from error
