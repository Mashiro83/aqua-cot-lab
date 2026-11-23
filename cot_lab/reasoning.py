import ast
from collections import Counter
from collections.abc import Callable
from fractions import Fraction
import operator
import re


FINAL_LINE = re.compile(r"FINAL:\s*([A-E])\.?\s*", re.IGNORECASE)
DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
EXPRESSION = re.compile(r"[0-9.()+*/\- \t]+")
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def final_answer(text: str, finish_reason: str | None) -> str | None:
    lines = text.strip().splitlines()
    if not lines or finish_reason in {"length", "content_filter", "tool_calls"}:
        return None
    match = FINAL_LINE.fullmatch(lines[-1].strip())
    if match and sum(bool(FINAL_LINE.fullmatch(line.strip())) for line in lines) == 1:
        return match[1].upper()
    return None


def evaluate(expression: str) -> Fraction:
    """Interpret a small arithmetic grammar with exact fractions."""
    expression = expression.strip()
    if len(expression) > 160 or not EXPRESSION.fullmatch(expression):
        raise ValueError("expression outside supported arithmetic grammar")

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant):
            literal = ast.get_source_segment(expression, node) or ""
            if not DECIMAL.fullmatch(literal):
                raise ValueError("unsupported literal")
            value = Fraction(literal)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            operand = visit(node.operand)
            value = -operand if isinstance(node.op, ast.USub) else operand
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right) <= 12:
                value = left ** right.numerator
            elif type(node.op) in OPERATORS:
                value = OPERATORS[type(node.op)](left, right)
            else:
                raise ValueError("unsupported operator")
        else:
            raise ValueError("unsupported syntax")
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 512:
            raise ValueError("intermediate value too large")
        return value

    return visit(ast.parse(expression, mode="eval").body)


def vote(paths: list[dict], *, weight: Callable[[dict], int] | None = None) -> tuple[str | None, dict[str, int]]:
    scores = Counter()
    for path in paths:
        if path["answer"] is not None:
            scores[path["answer"]] += weight(path) if weight is not None else 1
    # Counter preserves insertion order, so ties favor the earliest sampled answer.
    return max(scores, key=scores.get, default=None), dict(scores)
