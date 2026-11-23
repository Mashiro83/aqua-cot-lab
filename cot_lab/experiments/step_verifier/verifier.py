from dataclasses import dataclass
from fractions import Fraction
import re

from ...reasoning import DECIMAL, evaluate, vote


CALC_LINE = re.compile(r"^\s*(?:[-*]\s+)?CALC:\s*(.*?)\s*$", re.IGNORECASE)
EQUATION = re.compile(r"(.+?)\s*(~=|≈|=)\s*(.+)")
VOTE_WEIGHTS = {"verified": 5, "unchecked": 4, "contradicted": 1}


@dataclass(frozen=True)
class Check:
    equation: str
    status: str


def verify_steps(text: str) -> tuple[Check, ...]:
    checks = []
    for line in text.splitlines():
        marker = CALC_LINE.fullmatch(line)
        if not marker:
            continue
        equation = marker[1]
        match = EQUATION.fullmatch(equation)
        status = "unsupported"
        if match:
            left, operation, right = match.groups()
            right = right.strip()
            try:
                difference = abs(evaluate(left) - evaluate(right))
                if operation == "=":
                    tolerance = Fraction(0)
                elif DECIMAL.fullmatch(right):
                    decimal_places = len(right.partition(".")[2])
                    tolerance = Fraction(1, 2 * 10**decimal_places)
                else:
                    raise ValueError("expected a decimal approximation")
                status = "pass" if difference <= tolerance else "fail"
            except ZeroDivisionError:
                status = "fail"
            except (SyntaxError, ValueError):
                pass
        checks.append(Check(equation, status))
    return tuple(checks)


def evidence(checks: tuple[Check, ...]) -> str:
    statuses = {check.status for check in checks}
    if "fail" in statuses:
        return "contradicted"
    if "pass" in statuses:
        return "verified"
    return "unchecked"


def weighted_vote(paths: list[dict]) -> tuple[str | None, dict[str, int]]:
    return vote(paths, weight=lambda path: VOTE_WEIGHTS[path["evidence"]])


def can_stop(paths: list[dict], minimum: int) -> bool:
    if len(paths) < minimum:
        return False
    first_answer = paths[0]["answer"]
    return (
        first_answer is not None
        and all(path["answer"] == first_answer and path["evidence"] != "contradicted" for path in paths)
        and any(path["evidence"] == "verified" for path in paths)
    )
