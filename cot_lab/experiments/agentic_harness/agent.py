from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
import re

from ...data import LABELS, Question
from ...inference import generate_response, prediction
from ...knowledge import KnowledgeBase, render_hits
from ...model import FORMAT_EXAMPLE, ModelClient, question_text
from ...reasoning import evaluate


ACTION_LINE = re.compile(r"(CALCULATE|LOOKUP|FINAL):\s*(.+)", re.IGNORECASE)
AGENT_PROMPT = (
    "Solve a multiple-choice math question using tools. Briefly reason about the next step, "
    "then put exactly one action on the last line. Use CALCULATE: arithmetic to get an exact "
    "result (only +, -, *, /, ** and parentheses), or LOOKUP: keywords for a mathematical rule. "
    "Wait for the observation. Use a successful tool observation and then submit FINAL: A, "
    "replacing A with the chosen letter. Use a separate turn for each action. Plain text only."
)


@dataclass(frozen=True)
class Action:
    name: str
    argument: str

    @property
    def identity(self) -> tuple[str, str]:
        argument = self.argument.casefold().split()
        normalized = "".join(argument) if self.name == "calculate" else " ".join(argument)
        return self.name, normalized


def parse_action(response: dict) -> Action:
    lines = response["text"].strip().splitlines()
    if not lines or response["finish_reason"] not in (None, "stop"):
        raise ValueError("the response ended before a complete action")
    parsed = [match for line in lines if (match := ACTION_LINE.fullmatch(line.strip()))]
    if len(parsed) != 1 or not ACTION_LINE.fullmatch(lines[-1].strip()):
        raise ValueError("expected exactly one action on the last line")
    name, argument = parsed[0].groups()
    name, argument = name.casefold(), argument.strip()
    if not argument or len(argument) > 160:
        raise ValueError("action argument must contain 1 to 160 characters")
    if name == "final":
        argument = argument.upper()
        if len(argument) != 1 or argument not in LABELS:
            raise ValueError("FINAL must be followed by one letter A through E")
    return Action(name, argument)


def tool_observation(action: Action, knowledge: KnowledgeBase) -> tuple[str, list[str]]:
    if action.name == "calculate":
        try:
            result = evaluate(action.argument)
        except (ValueError, SyntaxError, ZeroDivisionError) as error:
            raise ValueError("calculator could not evaluate the expression") from error
        return f"Exact result: {result}. Decimal: {float(result):.10g}.", []
    if action.name == "lookup":
        hits = knowledge.search(action.argument, 2)
        if not hits:
            raise ValueError("lookup found no matching formula")
        return render_hits(hits), [hit["id"] for hit in hits]
    raise ValueError("unknown tool")


def opening_messages(q: Question, budget: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": FORMAT_EXAMPLE},
        {"role": "assistant", "content": "CALCULATE: 8 * 6 - 7"},
        {"role": "user", "content": "Tool observation: Exact result: 41. Decimal: 41."},
        {"role": "assistant", "content": "FINAL: A"},
        {"role": "user", "content": f"{question_text(q)}\n\nAction budget: {budget} turns."},
    ]


def run_agent(
    client: ModelClient, q: Question, config: dict, knowledge: KnowledgeBase,
    opening: list[dict[str, str]], initial: dict, *, recover: bool,
) -> tuple[dict, dict]:
    budget = config["agent_steps"]
    messages = list(opening)
    steps = []
    calls = [initial]
    seen = set()
    tool_successes = 0
    answer = None
    stop_reason = "budget_exhausted"
    for index in range(budget):
        if index:
            calls.append(generate_response(
                client, q, config, f"agent-step-{index}", 0.0, messages=messages, max_tokens=96
            ))
        response = calls[-1]
        step = {"index": index + 1, "action": None}
        if index:
            step["generation"] = response
        steps.append(step)
        try:
            action = parse_action(response)
            step["action"] = asdict(action)
            if action.name == "final":
                if not tool_successes:
                    raise ValueError("submit FINAL after a successful tool call")
                answer = action.argument
                step["status"] = "submitted"
                stop_reason = "submitted"
                break
            if recover and action.identity in seen:
                raise ValueError("duplicate tool call; use its previous result or choose a different action")
            result, hit_ids = tool_observation(action, knowledge)
            seen.add(action.identity)
            tool_successes += 1
            step.update(status="tool_ok", observation=result, hit_ids=hit_ids)
            feedback = f"Tool observation: {result}"
        except ValueError as error:
            step.update(status="rejected", observation=str(error))
            if not recover:
                stop_reason = "invalid_action"
                break
            feedback = f"Action rejected: {error}. Try again with CALCULATE: 2 + 2, LOOKUP: topic, or FINAL: A."

        if recover:
            remaining = budget - index - 1
            feedback += f" Turns remaining: {remaining}. Successful tool calls: {tool_successes}."
            if remaining == 1 and tool_successes:
                feedback += " Submit FINAL with your selected letter on the next turn."
        messages.extend([
            {"role": "assistant", "content": response["text"]},
            {"role": "user", "content": feedback},
        ])

    detail = {"steps": steps, "answer": answer, "stop_reason": stop_reason, "tool_successes": tool_successes}
    return prediction([{"answer": answer}], answer, cost_paths=calls), detail


def run_agent_comparison(
    client: ModelClient, q: Question, config: dict, knowledge: KnowledgeBase,
) -> tuple[dict, dict]:
    opening = opening_messages(q, config["agent_steps"])
    initial = generate_response(client, q, config, "agent-step-0", 0.0, messages=opening, max_tokens=96)
    basic, basic_trace = run_agent(client, q, config, knowledge, opening, initial, recover=False)
    harness, harness_trace = run_agent(client, q, config, knowledge, opening, initial, recover=True)
    return (
        {"agent_basic": basic, "agent_harness": harness},
        {"initial_generation": initial, "agent_basic": basic_trace, "agent_harness": harness_trace},
    )


def agent_calls(trace: dict) -> Iterator[dict]:
    yield trace["initial_generation"]
    for name in ("agent_basic", "agent_harness"):
        for step in trace[name]["steps"][1:]:
            yield step["generation"]


def summarize_agents(agent_traces: list[dict]) -> dict:
    stats = {}
    for name in ("agent_basic", "agent_harness"):
        traces = [trace[name] for trace in agent_traces]
        stats[name] = {
            "stop_reasons": dict(Counter(trace["stop_reason"] for trace in traces)),
            "tool_successes": sum(trace["tool_successes"] for trace in traces),
            "rejected_actions": sum(step["status"] == "rejected" for trace in traces for step in trace["steps"]),
            "recovered_after_rejection": sum(
                trace["stop_reason"] == "submitted" and any(step["status"] == "rejected" for step in trace["steps"])
                for trace in traces
            ),
            "successful_tools_by_name": dict(Counter(
                step["action"]["name"] for trace in traces for step in trace["steps"] if step["status"] == "tool_ok"
            )),
        }
    return stats
