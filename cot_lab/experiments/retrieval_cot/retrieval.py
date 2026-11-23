from collections.abc import Iterator
import re

from ...data import Question
from ...inference import generate_path, generate_response, prediction
from ...knowledge import KnowledgeBase, render_hits
from ...model import ModelClient, cot_with_context, question_text


QUERY_LINE = re.compile(r"QUERY:\s*(.{1,160})", re.IGNORECASE)
PLAN_LINE = re.compile(r"PLAN:\s*(.{1,240})", re.IGNORECASE)
PLAN_PROMPT = (
    "Develop one short intermediate reasoning step for the math question using the given references. "
    "Write PLAN: followed by your step, then QUERY: followed by up to eight keywords for a missing "
    "mathematical rule. Keep the query generic. Do not select a final option yet. "
    "Use plain text. Example:\nPLAN: Use the work completed per hour for each worker.\nQUERY: combined work rates"
)


def query_from(response: dict) -> str | None:
    lines = response["text"].strip().splitlines()
    if not lines or response["finish_reason"] not in (None, "stop"):
        return None
    match = QUERY_LINE.fullmatch(lines[-1].strip())
    return match[1].strip() if match and match[1].strip() else None


def plan_from(response: dict) -> str | None:
    for line in response["text"].strip().splitlines():
        match = PLAN_LINE.match(line.strip())
        if match and match[1].strip():
            return match[1].strip()
    return None


def run_retrieval(client: ModelClient, q: Question, config: dict, knowledge: KnowledgeBase) -> tuple[dict, dict]:
    static_hits = knowledge.search(q.text, 4)
    static_context = f"Retrieved math references. Use any that apply.\n{render_hits(static_hits)}"
    static = generate_path(client, q, config, "retrieval-final", 0.0, messages=cot_with_context(q, static_context))

    initial_hits = static_hits[:2]
    plan = generate_response(
        client, q, config, "retrieval-plan", 0.0, max_tokens=112, messages=[
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": f"{question_text(q)}\n\nMath references\n{render_hits(initial_hits)}"},
        ],
    )
    query = query_from(plan)
    followup = knowledge.search(query, 2, exclude=frozenset(hit["id"] for hit in initial_hits)) if query else []
    selected = {hit["id"] for hit in initial_hits + followup}
    fallback = [hit for hit in static_hits if hit["id"] not in selected][:2 - len(followup)]
    references = render_hits(initial_hits + followup + fallback)
    context = f"Retrieved math references. Use any that apply.\n{references}"
    intermediate = plan_from(plan) if query else None
    if intermediate:
        context += f"\n\nIntermediate idea to check: {intermediate}"
    refined = generate_path(client, q, config, "retrieval-final", 0.0, messages=cot_with_context(q, context))
    return (
        {"retrieval_cot": prediction([static], static["answer"]),
         "interleaved_cot": prediction([refined], refined["answer"], cost_paths=[plan, refined])},
        {"static_hits": static_hits, "initial_hits": initial_hits, "refined_query": query,
         "additional_hits": followup, "fallback_hits": fallback, "intermediate_reasoning": intermediate,
         "static_response": static,
         "plan_response": plan, "refined_response": refined},
    )


def retrieval_calls(trace: dict) -> Iterator[dict]:
    for key in ("static_response", "plan_response", "refined_response"):
        yield trace[key]


def summarize_retrieval(traces: list[dict]) -> dict:
    return {
        "valid_refined_queries": sum(trace["refined_query"] is not None for trace in traces),
        "questions_with_additional_hits": sum(bool(trace["additional_hits"]) for trace in traces),
        "mean_static_cards": sum(len(trace["static_hits"]) for trace in traces) / len(traces),
        "mean_interleaved_cards": sum(
            len(trace["initial_hits"]) + len(trace["additional_hits"]) + len(trace["fallback_hits"])
            for trace in traces
        ) / len(traces),
    }
