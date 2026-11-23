from collections.abc import Iterator

from ...data import Question
from ...inference import generate_path, prediction
from ...model import ModelClient, cot_with_context
from ...reasoning import vote


ROLE_INSTRUCTIONS = (
    "Translate the givens into equations and derive the target quantity.",
    "Use a different strategy such as checking options or estimates. Check units and assumptions.",
)


def render_draft(path: dict) -> str:
    draft = path["text"].strip()
    if len(draft) > 720:
        return f"{draft[:450]}\n[... omitted ...]\n{draft[-250:]}"
    return draft


def review_context(role: str, own: dict, peer: dict | None) -> str:
    prompt = f"Your role: {role}\nIndependently check the math and correct any mistake you find."
    prompt += f"\n\nYour previous draft\n{render_draft(own)}"
    if peer is not None:
        prompt += f"\n\nThe other solver's draft, which may be wrong\n{render_draft(peer)}"
    return prompt


def run_debate(client: ModelClient, q: Question, config: dict) -> tuple[dict, dict]:
    temperature = config["temperature"]
    initial = [
        generate_path(
            client, q, config, f"debate-initial-{index}", temperature,
            messages=cot_with_context(q, f"Your role: {role}"),
        )
        for index, role in enumerate(ROLE_INSTRUCTIONS)
    ]
    methods = {"debate_independent": prediction(initial, *vote(initial))}
    detail = {"roles": ROLE_INSTRUCTIONS, "initial": initial}
    for use_peer, name in ((False, "debate_self_review"), (True, "multi_agent_debate")):
        previous = initial
        rounds = []
        for turn in range(config["debate_rounds"]):
            # Each participant reads the previous round. The current round is published together.
            previous = [
                generate_path(
                    client, q, config, f"debate-review-{turn}-{index}", temperature,
                    messages=cot_with_context(q, review_context(role, previous[index], previous[1 - index] if use_peer else None)),
                )
                for index, role in enumerate(ROLE_INSTRUCTIONS)
            ]
            rounds.append(previous)
        costs = initial + [path for round_paths in rounds for path in round_paths]
        methods[name] = prediction(previous, *vote(previous), cost_paths=costs)
        detail[name] = rounds
    return methods, detail


def debate_calls(trace: dict) -> Iterator[dict]:
    yield from trace["initial"]
    for name in ("debate_self_review", "multi_agent_debate"):
        for round_paths in trace[name]:
            yield from round_paths


def agreement(paths: list[dict]) -> bool:
    return paths[0]["answer"] is not None and paths[0]["answer"] == paths[1]["answer"]


def summarize_debate(traces: list[dict]) -> dict:
    return {
        "initial_agreements": sum(agreement(trace["initial"]) for trace in traces),
        "self_review_agreements": sum(agreement(trace["debate_self_review"][-1]) for trace in traces),
        "debate_agreements": sum(agreement(trace["multi_agent_debate"][-1]) for trace in traces),
    }
