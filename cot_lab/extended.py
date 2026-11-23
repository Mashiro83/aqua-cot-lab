from collections.abc import Iterator

from .data import Question
from .experiments.agentic_harness.agent import agent_calls, run_agent_comparison, summarize_agents
from .experiments.multi_agent_debate.debate import debate_calls, run_debate, summarize_debate
from .experiments.retrieval_cot.retrieval import retrieval_calls, run_retrieval, summarize_retrieval
from .knowledge import KnowledgeBase
from .model import ModelClient


EXTENDED_METHODS = (
    "retrieval_cot", "interleaved_cot", "debate_independent", "debate_self_review",
    "multi_agent_debate", "agent_basic", "agent_harness",
)
EXTENDED_COMPARISONS = {
    "retrieval_cot_vs_cot": ("cot", "retrieval_cot"),
    "interleaved_cot_vs_retrieval_cot": ("retrieval_cot", "interleaved_cot"),
    "self_review_vs_independent": ("debate_independent", "debate_self_review"),
    "debate_vs_independent": ("debate_independent", "multi_agent_debate"),
    "debate_vs_self_review": ("debate_self_review", "multi_agent_debate"),
    "agent_harness_vs_basic": ("agent_basic", "agent_harness"),
}


def run_extended(
    client: ModelClient, q: Question, config: dict, knowledge: KnowledgeBase,
) -> tuple[dict, dict]:
    retrieval, retrieval_trace = run_retrieval(client, q, config, knowledge)
    debate, debate_trace = run_debate(client, q, config)
    agent, agent_trace = run_agent_comparison(client, q, config, knowledge)
    return (
        {**retrieval, **debate, **agent},
        {"retrieval": retrieval_trace, "debate": debate_trace, "agent": agent_trace},
    )


def extended_calls(trace: dict) -> Iterator[dict]:
    yield from retrieval_calls(trace["retrieval"])
    yield from debate_calls(trace["debate"])
    yield from agent_calls(trace["agent"])


def summarize_extended(rows: list[dict]) -> dict:
    traces = [row["extended"] for row in rows]
    return {
        "retrieval": summarize_retrieval([trace["retrieval"] for trace in traces]),
        "debate": summarize_debate([trace["debate"] for trace in traces]),
        "agent": summarize_agents([trace["agent"] for trace in traces]),
    }
