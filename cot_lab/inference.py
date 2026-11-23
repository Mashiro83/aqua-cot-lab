from dataclasses import asdict
import hashlib

from .data import Question
from .experiments.step_verifier.verifier import evidence, verify_steps
from .model import ModelClient
from .reasoning import final_answer


def stable_seed(seed: int, question_id: str, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{question_id}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def generate_response(
    client: ModelClient, q: Question, config: dict, name: str, temperature: float, *,
    messages: list[dict[str, str]] | None = None, max_tokens: int | None = None,
) -> dict:
    direct = name == "direct"
    seed = stable_seed(config["seed"], q.id, name)
    if max_tokens is None:
        max_tokens = config["direct_tokens" if direct else "cot_tokens"]
    generation = client.generate(
        q,
        direct=direct,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        messages=messages,
    )
    return {**asdict(generation), "seed": seed if config["api_seed"] else None, "temperature": temperature}


def generate_path(
    client: ModelClient, q: Question, config: dict, name: str, temperature: float, *,
    messages: list[dict[str, str]] | None = None,
) -> dict:
    response = generate_response(client, q, config, name, temperature, messages=messages)
    checks = () if name == "direct" else verify_steps(response["text"])
    return {
        **response,
        "answer": final_answer(response["text"], response["finish_reason"]),
        "evidence": evidence(checks),
        "checks": [asdict(check) for check in checks],
    }


def token_sum(paths: list[dict], key: str) -> int | None:
    values = [path[key] for path in paths]
    return None if None in values else sum(values)


def prediction(
    paths: list[dict], answer: str | None, scores: dict[str, int] | None = None, *,
    cost_paths: list[dict] | None = None,
) -> dict:
    calls = paths if cost_paths is None else cost_paths
    return {
        "answer": answer,
        "scores": scores,
        "paths": len(paths),
        "valid_paths": sum(path["answer"] is not None for path in paths),
        "http_requests": sum(path["requests"] for path in calls),
        "prompt_tokens": token_sum(calls, "prompt_tokens"),
        "completion_tokens": token_sum(calls, "completion_tokens"),
        "model_seconds": sum(path["seconds"] for path in calls),
        **({"model_calls": len(calls)} if cost_paths is not None else {}),
    }
