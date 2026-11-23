from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path

from .data import Question, write_atomic
from .experiments.self_consistency.sampling import predict_self_consistency, sample_paths
from .experiments.step_verifier.verifier import can_stop, weighted_vote
from .extended import EXTENDED_COMPARISONS, EXTENDED_METHODS, extended_calls, run_extended, summarize_extended
from .inference import generate_path, prediction, token_sum
from .knowledge import KnowledgeBase
from .model import ModelClient


COMPARE_METHODS = ("direct", "cot", "self_consistency", "verified_vote", "adaptive_vote")


def run_question(client: ModelClient, q: Question, config: dict, knowledge: KnowledgeBase | None = None) -> dict:
    compare = config["mode"] == "compare"
    methods = {}
    baselines = {}
    if compare:
        for name in ("direct", "cot"):
            path = generate_path(client, q, config, name, 0.0)
            baselines[name] = path
            methods[name] = prediction([path], path["answer"])

    paths = []
    stop_at = None
    for path in sample_paths(client, q, config):
        paths.append(path)
        if stop_at is None and can_stop(paths, config["min_paths"]):
            stop_at = len(paths)
            if not compare:
                break

    if compare:
        methods["self_consistency"] = predict_self_consistency(paths)
        methods["verified_vote"] = prediction(paths, *weighted_vote(paths))
    prefix = paths[:stop_at]
    methods["adaptive_vote"] = prediction(prefix, *weighted_vote(prefix))
    row = {
        "id": q.id,
        "question": q.text,
        "options": q.options,
        "correct": q.correct,
        "predictions": methods,
        "baseline_paths": baselines,
        "sampled_paths": paths,
        "adaptive_stopped_early": stop_at is not None and stop_at < config["paths"],
    }
    if knowledge is not None:
        additional, trace = run_extended(client, q, config, knowledge)
        methods.update(additional)
        row["extended"] = trace
    return row


def mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows]
    return None if None in values else sum(values) / len(values)


def paired_change(rows: list[dict], baseline: str, method: str) -> dict:
    pairs = [
        (row["predictions"][baseline]["answer"] == row["correct"],
         row["predictions"][method]["answer"] == row["correct"])
        for row in rows
    ]
    corrections = pairs.count((False, True))
    regressions = pairs.count((True, False))
    return {
        "corrections": corrections,
        "regressions": regressions,
        "accuracy_delta": (corrections - regressions) / len(rows),
    }


def summarize(rows: list[dict], config: dict) -> dict:
    method_names = COMPARE_METHODS if config["mode"] == "compare" else ("adaptive_vote",)
    extended = config.get("suite") == "full"
    if extended:
        method_names += EXTENDED_METHODS
    methods = []
    for name in method_names:
        predictions = [row["predictions"][name] for row in rows]
        correct = sum(p["answer"] == row["correct"] for row, p in zip(rows, predictions))
        valid = sum(p["answer"] is not None for p in predictions)
        methods.append({
            "method": name,
            "questions": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
            "valid_answer_rate": valid / len(rows),
            "mean_paths": mean(predictions, "paths"),
            "mean_http_requests": mean(predictions, "http_requests"),
            "mean_prompt_tokens": mean(predictions, "prompt_tokens"),
            "mean_completion_tokens": mean(predictions, "completion_tokens"),
            "mean_model_seconds": mean(predictions, "model_seconds"),
        })
        if extended:
            methods[-1]["mean_model_calls"] = sum(
                p.get("model_calls", p["paths"]) for p in predictions
            ) / len(rows)

    sampled = [path for row in rows for path in row["sampled_paths"]]
    saved_paths = sum(len(row["sampled_paths"]) - row["predictions"]["adaptive_vote"]["paths"] for row in rows)
    all_calls = sampled + [path for row in rows for path in row["baseline_paths"].values()]
    if extended:
        all_calls.extend(call for row in rows for call in extended_calls(row["extended"]))
    summary = {
        "methods": methods,
        "sampled_path_oracle_accuracy": sum(
            any(path["answer"] == row["correct"] for path in row["sampled_paths"]) for row in rows
        ) / len(rows),
        "sampled_paths_by_evidence": dict(Counter(path["evidence"] for path in sampled)),
        "equations_by_status": dict(Counter(check["status"] for path in sampled for check in path["checks"])),
        "adaptive": {
            "stopped_early": sum(row["adaptive_stopped_early"] for row in rows),
            "paths_saved_vs_maximum": len(rows) * config["paths"] - sum(
                row["predictions"]["adaptive_vote"]["paths"] for row in rows
            ),
            "paths_actually_generated_after_stop": saved_paths,
        },
        "total": {
            "generations": len(all_calls),
            "http_requests": sum(path["requests"] for path in all_calls),
            "prompt_tokens": token_sum(all_calls, "prompt_tokens"),
            "completion_tokens": token_sum(all_calls, "completion_tokens"),
            "model_seconds": sum(path["seconds"] for path in all_calls),
        },
    }
    if config["mode"] == "compare":
        summary["paired_changes"] = {
            "cot_vs_direct": paired_change(rows, "direct", "cot"),
            "self_consistency_vs_cot": paired_change(rows, "cot", "self_consistency"),
            "verified_vote_vs_self_consistency": paired_change(rows, "self_consistency", "verified_vote"),
            "adaptive_vote_vs_verified_vote": paired_change(rows, "verified_vote", "adaptive_vote"),
        }
    if extended:
        summary["paired_changes"].update({
            name: paired_change(rows, baseline, method)
            for name, (baseline, method) in EXTENDED_COMPARISONS.items()
        })
        summary["extended"] = summarize_extended(rows)
    return summary


def write_json(path: Path, value: dict) -> None:
    write_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def run(
    client: ModelClient, questions: list[Question], config: dict, output: Path,
    knowledge: KnowledgeBase | None = None,
) -> None:
    manifest = output / "run_config.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("run configuration changed; choose a new --output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output is not empty and has no run configuration")
        write_json(manifest, config)

    def get_row(question: Question) -> tuple[dict, str]:
        checkpoint = output / "checkpoints" / f"{question.id}.json"
        if checkpoint.exists():
            return json.loads(checkpoint.read_text(encoding="utf-8")), "cached"
        row = run_question(client, question, config, knowledge)
        write_json(checkpoint, row)
        return row, "done"

    rows = []
    workers = ThreadPoolExecutor(max_workers=config.get("workers", 1))
    try:
        for index, (row, state) in enumerate(workers.map(get_row, questions), 1):
            rows.append(row)
            print(f"[{index}/{len(questions)}] {row['id']} {state}", flush=True)
    finally:
        workers.shutdown(wait=True, cancel_futures=True)

    summary = summarize(rows, config)
    write_json(output / "summary.json", summary)
    jsonl = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_atomic(output / "predictions.jsonl", jsonl.encode("utf-8"))
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(summary["methods"][0]))
    writer.writeheader()
    writer.writerows(summary["methods"])
    write_atomic(output / "metrics.csv", table.getvalue().encode("utf-8"))
    for method in summary["methods"]:
        print(f'{method["method"]:20s} {method["accuracy"]:6.1%}  paths={method["mean_paths"]:.2f}')
