import argparse
import math
import os
from pathlib import Path
import random
import sys

from .data import AQUA_REVISION, SPLITS, load_dataset
from .experiment import run
from .knowledge import DEFAULT_KNOWLEDGE, KnowledgeBase
from .model import ModelClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate chain-of-thought methods on AQuA.")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--data-file", type=Path, help="Use a local AQuA JSONL file.")
    parser.add_argument("--cache", type=Path, default=Path(".cache/aqua"))
    parser.add_argument("--limit", type=int, help="Evaluate a seeded subset of this size.")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="New or matching run directory, for example results/<model>/<run>.",
    )
    parser.add_argument("--mode", choices=("compare", "adaptive"), default="compare")
    parser.add_argument("--suite", choices=("core", "full"), default="core", help="Run the core or all course experiments.")
    parser.add_argument("--workers", type=int, default=1, help="Number of questions evaluated concurrently.")
    parser.add_argument("--knowledge-file", type=Path, default=DEFAULT_KNOWLEDGE, help="Formula cards for the full suite.")
    parser.add_argument("--debate-rounds", type=int, default=1, help="Revision rounds after initial independent drafts.")
    parser.add_argument("--agent-steps", type=int, default=4, help="Model turn budget per tool agent.")
    parser.add_argument("--paths", type=int, default=5, help="Maximum number of sampled CoT paths.")
    parser.add_argument("--min-paths", type=int, default=3, help="Paths required before checking early stop.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direct-tokens", type=int, default=24)
    parser.add_argument("--cot-tokens", type=int, default=384)
    parser.add_argument("--model", default=os.getenv("COT_MODEL"), help="Model ID served by the endpoint. Defaults to COT_MODEL.")
    parser.add_argument("--base-url", default=os.getenv("COT_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-api-seed", action="store_true", help="Omit seed for endpoints that reject it.")
    args = parser.parse_args()

    if not args.model or not args.model.strip():
        parser.error("set --model or the COT_MODEL environment variable")
    if not 2 <= args.min_paths <= args.paths:
        parser.error("expected 2 <= --min-paths <= --paths")
    if not math.isfinite(args.temperature) or not 0 < args.temperature <= 2:
        parser.error("--temperature must be within (0, 2]")
    if min(args.direct_tokens, args.cot_tokens) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("token budgets and --limit must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.retries < 0:
        parser.error("--timeout must be positive and --retries cannot be negative")
    if args.workers < 1 or args.debate_rounds < 1 or args.agent_steps < 2:
        parser.error("expected positive --workers and --debate-rounds and at least 2 --agent-steps")
    if args.suite == "full" and args.mode != "compare":
        parser.error("--suite full requires --mode compare; use --mode adaptive with --suite core")

    try:
        client = ModelClient(
            args.base_url, args.model, os.getenv("COT_API_KEY", ""),
            args.timeout, args.retries, not args.no_api_seed,
        )
        questions, digest = load_dataset(args.split, args.cache, args.data_file)
        if args.limit is not None:
            if args.limit > len(questions):
                parser.error(f"--limit exceeds split size ({len(questions)})")
            random.Random(args.seed).shuffle(questions)
            questions = questions[:args.limit]
        config = {
            "protocol_version": 2,
            "split": args.split,
            "dataset_revision": AQUA_REVISION if args.data_file is None else None,
            "dataset_sha256": digest,
            "question_ids": [q.id for q in questions],
            "mode": args.mode,
            "paths": args.paths,
            "min_paths": args.min_paths,
            "temperature": args.temperature,
            "seed": args.seed,
            "direct_tokens": args.direct_tokens,
            "cot_tokens": args.cot_tokens,
            "model": args.model,
            "base_url": args.base_url.rstrip("/"),
            "api_seed": not args.no_api_seed,
        }
        knowledge = None
        if args.suite == "full":
            knowledge = KnowledgeBase(args.knowledge_file)
            config.update(
                protocol_version=4, suite="full", knowledge_sha256=knowledge.sha256,
                debate_rounds=args.debate_rounds, agent_steps=args.agent_steps, workers=args.workers,
            )
        elif args.workers != 1:
            config["workers"] = args.workers
        run(client, questions, config, args.output, knowledge)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted; completed questions are saved for resume", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
