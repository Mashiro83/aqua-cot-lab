from collections.abc import Iterator

from ...data import Question
from ...inference import generate_path, prediction
from ...model import ModelClient
from ...reasoning import vote


def sample_paths(client: ModelClient, question: Question, config: dict) -> Iterator[dict]:
    for index in range(config["paths"]):
        yield generate_path(client, question, config, f"sample-{index}", config["temperature"])


def predict_self_consistency(paths: list[dict]) -> dict:
    return prediction(paths, *vote(paths))
