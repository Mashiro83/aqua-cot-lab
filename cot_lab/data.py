from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


LABELS = "ABCDE"
AQUA_REVISION = "26b64ed1f22208c6742f47df547dfd65088ec30d"
SPLITS = {"dev": "dev.json", "test": "test.json"}
OPTION = re.compile(r"^\s*([A-E])\s*[).:]\s*(\S.*)$", re.DOTALL)


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    options: tuple[str, ...]
    correct: str


def write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def parse_dataset(content: bytes, split: str) -> list[Question]:
    questions = []
    for line_no, line in enumerate(content.decode("utf-8-sig").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            text = row["question"].strip()
            options = row["options"]
            choices = {}
            for option in options:
                match = OPTION.fullmatch(option)
                if match:
                    choices[match[1]] = match[2].strip()
            if not text or len(options) != len(LABELS) or set(choices) != set(LABELS):
                raise ValueError("expected a question and five options labeled A through E")
            correct = row["correct"].strip().upper()
            if correct not in set(LABELS):
                raise ValueError("expected an answer label from A through E")
            labeled_choices = tuple(f"{label}) {choices[label]}" for label in LABELS)
            questions.append(Question(f"{split}-{line_no:04d}", text, labeled_choices, correct))
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            raise ValueError(f"{split} line {line_no}: {error}") from error
    if not questions:
        raise ValueError(f"{split}: empty dataset")
    return questions


def load_dataset(split: str, cache: Path, local: Path | None) -> tuple[list[Question], str]:
    path = local if local is not None else cache / AQUA_REVISION / SPLITS[split]
    if path.is_file() or local is not None:
        content = path.read_bytes()
        return parse_dataset(content, split), hashlib.sha256(content).hexdigest()
    url = f"https://raw.githubusercontent.com/google-deepmind/AQuA/{AQUA_REVISION}/{SPLITS[split]}"
    with urlopen(url, timeout=30) as response:
        content = response.read()
    questions = parse_dataset(content, split)
    write_atomic(path, content)
    return questions, hashlib.sha256(content).hexdigest()
