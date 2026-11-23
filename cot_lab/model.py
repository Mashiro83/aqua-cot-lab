from dataclasses import dataclass
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .data import Question


DIRECT_PROMPT = "Choose one option. Reply with a single line in the format FINAL: <A-E>."
COT_PROMPT = (
    "Solve each multiple-choice math problem in plain text. Use at most four short steps. "
    "Put each numeric calculation on its own line as CALC: expression = result. "
    "Use numbers and +, -, *, /, **, and parentheses. Write ~= for a rounded decimal. "
    "Do not use Markdown or LaTeX. Finish with one line in the exact format FINAL: A, "
    "replacing A with the chosen letter."
)
FORMAT_EXAMPLE = """A shop has 8 boxes with 6 pencils each. A student buys 7 pencils. How many pencils are left?

Options
A) 41
B) 42
C) 48
D) 49
E) 55"""
COT_EXAMPLE_ANSWER = "CALC: 8 * 6 = 48\nCALC: 48 - 7 = 41\nFINAL: A"


def messages_for(question: Question, direct: bool) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIRECT_PROMPT if direct else COT_PROMPT},
        {"role": "user", "content": FORMAT_EXAMPLE},
        {"role": "assistant", "content": "FINAL: A" if direct else COT_EXAMPLE_ANSWER},
        {"role": "user", "content": question_text(question)},
    ]


def question_text(question: Question) -> str:
    options = "\n".join(question.options)
    return f"{question.text}\n\nOptions\n{options}"


def cot_with_context(q: Question, context: str) -> list[dict[str, str]]:
    messages = messages_for(q, False)
    messages[-1]["content"] += (
        f"\n\n{context}\n\nAnswer the original question in at most four short plain-text lines. "
        "Do not repeat any draft or query. Write FINAL: followed by your single answer letter on the last line."
    )
    return messages


@dataclass(frozen=True)
class Generation:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    seconds: float
    requests: int


class ModelClient:
    def __init__(self, base_url: str, model: str, key: str, timeout: float, retries: int, api_seed: bool):
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username is not None or url.query or url.fragment:
            raise ValueError("base URL must be an HTTP(S) endpoint without credentials, query or fragment")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = key
        self.timeout = timeout
        self.retries = retries
        self.api_seed = api_seed

    def generate(
        self, question: Question, *, direct: bool, temperature: float, max_tokens: int, seed: int,
        messages: list[dict[str, str]] | None = None,
    ) -> Generation:
        body = {
            "model": self.model,
            "messages": messages_for(question, direct) if messages is None else messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.api_seed:
            body["seed"] = seed
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = Request(self.endpoint, json.dumps(body).encode(), headers, method="POST")
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                break
            except HTTPError as error:
                error.close()
                retryable = error.code in (408, 429) or 500 <= error.code < 600
                if not retryable or attempt == self.retries:
                    raise RuntimeError(f"model endpoint returned HTTP {error.code}") from None
            except (URLError, OSError) as error:
                if attempt == self.retries:
                    raise RuntimeError(f"model request failed: {error}") from None
            time.sleep(min(2**attempt, 8))

        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"] or ""
            if not isinstance(content, str):
                raise TypeError("expected text content")
            usage = payload.get("usage") or {}
            return Generation(
                content,
                choice.get("finish_reason"),
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                time.perf_counter() - started,
                attempt + 1,
            )
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise RuntimeError(f"unexpected chat completion response: {error}") from None
