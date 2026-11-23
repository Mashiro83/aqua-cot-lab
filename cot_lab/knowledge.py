from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


DEFAULT_KNOWLEDGE = Path(__file__).with_name("math_knowledge.json")
WORD = re.compile(r"[a-z]+")
STOP_WORDS = frozenset(
    "a an and are at be by can choose does for from has have how if in is it many much of on or "
    "per that the their them then there these they this to was were what when which will with would "
    "calculate compute find know math number problem question use using value".split()
)


def terms(text: str) -> list[str]:
    return [word for word in WORD.findall(text.lower()) if word not in STOP_WORDS]


@dataclass(frozen=True)
class Card:
    id: str
    title: str
    keywords: str
    content: str


class KnowledgeBase:
    def __init__(self, path: Path = DEFAULT_KNOWLEDGE):
        content = path.read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        raw = json.loads(content)
        try:
            self.cards = tuple(Card(**item) for item in raw)
            if not self.cards or len({card.id for card in self.cards}) != len(self.cards):
                raise ValueError("expected nonempty math knowledge with unique card IDs")
            self.index = tuple(Counter(terms(f"{c.title} {c.keywords} {c.content}")) for c in self.cards)
        except (AttributeError, TypeError) as error:
            raise ValueError(f"invalid math knowledge: {error}") from error
        self.lengths = tuple(counter.total() for counter in self.index)
        self.mean_length = sum(self.lengths) / len(self.cards)
        if not self.mean_length:
            raise ValueError("math knowledge contains no indexable words")
        counts = Counter(term for counter in self.index for term in counter)
        self.idf = {
            term: math.log1p((len(self.cards) - count + 0.5) / (count + 0.5)) for term, count in counts.items()
        }

    def search(self, query: str, limit: int, *, exclude: frozenset[str] = frozenset()) -> list[dict]:
        query_terms = set(terms(query)).intersection(self.idf)
        hits = []
        for card, frequencies, length in zip(self.cards, self.index, self.lengths):
            if card.id in exclude:
                continue
            norm = 1.2 * (0.25 + 0.75 * length / self.mean_length)
            score = sum(
                self.idf[term] * 2.2 * frequencies[term] / (frequencies[term] + norm)
                for term in sorted(query_terms.intersection(frequencies))
            )
            if score > 0:
                hits.append({"id": card.id, "title": card.title, "content": card.content, "score": score})
        return sorted(hits, key=lambda item: (-item["score"], item["id"]))[:limit]


def render_hits(hits: list[dict]) -> str:
    return "\n".join(f'[{hit["id"]}] {hit["title"]}: {hit["content"]}' for hit in hits) or "No matching references."
