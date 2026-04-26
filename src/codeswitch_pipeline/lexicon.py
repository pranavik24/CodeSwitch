from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .text_utils import tokenize_text


class TranslationLexicon:
    def __init__(self, token_map: dict[str, list[str]]) -> None:
        self.token_map = token_map

    @classmethod
    def from_tatoeba(cls, path: str | Path, min_count: int = 2) -> "TranslationLexicon":
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                english = parts[0].strip()
                spanish = parts[1].strip()
                english_tokens = [token.lower() for token in tokenize_text(english) if token.isalpha()]
                spanish_tokens = [token.lower() for token in tokenize_text(spanish) if token.isalpha()]
                if not english_tokens or not spanish_tokens:
                    continue
                if len(english_tokens) == 1 and len(spanish_tokens) == 1:
                    counts[english_tokens[0]][spanish_tokens[0]] += 4
                if len(english_tokens) == len(spanish_tokens) and len(english_tokens) <= 6:
                    for source, target in zip(english_tokens, spanish_tokens):
                        counts[source][target] += 1

        token_map: dict[str, list[str]] = {}
        for english_token, spanish_counts in counts.items():
            valid = [token for token, count in spanish_counts.most_common() if count >= min_count and token != english_token]
            if valid:
                token_map[english_token] = valid[:5]
        return cls(token_map)

    def get(self, token: str) -> list[str]:
        return self.token_map.get(token.lower(), [])

    def candidates_for_text(self, text: str, limit: int = 20) -> dict[str, list[str]]:
        suggestions: dict[str, list[str]] = {}
        for token in tokenize_text(text):
            if not token.isalpha():
                continue
            token_lower = token.lower()
            if token_lower in suggestions:
                continue
            candidates = self.get(token_lower)
            if candidates:
                suggestions[token_lower] = candidates[:3]
            if len(suggestions) >= limit:
                break
        return suggestions
