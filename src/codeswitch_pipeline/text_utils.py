from __future__ import annotations

import random
import re
from collections import Counter
from typing import Iterable

import numpy as np
import torch

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", re.UNICODE)
QUOTED_TOKEN_PATTERN = re.compile(r"'([^']*)'|\"([^\"]*)\"")
NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}", "'s", "n't"}
NO_SPACE_AFTER = {"(", "[", "{", "¿", "¡"}

COMMON_ENGLISH = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "is",
    "are",
    "to",
    "for",
    "please",
    "need",
    "want",
    "find",
    "book",
    "hotel",
    "restaurant",
    "train",
    "taxi",
}
COMMON_SPANISH = {
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "y",
    "o",
    "por",
    "favor",
    "necesito",
    "quiero",
    "buscar",
    "reservar",
    "hotel",
    "restaurante",
    "tren",
    "taxi",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def parse_array_string(raw: str) -> list[str]:
    if not isinstance(raw, str):
        return []
    tokens: list[str] = []
    for single, double in QUOTED_TOKEN_PATTERN.findall(raw):
        token = single or double
        token = token.replace("\\'", "'").replace('\\"', '"')
        if token:
            tokens.append(token)
    return tokens


def join_tokens(tokens: Iterable[str]) -> str:
    built: list[str] = []
    for token in tokens:
        if not built:
            built.append(token)
            continue
        if token in NO_SPACE_BEFORE or token.startswith("'"):
            built[-1] = built[-1] + token
            continue
        if built[-1] in NO_SPACE_AFTER:
            built[-1] = built[-1] + token
            continue
        built.append(token)
    return normalize_whitespace(" ".join(built))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", normalize_whitespace(text))
    return [part for part in parts if part]


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def balanced_assignments(
    item_ids: list[str],
    switch_types: list[str],
    switch_ratios: list[float],
    seed: int,
) -> list[tuple[str, str, float]]:
    rng = random.Random(seed)
    strata = [(switch_type, switch_ratio) for switch_type in switch_types for switch_ratio in switch_ratios]
    repeated = [strata[idx % len(strata)] for idx in range(len(item_ids))]
    rng.shuffle(repeated)
    return [(item_id, switch_type, switch_ratio) for item_id, (switch_type, switch_ratio) in zip(item_ids, repeated)]


def repeated_token_penalty(text: str) -> float:
    tokens = [token.lower() for token in tokenize_text(text) if token.isalpha()]
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return min(1.0, repeated / max(1, len(tokens)))
