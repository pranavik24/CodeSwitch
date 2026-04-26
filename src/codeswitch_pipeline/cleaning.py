from __future__ import annotations

import re
import unicodedata

from .text_utils import normalize_whitespace, tokenize_text

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,])\1{2,}")
REPEATED_CHAR_PATTERN = re.compile(r"([A-Za-z])\1{3,}")
SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"\s+([?.!,;:])")
RT_PATTERN = re.compile(r"^rt\s+", re.IGNORECASE)


def ascii_quote_normalize(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def strip_control_characters(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")


def clean_generation_text(text: str) -> str:
    text = ascii_quote_normalize(text)
    text = strip_control_characters(text)
    text = REPEATED_PUNCT_PATTERN.sub(r"\1", text)
    text = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", text)
    text = normalize_whitespace(text)
    return text.strip(" \"'")


def clean_spanglish_social_text(text: str) -> str:
    text = ascii_quote_normalize(text)
    text = strip_control_characters(text)
    text = RT_PATTERN.sub("", text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r" \1 ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = REPEATED_PUNCT_PATTERN.sub(r"\1", text)
    text = REPEATED_CHAR_PATTERN.sub(r"\1\1", text)
    text = SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", text)
    return normalize_whitespace(text)


def is_usable_clean_text(text: str, min_alpha_tokens: int = 3) -> bool:
    if not text:
        return False
    tokens = tokenize_text(text)
    alpha_tokens = [token for token in tokens if token.isalpha()]
    if len(alpha_tokens) < min_alpha_tokens:
        return False
    if len(text) < 8:
        return False
    alpha_ratio = len(alpha_tokens) / max(1, len(tokens))
    if alpha_ratio < 0.45:
        return False
    return True
