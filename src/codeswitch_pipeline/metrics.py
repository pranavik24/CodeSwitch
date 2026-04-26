from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from rouge_score import rouge_scorer

from .text_utils import COMMON_ENGLISH, COMMON_SPANISH, safe_ratio, split_sentences, tokenize_text


@dataclass
class LanguageProfile:
    english_tokens: int
    spanish_tokens: int
    other_tokens: int
    total_alpha_tokens: int
    switch_points: int

    @property
    def spanish_ratio(self) -> float:
        return safe_ratio(self.spanish_tokens, self.total_alpha_tokens)

    @property
    def has_codeswitch(self) -> bool:
        return self.english_tokens > 0 and self.spanish_tokens > 0


class LanguageIdentifier:
    def __init__(self, translation_lexicon: dict[str, list[str]] | None = None) -> None:
        self.translation_lexicon = translation_lexicon or {}
        self._external = None
        try:
            from codeswitch.codeswitch import LanguageIdentification  # type: ignore

            self._external = LanguageIdentification("spa-eng")
        except Exception:
            self._external = None

    def identify_tokens(self, text: str) -> list[tuple[str, str]]:
        if self._external is not None:
            try:
                result = self._external.identify(text)
                parsed = self._normalize_external_result(result)
                if parsed:
                    return parsed
            except Exception:
                pass
        return self._heuristic_identify(text)

    def profile(self, text: str) -> LanguageProfile:
        labeled_tokens = self.identify_tokens(text)
        english = 0
        spanish = 0
        other = 0
        switch_points = 0
        previous = None
        for token, label in labeled_tokens:
            if not token.isalpha():
                continue
            if label == "en":
                english += 1
            elif label == "es":
                spanish += 1
            else:
                other += 1
            if previous is not None and label in {"en", "es"} and previous in {"en", "es"} and label != previous:
                switch_points += 1
            if label in {"en", "es"}:
                previous = label
        total = english + spanish + other
        return LanguageProfile(
            english_tokens=english,
            spanish_tokens=spanish,
            other_tokens=other,
            total_alpha_tokens=total,
            switch_points=switch_points,
        )

    def detect_switch_type(self, text: str) -> str:
        profile = self.profile(text)
        if not profile.has_codeswitch:
            return "monolingual"

        sentences = split_sentences(text)
        sentence_labels: list[str] = []
        for sentence in sentences:
            sent_profile = self.profile(sentence)
            if sent_profile.spanish_tokens > sent_profile.english_tokens:
                sentence_labels.append("es")
            elif sent_profile.english_tokens > sent_profile.spanish_tokens:
                sentence_labels.append("en")
            elif sent_profile.has_codeswitch:
                sentence_labels.append("mixed")

        if len(sentence_labels) >= 2 and len({label for label in sentence_labels if label != "mixed"}) >= 2:
            return "inter-sentential"
        return "intra-sentential"

    def lince_style_score(
        self,
        text: str,
        target_ratio: float | None = None,
        target_switch_type: str | None = None,
    ) -> float:
        profile = self.profile(text)
        if target_ratio is not None and target_ratio <= 0.0:
            monolingual_penalty = min(1.0, profile.spanish_ratio)
            switch_penalty = 0.0 if self.detect_switch_type(text) in {"monolingual", "none"} else 0.1
            return round(max(0.0, 1.0 - monolingual_penalty - switch_penalty), 4)

        score = 0.0
        if profile.has_codeswitch:
            score += 0.4
        if profile.switch_points > 0:
            score += 0.2
        if target_ratio is not None:
            ratio_gap = abs(profile.spanish_ratio - target_ratio)
            score += max(0.0, 0.3 - ratio_gap)
        else:
            score += min(0.2, profile.spanish_ratio)
        if target_switch_type is not None:
            observed = self.detect_switch_type(text)
            if observed == target_switch_type:
                score += 0.1
        return round(min(1.0, score), 4)

    def _normalize_external_result(self, result: object) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    token = str(item.get("word", item.get("token", "")))
                    label = str(item.get("entity", item.get("label", ""))).lower()
                    if "lang1" in label or label.endswith("en"):
                        normalized.append((token, "en"))
                    elif "lang2" in label or label.endswith("es"):
                        normalized.append((token, "es"))
                    else:
                        normalized.append((token, "other"))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    token = str(item[0])
                    label = str(item[1]).lower()
                    if "lang1" in label or label.endswith("en"):
                        normalized.append((token, "en"))
                    elif "lang2" in label or label.endswith("es"):
                        normalized.append((token, "es"))
                    else:
                        normalized.append((token, "other"))
        return normalized

    def _heuristic_identify(self, text: str) -> list[tuple[str, str]]:
        labeled: list[tuple[str, str]] = []
        spanish_vocab = set(COMMON_SPANISH)
        english_vocab = set(COMMON_ENGLISH)
        lexicon_targets = {target for targets in self.translation_lexicon.values() for target in targets}

        for token in tokenize_text(text):
            token_lower = token.lower()
            if not token.isalpha():
                labeled.append((token, "other"))
            elif token_lower in spanish_vocab or token_lower in lexicon_targets:
                labeled.append((token, "es"))
            elif token_lower in english_vocab or token_lower in self.translation_lexicon:
                labeled.append((token, "en"))
            elif token_lower.endswith(("ando", "iendo", "ción", "mente", "ado", "ada")):
                labeled.append((token, "es"))
            else:
                labeled.append((token, "en"))
        return labeled


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
    }


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = [token.lower() for token in tokenize_text(prediction) if token.isalpha()]
    ref_tokens = [token.lower() for token in tokenize_text(reference) if token.isalpha()]
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    precision = safe_ratio(overlap, sum(pred_counts.values()))
    recall = safe_ratio(overlap, sum(ref_counts.values()))
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
