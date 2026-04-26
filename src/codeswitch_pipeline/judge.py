from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .metrics import LanguageIdentifier
from .text_utils import cosine_similarity, normalize_whitespace, repeated_token_penalty, safe_ratio


@dataclass
class PromptJudgeResult:
    naturalness: int
    code_switching_quality: int
    grammar_fluency: int
    conversation_realism: int
    emotion_consistency: int
    overall: int
    observed_spanish_ratio: float
    observed_switch_type: str


@dataclass
class ResponseJudgeResult:
    relevance: int
    fluency: int
    completeness: int
    alignment: int
    code_switch_fit: int
    overall: int


class XLMRSemanticEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Iterable[str], batch_size: int = 16) -> np.ndarray:
        batches = list(texts)
        embeddings: list[np.ndarray] = []
        for start in range(0, len(batches), batch_size):
            chunk = batches[start : start + batch_size]
            tokens = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**tokens).last_hidden_state
            attention = tokens["attention_mask"].unsqueeze(-1)
            pooled = (outputs * attention).sum(dim=1) / attention.sum(dim=1).clamp(min=1)
            embeddings.extend(pooled.detach().cpu().numpy())
        return np.asarray(embeddings)


class PromptJudge:
    def __init__(
        self,
        model_name: str,
        natural_reference_texts: list[str],
        language_identifier: LanguageIdentifier,
        reference_limit: int = 2048,
    ) -> None:
        self.encoder = XLMRSemanticEncoder(model_name)
        self.language_identifier = language_identifier
        self.reference_texts = natural_reference_texts[:reference_limit]
        self.reference_embeddings = (
            self.encoder.encode(self.reference_texts, batch_size=32) if self.reference_texts else np.zeros((0, 768))
        )

    @lru_cache(maxsize=4096)
    def _embed_one(self, text: str) -> np.ndarray:
        return self.encoder.encode([text], batch_size=1)[0]

    def score_prompt(
        self,
        original_prompt: str,
        rewritten_prompt: str,
        target_spanish_ratio: float,
        target_switch_type: str,
    ) -> PromptJudgeResult:
        original_prompt = normalize_whitespace(original_prompt)
        rewritten_prompt = normalize_whitespace(rewritten_prompt)

        observed_profile = self.language_identifier.profile(rewritten_prompt)
        observed_switch_type = self.language_identifier.detect_switch_type(rewritten_prompt)

        similarity = cosine_similarity(self._embed_one(original_prompt), self._embed_one(rewritten_prompt))
        naturalness_similarity = self._naturalness_similarity(rewritten_prompt)
        punctuation_match = self._punctuation_preservation(original_prompt, rewritten_prompt)
        repetition_penalty = repeated_token_penalty(rewritten_prompt)
        ratio_gap = abs(observed_profile.spanish_ratio - target_spanish_ratio)

        naturalness = self._to_five_point(0.55 * naturalness_similarity + 0.45 * (1.0 - repetition_penalty))
        code_switch = self._to_five_point(
            0.55 * max(0.0, 1.0 - ratio_gap)
            + 0.25 * float(observed_profile.has_codeswitch)
            + 0.20 * float(observed_switch_type == target_switch_type)
        )
        grammar = self._to_five_point(
            0.60 * (1.0 - repetition_penalty) + 0.25 * punctuation_match + 0.15 * min(1.0, safe_ratio(len(rewritten_prompt), 96))
        )
        realism = self._to_five_point(0.75 * similarity + 0.25 * punctuation_match)
        emotion = self._to_five_point(0.65 * similarity + 0.35 * punctuation_match)
        overall = int(round(np.mean([naturalness, code_switch, grammar, realism, emotion])))
        if not observed_profile.has_codeswitch:
            code_switch = 1
            overall = min(overall, 2)
        elif observed_switch_type != target_switch_type:
            overall = min(overall, 4)

        return PromptJudgeResult(
            naturalness=naturalness,
            code_switching_quality=code_switch,
            grammar_fluency=grammar,
            conversation_realism=realism,
            emotion_consistency=emotion,
            overall=max(1, min(5, overall)),
            observed_spanish_ratio=round(observed_profile.spanish_ratio, 4),
            observed_switch_type=observed_switch_type,
        )

    def _naturalness_similarity(self, text: str) -> float:
        if self.reference_embeddings.size == 0:
            return 0.5
        vector = self._embed_one(text)
        similarities = np.dot(self.reference_embeddings, vector) / (
            np.linalg.norm(self.reference_embeddings, axis=1) * np.linalg.norm(vector) + 1e-8
        )
        top_k = np.sort(similarities)[-5:]
        return float(np.clip(np.mean(top_k), 0.0, 1.0))

    def _punctuation_preservation(self, original: str, rewritten: str) -> float:
        original_question = "?" in original
        rewritten_question = "?" in rewritten
        original_exclaim = "!" in original
        rewritten_exclaim = "!" in rewritten
        return (
            0.5 * float(original_question == rewritten_question)
            + 0.5 * float(original_exclaim == rewritten_exclaim)
        )

    def _to_five_point(self, raw_value: float) -> int:
        clamped = max(0.0, min(1.0, raw_value))
        return max(1, min(5, int(math.ceil(clamped * 5))))


class ResponseJudge:
    def __init__(self, model_name: str, language_identifier: LanguageIdentifier) -> None:
        self.encoder = XLMRSemanticEncoder(model_name)
        self.language_identifier = language_identifier

    @lru_cache(maxsize=8192)
    def _embed_one(self, text: str) -> np.ndarray:
        return self.encoder.encode([normalize_whitespace(text)], batch_size=1)[0]

    def score_response(
        self,
        prompt: str,
        prediction: str,
        reference: str,
        target_spanish_ratio: float | None = None,
        target_switch_type: str | None = None,
    ) -> ResponseJudgeResult:
        prompt_sim = cosine_similarity(self._embed_one(prompt), self._embed_one(prediction))
        ref_sim = cosine_similarity(self._embed_one(reference), self._embed_one(prediction))
        fluency_raw = 1.0 - repeated_token_penalty(prediction)
        length_fit = 1.0 - min(1.0, abs(len(prediction) - len(reference)) / max(1, len(reference)))
        lince_fit = self.language_identifier.lince_style_score(
            prediction,
            target_ratio=target_spanish_ratio,
            target_switch_type=target_switch_type,
        )

        relevance = self._to_five_point(0.7 * ref_sim + 0.3 * prompt_sim)
        fluency = self._to_five_point(fluency_raw)
        completeness = self._to_five_point(0.65 * ref_sim + 0.35 * length_fit)
        alignment = self._to_five_point(ref_sim)
        code_switch_fit = self._to_five_point(lince_fit)
        overall = int(round(np.mean([relevance, fluency, completeness, alignment, code_switch_fit])))
        return ResponseJudgeResult(
            relevance=relevance,
            fluency=fluency,
            completeness=completeness,
            alignment=alignment,
            code_switch_fit=code_switch_fit,
            overall=max(1, min(5, overall)),
        )

    def _to_five_point(self, raw_value: float) -> int:
        clamped = max(0.0, min(1.0, raw_value))
        return max(1, min(5, int(math.ceil(clamped * 5))))
