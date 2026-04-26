from __future__ import annotations

import random
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .cleaning import clean_generation_text
from .judge import PromptJudge
from .lexicon import TranslationLexicon
from .metrics import LanguageIdentifier
from .text_utils import balanced_assignments, normalize_whitespace, tokenize_text

FIXED_REFERENCE_EXAMPLES: dict[str, list[dict[str, object]]] = {
    "intra-sentential": [
        {
            "target_ratio": 0.10,
            "text": "Hi, I need a hotel cerca del airport for tomorrow night.",
        },
        {
            "target_ratio": 0.25,
            "text": "Can you help me find a restaurant barato near downtown for dinner tonight?",
        },
        {
            "target_ratio": 0.50,
            "text": "I need un taxi para the train station at ocho because my friend llega tonight.",
        },
        {
            "target_ratio": 0.75,
            "text": "Necesito cambiar my reservation porque el check-in time es too late para mi familia.",
        },
    ],
    "inter-sentential": [
        {
            "target_ratio": 0.10,
            "text": "I need a taxi to the hospital. Por favor, send it as soon as you can.",
        },
        {
            "target_ratio": 0.25,
            "text": "Can you book a table for four tonight? Quiero que sea cerca del centro.",
        },
        {
            "target_ratio": 0.50,
            "text": "I need a hotel with free parking. Y tambien quiero desayuno incluido.",
        },
        {
            "target_ratio": 0.75,
            "text": "Necesito un restaurante para manana por la noche. It should still be close to the station.",
        },
    ],
}

SWITCH_TYPE_DEFINITIONS = {
    "intra-sentential": (
        "Mix English and Spanish within the same sentence or clause. "
        "Both languages should appear inside a single sentence."
    ),
    "inter-sentential": (
        "Switch languages across sentence boundaries. "
        "Use at least two short sentences, with English dominant in one and Spanish dominant in another."
    ),
}


class HFRewriteGenerator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        quantize_4bit: bool = True,
        adapter_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if quantize_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quant_config,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def rewrite(self, prompt: str, switch_type: str, target_ratio: float, lexicon_hints: dict[str, list[str]], examples: list[str]) -> str:
        instruction = self._build_instruction(prompt, switch_type, target_ratio, lexicon_hints, examples)
        messages = [
            {
                "role": "system",
                "content": (
                    "You create natural English-Spanish code-switched user requests for dialogue datasets. "
                    "Preserve meaning, entities, domain details, politeness, and user intent. "
                    "Return only the rewritten user request."
                ),
            },
            {"role": "user", "content": instruction},
        ]
        decoded = self._generate_from_messages(messages, temperature=self.temperature)
        return self._clean_generation(decoded, prompt)

    def respond(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful dialogue assistant. Give a direct user-facing reply that is concise, "
                    "relevant, and natural for a task-oriented conversation."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        decoded = self._generate_from_messages(messages, temperature=0.2)
        return self._clean_generation(decoded, prompt)

    def _build_instruction(
        self,
        prompt: str,
        switch_type: str,
        target_ratio: float,
        lexicon_hints: dict[str, list[str]],
        examples: list[str],
    ) -> str:
        hints = ", ".join(f"{token} -> {'/'.join(candidates)}" for token, candidates in lexicon_hints.items()) or "No lexicon hints."
        fixed_examples = select_fixed_examples(switch_type=switch_type, target_ratio=target_ratio)
        example_block = "\n".join(
            f"- {example['text']} (target Spanish ratio: {int(float(example['target_ratio']) * 100)}%)"
            for example in fixed_examples
        )
        percentage = int(target_ratio * 100)
        switch_definition = SWITCH_TYPE_DEFINITIONS.get(switch_type, "Use natural English-Spanish code-switching.")
        return (
            f"Rewrite the English prompt into natural Spanglish.\n"
            f"Prompt: {prompt}\n"
            f"Switch type: {switch_type}\n"
            f"Switch type definition: {switch_definition}\n"
            f"Target Spanish token percentage: about {percentage}%\n"
            f"Lexicon hints: {hints}\n"
            f"Fixed high-quality Spanglish references:\n{example_block}\n"
            "Rules:\n"
            "1. Keep all factual constraints, slot values, places, numbers, and intents unchanged.\n"
            "2. Make the output sound like a real user request, not a literal translation.\n"
            "3. The output must contain both English and Spanish. Do not produce fully English or fully Spanish text.\n"
            "4. Match the requested switch type closely.\n"
            "5. Use only one line unless the switch type is inter-sentential.\n"
            "6. Do not explain the rewrite.\n"
        )

    def _clean_generation(self, raw_text: str, original_prompt: str) -> str:
        first_line = normalize_whitespace(raw_text.splitlines()[0] if raw_text.strip() else "")
        cleaned = re.sub(r"^(assistant:|rewrite:)\s*", "", first_line, flags=re.IGNORECASE)
        cleaned = clean_generation_text(cleaned)
        return cleaned or original_prompt

    def _generate_from_messages(self, messages: list[dict[str, str]], temperature: float) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages) + "\nASSISTANT:"
        tokens = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **tokens,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(generated[0][tokens["input_ids"].shape[1] :], skip_special_tokens=True)


def lexical_rewrite(
    prompt: str,
    switch_type: str,
    target_ratio: float,
    lexicon: TranslationLexicon,
) -> str:
    tokens = tokenize_text(prompt)
    alpha_indices = [idx for idx, token in enumerate(tokens) if token.isalpha()]
    candidate_indices = [idx for idx in alpha_indices if lexicon.get(tokens[idx])]
    if not candidate_indices:
        return prompt

    target_replacements = max(1, round(len(alpha_indices) * target_ratio))
    target_replacements = min(target_replacements, len(candidate_indices))

    ordered = _ordered_candidate_indices(tokens, candidate_indices, switch_type, target_replacements)
    replaced = 0
    for index in ordered:
        token = tokens[index]
        candidates = lexicon.get(token)
        if not candidates:
            continue
        tokens[index] = _match_case(token, candidates[0])
        replaced += 1
        if replaced >= target_replacements:
            break

    if switch_type == "inter-sentential":
        tokens = _ensure_inter_sentential_shape(tokens)

    text = ""
    for token in tokens:
        if not text:
            text = token
        elif re.fullmatch(r"[.,!?;:]", token):
            text += token
        else:
            text += " " + token
    return clean_generation_text(text)


def build_codeswitch_dataset(
    base_samples: pd.DataFrame,
    output_csv: str | Path,
    candidate_csv: str | Path,
    generator: HFRewriteGenerator | None,
    judge: PromptJudge,
    lexicon: TranslationLexicon,
    natural_texts: list[str],
    language_identifier: LanguageIdentifier,
    max_attempts_per_prompt: int,
    seed: int,
    dataset_name: str,
    min_overall_score: int,
    switch_types: list[str] | None = None,
    switch_ratios: list[float] | None = None,
) -> pd.DataFrame:
    rng = random.Random(seed)
    assignments = balanced_assignments(
        item_ids=base_samples["sample_id"].tolist(),
        switch_types=switch_types or ["intra-sentential", "inter-sentential"],
        switch_ratios=switch_ratios or [0.10, 0.25, 0.50, 0.75],
        seed=seed,
    )
    assignment_frame = pd.DataFrame(assignments, columns=["sample_id", "switch_type", "target_spanish_ratio"])
    frame = base_samples.merge(assignment_frame, on="sample_id", how="left")

    accepted_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    failed_sample_ids: list[str] = []

    for row in frame.to_dict(orient="records"):
        prompt = str(row["prompt"])
        switch_type = str(row["switch_type"])
        target_ratio = float(row["target_spanish_ratio"])
        hints = lexicon.candidates_for_text(prompt)
        best_candidate: dict[str, object] | None = None

        for attempt in range(1, max_attempts_per_prompt + 1):
            if generator is not None:
                try:
                    rewritten = generator.rewrite(prompt, switch_type, target_ratio, hints, [])
                except Exception:
                    rewritten = lexical_rewrite(prompt, switch_type, target_ratio, lexicon)
            else:
                rewritten = lexical_rewrite(prompt, switch_type, target_ratio, lexicon)

            observed_profile = language_identifier.profile(rewritten)
            if not observed_profile.has_codeswitch:
                fallback_rewrite = lexical_rewrite(prompt, switch_type, target_ratio, lexicon)
                fallback_profile = language_identifier.profile(fallback_rewrite)
                if fallback_profile.has_codeswitch:
                    rewritten = fallback_rewrite

            judged = judge.score_prompt(prompt, rewritten, target_ratio, switch_type)
            candidate = {
                **row,
                "dataset_name": dataset_name,
                "attempt": attempt,
                "original_prompt": prompt,
                "rewritten_prompt": rewritten,
                "observed_spanish_ratio": judged.observed_spanish_ratio,
                "observed_switch_type": judged.observed_switch_type,
                **asdict(judged),
            }
            candidate_rows.append(candidate)

            if best_candidate is None or int(candidate["overall"]) > int(best_candidate["overall"]):
                best_candidate = candidate

            if int(candidate["overall"]) >= min_overall_score:
                accepted_rows.append(candidate)
                break
        else:
            if best_candidate is None:
                failed_sample_ids.append(str(row["sample_id"]))
            elif min_overall_score <= 4:
                accepted_rows.append(best_candidate)
            else:
                failed_sample_ids.append(str(row["sample_id"]))

    accepted = pd.DataFrame(accepted_rows)
    if failed_sample_ids:
        pd.DataFrame(candidate_rows).to_csv(Path(candidate_csv), index=False)
        raise ValueError(
            f"Some prompts never reached judge score {min_overall_score}. "
            "Increase max_attempts_per_prompt or adjust the generator. "
            f"Failed sample_ids: {', '.join(failed_sample_ids[:10])}"
        )
    accepted = accepted.drop(columns=["prompt"], errors="ignore")
    accepted = accepted.rename(columns={"rewritten_prompt": "prompt"})
    accepted["lince_prompt_score"] = accepted.apply(
        lambda item: language_identifier.lince_style_score(
            item["prompt"],
            target_ratio=float(item["target_spanish_ratio"]),
            target_switch_type=str(item["switch_type"]),
        ),
        axis=1,
    )
    accepted = accepted[
        [
            "sample_id",
            "dialogue_id",
            "turn_id",
            "services",
            "dataset_name",
            "switch_type",
            "target_spanish_ratio",
            "observed_spanish_ratio",
            "original_prompt",
            "prompt",
            "recommended_reply",
            "naturalness",
            "code_switching_quality",
            "grammar_fluency",
            "conversation_realism",
            "emotion_consistency",
            "overall",
            "lince_prompt_score",
        ]
    ]
    accepted.to_csv(Path(output_csv), index=False)
    pd.DataFrame(candidate_rows).to_csv(Path(candidate_csv), index=False)
    return accepted


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def _ordered_candidate_indices(
    tokens: list[str],
    candidate_indices: list[int],
    switch_type: str,
    target_replacements: int,
) -> list[int]:
    if switch_type != "inter-sentential":
        return candidate_indices

    sentence_groups = _sentence_candidate_groups(tokens, candidate_indices)
    if len(sentence_groups) >= 2:
        ranked_groups = sorted(
            sentence_groups,
            key=lambda group: abs(len(group) - target_replacements),
        )
        ordered: list[int] = []
        for index in ranked_groups[0]:
            ordered.append(index)
        for group in sentence_groups:
            if group is ranked_groups[0]:
                continue
            for index in group:
                ordered.append(index)
        return ordered
    return candidate_indices


def _sentence_candidate_groups(tokens: list[str], candidate_indices: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    candidate_set = set(candidate_indices)
    for index, token in enumerate(tokens):
        if index in candidate_set:
            current.append(index)
        if token in {".", "!", "?"}:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    return groups


def _ensure_inter_sentential_shape(tokens: list[str]) -> list[str]:
    if any(token in {".", "!", "?"} for token in tokens[:-1]):
        return tokens

    split_markers = {",", "and", "but", "because", "so", "then"}
    split_index = None
    midpoint = len(tokens) // 2
    for offset in range(len(tokens)):
        left = midpoint - offset
        right = midpoint + offset
        for candidate in (left, right):
            if 0 < candidate < len(tokens) - 1 and str(tokens[candidate]).lower() in split_markers:
                split_index = candidate
                break
        if split_index is not None:
            break

    if split_index is None:
        return tokens

    new_tokens = tokens[:]
    marker = new_tokens[split_index]
    if str(marker).lower() == ",":
        new_tokens[split_index] = "."
    else:
        new_tokens[split_index] = "."
    next_index = split_index + 1
    if next_index < len(new_tokens):
        new_tokens[next_index] = _capitalize_token(new_tokens[next_index])
    return new_tokens


def _capitalize_token(token: str) -> str:
    if not token:
        return token
    if token[0].isalpha():
        return token[0].upper() + token[1:]
    return token


def select_fixed_examples(switch_type: str, target_ratio: float) -> list[dict[str, object]]:
    examples = FIXED_REFERENCE_EXAMPLES.get(switch_type, [])
    if not examples:
        return []
    ranked = sorted(
        examples,
        key=lambda example: abs(float(example["target_ratio"]) - float(target_ratio)),
    )
    return ranked[:2]
