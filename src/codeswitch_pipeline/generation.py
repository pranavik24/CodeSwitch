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
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
        example_block = "\n".join(f"- {example}" for example in examples[:2]) or "- No extra examples."
        percentage = int(target_ratio * 100)
        return (
            f"Rewrite the English prompt into natural Spanglish.\n"
            f"Prompt: {prompt}\n"
            f"Switch type: {switch_type}\n"
            f"Target Spanish token percentage: about {percentage}%\n"
            f"Lexicon hints: {hints}\n"
            f"Natural Spanglish references:\n{example_block}\n"
            "Rules:\n"
            "1. Keep all factual constraints, slot values, places, numbers, and intents unchanged.\n"
            "2. Make the output sound like a real user request, not a literal translation.\n"
            "3. Use only one line.\n"
            "4. Do not explain the rewrite.\n"
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
    alpha_indices = [idx for idx, token in enumerate(tokens) if token.isalpha() and lexicon.get(token)]
    if not alpha_indices:
        return prompt

    target_replacements = max(1, round(len([token for token in tokens if token.isalpha()]) * target_ratio))
    if switch_type == "inter-sentential":
        midpoint = len(alpha_indices) // 2
        ordered = alpha_indices[midpoint:] + alpha_indices[:midpoint]
    else:
        ordered = alpha_indices

    replaced = 0
    for index in ordered:
        token = tokens[index]
        candidates = lexicon.get(token)
        if not candidates:
            continue
        tokens[index] = candidates[0]
        replaced += 1
        if replaced >= target_replacements:
            break

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
    require_score_five: bool = True,
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
            examples = rng.sample(natural_texts, k=min(2, len(natural_texts))) if natural_texts else []
            if generator is not None:
                try:
                    rewritten = generator.rewrite(prompt, switch_type, target_ratio, hints, examples)
                except Exception:
                    rewritten = lexical_rewrite(prompt, switch_type, target_ratio, lexicon)
            else:
                rewritten = lexical_rewrite(prompt, switch_type, target_ratio, lexicon)

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

            if (not require_score_five and int(candidate["overall"]) >= 4) or int(candidate["overall"]) == 5:
                accepted_rows.append(candidate)
                break
        else:
            if best_candidate is None:
                failed_sample_ids.append(str(row["sample_id"]))
            elif not require_score_five:
                accepted_rows.append(best_candidate)
            else:
                failed_sample_ids.append(str(row["sample_id"]))

    accepted = pd.DataFrame(accepted_rows)
    if failed_sample_ids:
        pd.DataFrame(candidate_rows).to_csv(Path(candidate_csv), index=False)
        raise ValueError(
            "Some prompts never reached judge score 5. Increase max_attempts_per_prompt or adjust the generator. "
            f"Failed sample_ids: {', '.join(failed_sample_ids[:10])}"
        )
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
