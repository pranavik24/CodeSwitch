from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_dataset

from .cleaning import clean_generation_text, clean_spanglish_social_text, is_usable_clean_text
from .text_utils import join_tokens, normalize_whitespace, parse_array_string


def load_multiwoz_pairs(
    dataset_id: str,
    splits: Iterable[str],
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    datasets_by_split = [
        load_dataset(dataset_id, split=split, trust_remote_code=True) for split in splits
    ]
    dataset = concatenate_datasets(datasets_by_split)

    rows: list[dict[str, object]] = []
    for dialogue in dataset:
        dialogue_id = dialogue.get("dialogue_id", "")
        services = "|".join(dialogue.get("services", []))
        turns = dialogue.get("turns", [])
        for turn_index in range(len(turns) - 1):
            current_turn = turns[turn_index]
            next_turn = turns[turn_index + 1]
            if current_turn.get("speaker") != "USER" or next_turn.get("speaker") != "SYSTEM":
                continue
            prompt = clean_generation_text(normalize_whitespace(current_turn.get("utterance", "")))
            reply = clean_generation_text(normalize_whitespace(next_turn.get("utterance", "")))
            if not prompt or not reply:
                continue
            rows.append(
                {
                    "sample_id": f"{dialogue_id}_{current_turn.get('turn_id', turn_index)}",
                    "dialogue_id": dialogue_id,
                    "turn_id": current_turn.get("turn_id", turn_index),
                    "services": services,
                    "prompt": prompt,
                    "recommended_reply": reply,
                    "source_split": dialogue.get("split", ""),
                }
            )

    if len(rows) < sample_size:
        raise ValueError(f"Requested {sample_size} samples but only found {len(rows)} user/system pairs.")

    frame = pd.DataFrame(rows)
    return frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def save_control_dataset(frame: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    control = frame.copy()
    control["original_prompt"] = control["prompt"]
    control["dataset_name"] = "control_eng"
    control["switch_type"] = "none"
    control["target_spanish_ratio"] = 0.0
    control["observed_spanish_ratio"] = 0.0
    control.to_csv(output, index=False)
    return output


def load_spanglish_corpus(csv_paths: Iterable[str | Path], text_limit: int | None = None) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        for raw_words in frame["words"].fillna(""):
            tokens = parse_array_string(raw_words)
            text = clean_spanglish_social_text(normalize_whitespace(join_tokens(tokens)))
            if text and is_usable_clean_text(text) and text not in seen:
                texts.append(text)
                seen.add(text)
            if text_limit is not None and len(texts) >= text_limit:
                return texts
    return texts


def save_cleaned_spanglish_corpus(texts: list[str], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": texts}).to_csv(output, index=False)
    return output
