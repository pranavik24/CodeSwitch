from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

from .cleaning import clean_generation_text, clean_spanglish_social_text, is_usable_clean_text
from .text_utils import join_tokens, normalize_whitespace, parse_array_string


def load_multiwoz_pairs(
    dataset_id: str,
    splits: Iterable[str],
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    datasets_by_split = _load_multiwoz_dataset(dataset_id=dataset_id, splits=splits)
    dataset = concatenate_datasets(datasets_by_split)

    rows: list[dict[str, object]] = []
    debug_samples: list[dict[str, object]] = []
    for dialogue in dataset:
        dialogue_id = dialogue.get("dialogue_id", "")
        services = "|".join(dialogue.get("services", []))
        turns = _normalize_turns(dialogue.get("turns", []))
        if turns and len(debug_samples) < 3:
            debug_samples.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn_count": len(turns),
                    "first_speakers": [_normalize_speaker(turn.get("speaker")) for turn in turns[:4]],
                    "first_utterances": [_stringify_value(turn.get("utterance", ""))[:80] for turn in turns[:2]],
                }
            )
        for turn_index in range(len(turns) - 1):
            current_turn = turns[turn_index]
            next_turn = turns[turn_index + 1]
            if _normalize_speaker(current_turn.get("speaker")) != "USER":
                continue
            if _normalize_speaker(next_turn.get("speaker")) != "SYSTEM":
                continue
            prompt = clean_generation_text(normalize_whitespace(_stringify_value(current_turn.get("utterance", ""))))
            reply = clean_generation_text(normalize_whitespace(_stringify_value(next_turn.get("utterance", ""))))
            if not prompt or not reply:
                continue
            rows.append(
                {
                    "sample_id": f"{dialogue_id}_{_stringify_value(current_turn.get('turn_id', turn_index))}",
                    "dialogue_id": dialogue_id,
                    "turn_id": _stringify_value(current_turn.get("turn_id", turn_index)),
                    "services": services,
                    "prompt": prompt,
                    "recommended_reply": reply,
                    "source_split": dialogue.get("split", ""),
                }
            )

    if len(rows) < sample_size:
        raise ValueError(
            f"Requested {sample_size} samples but only found {len(rows)} user/system pairs. "
            f"Turn parsing diagnostics: {debug_samples}"
        )

    frame = pd.DataFrame(rows)
    return frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def _normalize_turns(turns: object) -> list[dict[str, object]]:
    turns = _unwrap_singleton(turns)

    if isinstance(turns, list):
        normalized_list: list[dict[str, object]] = []
        for turn in turns:
            if isinstance(turn, dict):
                normalized_list.append({key: _unwrap_singleton(value) for key, value in turn.items()})
        return normalized_list

    if isinstance(turns, dict):
        keys = list(turns.keys())
        if not keys:
            return []

        lengths: list[int] = []
        for key in keys:
            value = turns.get(key)
            if isinstance(value, list):
                lengths.append(len(value))
        if not lengths:
            return []

        normalized: list[dict[str, object]] = []
        total_turns = max(lengths)
        for index in range(total_turns):
            turn: dict[str, object] = {}
            for key in keys:
                value = _unwrap_singleton(turns.get(key))
                if isinstance(value, list):
                    turn[key] = _unwrap_singleton(value[index]) if index < len(value) else None
                else:
                    turn[key] = _unwrap_singleton(value)
            normalized.append(turn)
        return normalized

    return []


def _unwrap_singleton(value: object) -> object:
    while True:
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
            continue
        if isinstance(value, dict) and len(value) == 1:
            only_value = next(iter(value.values()))
            if isinstance(only_value, (list, dict, str, int, float, bool)) or only_value is None:
                value = only_value
                continue
        break
    return value


def _stringify_value(value: object) -> str:
    value = _unwrap_singleton(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_stringify_value(item) for item in value]
        return " ".join(part for part in parts if part)
    return str(value)


def _normalize_speaker(value: object) -> str:
    normalized = _stringify_value(value).strip().upper()
    if normalized in {"USER", "USR", "0"}:
        return "USER"
    if normalized in {"SYSTEM", "SYS", "1", "ASSISTANT"}:
        return "SYSTEM"
    return normalized


def _load_multiwoz_dataset(dataset_id: str, splits: Iterable[str]) -> list:
    split_list = list(splits)
    try:
        return [load_dataset(dataset_id, split=split) for split in split_list]
    except RuntimeError as exc:
        message = str(exc)
        if "Dataset scripts are no longer supported" not in message:
            raise
        return _load_multiwoz_from_parquet(dataset_id=dataset_id, splits=split_list)


def _load_multiwoz_from_parquet(dataset_id: str, splits: list[str]) -> list:
    config_dirs = ["v2.2", "v2.2_active_only"]
    last_error: Exception | None = None

    for config_dir in config_dirs:
        datasets_by_split: list = []
        config_failed = False
        for split in splits:
            split_candidates = _multiwoz_parquet_candidates(
                dataset_id=dataset_id,
                config_dir=config_dir,
                split=split,
            )
            dataset_for_split = None
            split_error: Exception | None = None
            for candidate in split_candidates:
                try:
                    dataset_for_split = load_dataset("parquet", data_files=[candidate], split="train")
                    break
                except Exception as exc:
                    split_error = exc
                    try:
                        frame = pd.read_parquet(candidate)
                        dataset_for_split = Dataset.from_pandas(frame, preserve_index=False)
                        break
                    except Exception as pandas_exc:
                        split_error = pandas_exc
            if dataset_for_split is None:
                last_error = split_error
                config_failed = True
                break
            datasets_by_split.append(dataset_for_split)

        if not config_failed and len(datasets_by_split) == len(splits):
            return datasets_by_split

    raise RuntimeError(
        "Unable to load MultiWOZ from standard Parquet files. "
        "Tried the Hugging Face Parquet exports under 'v2.2' and 'v2.2_active_only', "
        "including the converted Parquet branch."
    ) from last_error


def _multiwoz_parquet_candidates(dataset_id: str, config_dir: str, split: str) -> list[str]:
    file_name = f"multi_woz_v22-{split}.parquet"
    branch = "refs%2Fconvert%2Fparquet"
    return [
        f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{config_dir}/{file_name}",
        f"https://huggingface.co/datasets/{dataset_id}/resolve/{branch}/{config_dir}/{file_name}",
        f"https://huggingface.co/datasets/{dataset_id}/resolve/{branch}/{config_dir}/{split}/0000.parquet",
        f"hf://datasets/{dataset_id}@refs/convert/parquet/{config_dir}/{split}/0000.parquet",
    ]


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
