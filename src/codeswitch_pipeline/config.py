from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    multiwoz_dataset_id: str = "pfb30/multi_woz_v22"
    multiwoz_splits: list[str] = field(default_factory=lambda: ["train", "validation", "test"])
    sample_size: int = 300
    seed: int = 42
    translation_path: str = "dataset/spa.txt"
    spanglish_paths: list[str] = field(
        default_factory=lambda: [
            "dataset/spanish_texts/lid_spaeng_train.csv",
            "dataset/spanish_texts/lid_spaeng_validation.csv",
            "dataset/spanish_texts/lid_spaeng_test.csv",
        ]
    )
    control_output_csv: str = "outputs/datasets/control_eng.csv"
    cleaned_spanglish_output_csv: str = "outputs/datasets/cleaned_spanglish_corpus.csv"


@dataclass
class GenerationConfig:
    final_target_size: int = 300
    switch_ratios: list[float] = field(default_factory=lambda: [0.10, 0.25, 0.50, 0.75])
    switch_types: list[str] = field(default_factory=lambda: ["intra-sentential", "inter-sentential"])
    unfinetuned_output_csv: str = "outputs/datasets/unfinetuned_engesp.csv"
    finetuned_output_csv: str = "outputs/datasets/finetuned_engesp.csv"
    unfinetuned_candidates_csv: str = "outputs/datasets/unfinetuned_candidates.csv"
    finetuned_candidates_csv: str = "outputs/datasets/finetuned_candidates.csv"
    base_generator_model: str = "Qwen/Qwen2.5-7B-Instruct"
    finetune_base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    eval_models: list[str] = field(
        default_factory=lambda: [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ]
    )
    max_attempts_per_prompt: int = 8
    max_new_tokens: int = 96
    temperature: float = 0.7
    top_p: float = 0.9
    quantize_4bit: bool = True
    unfinetuned_min_overall_score: int = 4
    finetuned_min_overall_score: int = 5


@dataclass
class FineTuneConfig:
    adapter_output_dir: str = "outputs/models/spanglish_adapter"
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    train_text_limit: int | None = 12000
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 10
    resume_from_checkpoint: bool = True


@dataclass
class JudgeConfig:
    xlmr_model: str = "FacebookAI/xlm-roberta-base"
    naturalness_reference_limit: int = 2048
    accept_only_score_five: bool = True


@dataclass
class EvaluationConfig:
    output_raw_csv: str = "outputs/evaluations/llm_eval_raw.csv"
    output_summary_csv: str = "outputs/evaluations/llm_eval_summary.csv"
    judge_batch_size: int = 16


@dataclass
class PipelineConfig:
    project_root: str = "."
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def resolve(self, value: str) -> Path:
        return Path(self.project_root).joinpath(value).resolve()


def _merge_dataclass(cls: type[Any], values: dict[str, Any] | None) -> Any:
    values = values or {}
    return cls(**values)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    project_root = str(config_path.resolve().parent.parent)
    return PipelineConfig(
        project_root=raw.get("project_root", project_root),
        datasets=_merge_dataclass(DatasetConfig, raw.get("datasets")),
        generation=_merge_dataclass(GenerationConfig, raw.get("generation")),
        finetune=_merge_dataclass(FineTuneConfig, raw.get("finetune")),
        judge=_merge_dataclass(JudgeConfig, raw.get("judge")),
        evaluation=_merge_dataclass(EvaluationConfig, raw.get("evaluation")),
    )


def apply_runtime_overrides(
    config: PipelineConfig,
    num_samples: int | None = None,
) -> PipelineConfig:
    if num_samples is not None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive. Got {num_samples}.")
        config.datasets.sample_size = num_samples
        config.generation.final_target_size = num_samples

    return config
