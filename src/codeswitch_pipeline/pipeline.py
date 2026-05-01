from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig, apply_runtime_overrides, load_config
from .data_sources import load_multiwoz_pairs, load_spanglish_corpus, save_cleaned_spanglish_corpus, save_control_dataset
from .evaluation import evaluate_models_on_datasets
from .finetune import adapter_artifacts_exist, finetune_spanglish_adapter, resolve_adapter_output_dir
from .generation import HFRewriteGenerator, build_codeswitch_dataset
from .judge import PromptJudge, ResponseJudge
from .lexicon import TranslationLexicon
from .metrics import LanguageIdentifier
from .text_utils import set_seed


def run_stage(
    stage: str,
    config_path: str | Path = "configs/pipeline.yaml",
    num_samples: int | None = None,
    eval_datasets: list[str] | None = None,
) -> None:
    config = load_config(config_path)
    config = apply_runtime_overrides(
        config,
        num_samples=num_samples,
    )
    set_seed(config.datasets.seed)
    ensure_output_dirs(config)
    if config.datasets.sample_size != config.generation.final_target_size:
        raise ValueError(
            "datasets.sample_size and generation.final_target_size should match for this pipeline "
            f"because the same 300 sampled prompts are reused across all datasets. Got "
            f"{config.datasets.sample_size} and {config.generation.final_target_size}."
        )

    if stage in {"sample", "datasets", "all", "evaluate"}:
        base_samples = load_multiwoz_pairs(
            dataset_id=config.datasets.multiwoz_dataset_id,
            splits=config.datasets.multiwoz_splits,
            sample_size=config.datasets.sample_size,
            seed=config.datasets.seed,
        )
        save_control_dataset(base_samples, config.resolve(config.datasets.control_output_csv))
    else:
        base_samples = None

    if stage in {"datasets", "all", "evaluate", "finetune"}:
        natural_texts = load_spanglish_corpus(
            [config.resolve(path) for path in config.datasets.spanglish_paths],
            text_limit=config.finetune.train_text_limit,
        )
        save_cleaned_spanglish_corpus(natural_texts, config.resolve(config.datasets.cleaned_spanglish_output_csv))

        if stage in {"finetune", "datasets", "all"}:
            adapter_path = resolve_adapter_output_dir(
                config.resolve(config.finetune.adapter_output_dir),
                config.generation.finetune_base_model,
            )
            if stage == "finetune" or not adapter_artifacts_exist(adapter_path):
                adapter_path = finetune_spanglish_adapter(
                    base_model_name=config.generation.finetune_base_model,
                    texts=natural_texts,
                    output_dir=adapter_path,
                    config=config.finetune,
                )

        if stage in {"datasets", "all"}:
            lexicon = TranslationLexicon.from_tatoeba(config.resolve(config.datasets.translation_path))
            language_identifier = LanguageIdentifier(translation_lexicon=lexicon.token_map)
            prompt_judge = PromptJudge(
                model_name=config.judge.xlmr_model,
                natural_reference_texts=natural_texts,
                language_identifier=language_identifier,
                reference_limit=config.judge.naturalness_reference_limit,
            )
            base_generator = HFRewriteGenerator(
                model_name=config.generation.base_generator_model,
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
                quantize_4bit=config.generation.quantize_4bit,
            )
            build_codeswitch_dataset(
                base_samples=base_samples,
                output_csv=config.resolve(config.generation.unfinetuned_output_csv),
                candidate_csv=config.resolve(config.generation.unfinetuned_candidates_csv),
                generator=base_generator,
                judge=prompt_judge,
                lexicon=lexicon,
                natural_texts=natural_texts,
                language_identifier=language_identifier,
                max_attempts_per_prompt=config.generation.max_attempts_per_prompt,
                seed=config.datasets.seed,
                dataset_name="unfinetuned_engesp",
                switch_types=config.generation.switch_types,
                switch_ratios=config.generation.switch_ratios,
            )

            finetuned_generator = HFRewriteGenerator(
                model_name=config.generation.finetune_base_model,
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p,
                quantize_4bit=config.generation.quantize_4bit,
                adapter_path=str(adapter_path),
            )
            build_codeswitch_dataset(
                base_samples=base_samples,
                output_csv=config.resolve(config.generation.finetuned_output_csv),
                candidate_csv=config.resolve(config.generation.finetuned_candidates_csv),
                generator=finetuned_generator,
                judge=prompt_judge,
                lexicon=lexicon,
                natural_texts=natural_texts,
                language_identifier=language_identifier,
                max_attempts_per_prompt=config.generation.max_attempts_per_prompt,
                seed=config.datasets.seed + 1,
                dataset_name="finetuned_engesp",
                switch_types=config.generation.switch_types,
                switch_ratios=config.generation.switch_ratios,
            )

        if stage in {"evaluate", "all"}:
            lexicon = TranslationLexicon.from_tatoeba(config.resolve(config.datasets.translation_path))
            language_identifier = LanguageIdentifier(translation_lexicon=lexicon.token_map)
            response_judge = ResponseJudge(config.judge.xlmr_model, language_identifier)
            available_dataset_paths = {
                "control_eng": config.resolve(config.datasets.control_output_csv),
                "unfinetuned_engesp": config.resolve(config.generation.unfinetuned_output_csv),
                "finetuned_engesp": config.resolve(config.generation.finetuned_output_csv),
            }
            selected_dataset_names = eval_datasets or list(available_dataset_paths.keys())
            unknown_dataset_names = [name for name in selected_dataset_names if name not in available_dataset_paths]
            if unknown_dataset_names:
                valid_names = ", ".join(sorted(available_dataset_paths.keys()))
                raise ValueError(
                    f"Unknown eval_datasets value(s): {', '.join(unknown_dataset_names)}. "
                    f"Valid options are: {valid_names}."
                )
            dataset_paths = [available_dataset_paths[name] for name in selected_dataset_names]
            evaluate_models_on_datasets(
                dataset_paths=dataset_paths,
                eval_model_names=config.generation.eval_models,
                generation_config=config.generation,
                response_judge=response_judge,
                language_identifier=language_identifier,
                output_raw_csv=config.resolve(config.evaluation.output_raw_csv),
                output_summary_csv=config.resolve(config.evaluation.output_summary_csv),
            )


def ensure_output_dirs(config: PipelineConfig) -> None:
    paths = [
        config.resolve(config.datasets.control_output_csv).parent,
        config.resolve(config.datasets.cleaned_spanglish_output_csv).parent,
        config.resolve(config.generation.unfinetuned_output_csv).parent,
        config.resolve(config.evaluation.output_raw_csv).parent,
        config.resolve(config.finetune.adapter_output_dir),
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
