from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import GenerationConfig
from .generation import HFRewriteGenerator
from .judge import ResponseJudge
from .metrics import LanguageIdentifier, mean, rouge_scores, token_f1


def evaluate_models_on_datasets(
    dataset_paths: list[str | Path],
    eval_model_names: list[str],
    generation_config: GenerationConfig,
    response_judge: ResponseJudge,
    language_identifier: LanguageIdentifier,
    output_raw_csv: str | Path,
    output_summary_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []

    for model_name in eval_model_names:
        generator = HFRewriteGenerator(
            model_name=model_name,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            quantize_4bit=generation_config.quantize_4bit,
        )

        for dataset_path in dataset_paths:
            frame = pd.read_csv(dataset_path)
            for row in frame.to_dict(orient="records"):
                prompt = str(row["prompt"])
                reference = str(row["recommended_reply"])
                prediction = generator.respond(prompt)
                judged = response_judge.score_response(
                    prompt=prompt,
                    prediction=prediction,
                    reference=reference,
                    target_spanish_ratio=float(row.get("target_spanish_ratio", 0.0)),
                    target_switch_type=str(row.get("switch_type", "intra-sentential")),
                )
                rouge = rouge_scores(prediction, reference)
                lince_score = language_identifier.lince_style_score(
                    prediction,
                    target_ratio=float(row.get("target_spanish_ratio", 0.0)),
                    target_switch_type=str(row.get("switch_type", "intra-sentential")),
                )
                raw_rows.append(
                    {
                        "dataset_name": row.get("dataset_name", Path(dataset_path).stem),
                        "model_name": model_name,
                        "sample_id": row["sample_id"],
                        "prompt": prompt,
                        "recommended_reply": reference,
                        "model_response": prediction,
                        "judge_overall": judged.overall,
                        "judge_relevance": judged.relevance,
                        "judge_fluency": judged.fluency,
                        "judge_completeness": judged.completeness,
                        "judge_alignment": judged.alignment,
                        "judge_code_switch_fit": judged.code_switch_fit,
                        "rouge1_f": rouge["rouge1_f"],
                        "rouge2_f": rouge["rouge2_f"],
                        "token_f1": token_f1(prediction, reference),
                        "lince_style_score": lince_score,
                    }
                )

    raw_frame = pd.DataFrame(raw_rows)
    raw_frame.to_csv(Path(output_raw_csv), index=False)

    summary = (
        raw_frame.groupby(["dataset_name", "model_name"], as_index=False)
        .agg(
            judge_overall=("judge_overall", mean),
            judge_relevance=("judge_relevance", mean),
            judge_fluency=("judge_fluency", mean),
            judge_completeness=("judge_completeness", mean),
            judge_alignment=("judge_alignment", mean),
            judge_code_switch_fit=("judge_code_switch_fit", mean),
            rouge1_f=("rouge1_f", mean),
            rouge2_f=("rouge2_f", mean),
            token_f1=("token_f1", mean),
            lince_style_score=("lince_style_score", mean),
        )
    )
    summary.to_csv(Path(output_summary_csv), index=False)
    return raw_frame, summary
