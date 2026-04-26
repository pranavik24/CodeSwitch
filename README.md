# English-Spanish Code-Switch Pipeline

This project scaffolds an end-to-end pipeline for:

1. Sampling 300 English task prompts from `pfb30/multi_woz_v22`.
2. Building two English-Spanish code-switch datasets from the same 300 base prompts.
3. Saving the original 300 prompts as a control dataset.
4. Fine-tuning a lightweight generator on your local `dataset/spanish_texts/*.csv` Spanglish corpus.
5. Evaluating four LLMs on the resulting datasets with overlap metrics and multilingual judging.

## Expected Outputs

The pipeline writes the files below:

- `outputs/datasets/control_eng.csv`
- `outputs/datasets/unfinetuned_engesp.csv`
- `outputs/datasets/finetuned_engesp.csv`
- `outputs/datasets/unfinetuned_candidates.csv`
- `outputs/datasets/finetuned_candidates.csv`
- `outputs/evaluations/llm_eval_raw.csv`
- `outputs/evaluations/llm_eval_summary.csv`

## Project Layout

```text
configs/
  pipeline.yaml
dataset/
  spa.txt
  spanish_texts/
notebooks/
  Example.ipynb
outputs/
  datasets/
  evaluations/
  models/
scripts/
  run_pipeline.py
src/codeswitch_pipeline/
  config.py
  data_sources.py
  evaluation.py
  finetune.py
  generation.py
  judge.py
  lexicon.py
  metrics.py
  pipeline.py
  text_utils.py
```

## How It Works

### Task sampling

The sampler loads `pfb30/multi_woz_v22` with Hugging Face `datasets`, extracts `USER -> SYSTEM` turn pairs, and randomly samples 300 prompts with their recommended system replies.

### Dataset 1: `unfinetuned_engesp.csv`

- Uses the local English-Spanish translation corpus in `dataset/spa.txt` to build a lexical hint table.
- Uses a lightweight instruction model to rewrite the 300 sampled English prompts into Spanglish.
- Balances prompts across:
  - `10%`, `25%`, `50%`, `75%` Spanish token targets
  - `intra-sentential` and `inter-sentential` switching
- Generates a candidate pool of at least 500 rewrites.
- Scores each rewrite with an XLM-R based rubric judge.
- Accepts only score-5 prompts when `judge.accept_only_score_five: true`.

### Dataset 2: `finetuned_engesp.csv`

- Fine-tunes the base generator with LoRA on the raw Spanglish text in `dataset/spanish_texts/*.csv`.
- Reuses the same 300 sampled prompts and the same switch-balance targets.
- Applies the same 500-candidate generation and score-5 filtering flow.

### Dataset 3: `control_eng.csv`

- Saves the original 300 sampled English prompts and recommended replies.

### Evaluation

The evaluation stage runs four LLMs from `configs/pipeline.yaml` against the datasets and writes:

- per-example outputs in `llm_eval_raw.csv`
- grouped summaries in `llm_eval_summary.csv`

Metrics included:

- XLM-R rubric judge score
- `ROUGE-1`
- `ROUGE-2`
- token-level `F1`
- `LinCE-style` code-switch score

## Important Notes

- The `LinCE-style` score here is a practical code-switch diagnostic built around token-level language ID and switch behavior. It is not the official LinCE shared-task benchmark score, because official LinCE evaluation requires task-specific gold labels.
- The default Llama models in the config are Hugging Face gated models. In Colab, log in with a Hugging Face token before running evaluation or switch to ungated alternatives.
- The XLM-R judge is a multilingual semantic scorer plus code-switch heuristics, not a generative judge.

## Running Locally Or In Colab

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
PYTHONPATH=src python scripts/run_pipeline.py --stage all
```

Run only dataset generation:

```bash
PYTHONPATH=src python scripts/run_pipeline.py --stage datasets
```

Run only evaluation:

```bash
PYTHONPATH=src python scripts/run_pipeline.py --stage evaluate
```

## Sources

- MultiWOZ dataset card: https://huggingface.co/datasets/pfb30/multi_woz_v22
- XLM-RoBERTa model card: https://huggingface.co/FacebookAI/xlm-roberta-base
- CodeSwitch package docs: https://codeswitch.readthedocs.io/
