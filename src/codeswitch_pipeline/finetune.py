from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import FineTuneConfig


def finetune_spanglish_adapter(
    base_model_name: str,
    texts: list[str],
    output_dir: str | Path,
    config: FineTuneConfig,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quant_config,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora)

    dataset = Dataset.from_list([{"text": text} for text in texts if text.strip()])

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_seq_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    training_arg_values = {
        "output_dir": str(output_path),
        "overwrite_output_dir": False,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "bf16": False,
        "save_safetensors": True,
    }
    supported_args = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    filtered_training_args = {
        key: value for key, value in training_arg_values.items() if key in supported_args
    }
    training_args = TrainingArguments(**filtered_training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    resume_checkpoint = _latest_checkpoint(output_path) if config.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    return output_path


def adapter_artifacts_exist(output_dir: str | Path) -> bool:
    output_path = Path(output_dir)
    required = [
        output_path / "adapter_config.json",
        output_path / "adapter_model.safetensors",
    ]
    return all(path.exists() for path in required)


def adapter_matches_base_model(output_dir: str | Path, base_model_name: str) -> bool:
    config_path = Path(output_dir) / "adapter_config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return False
    saved_base = str(data.get("base_model_name_or_path", "")).strip()
    return saved_base == base_model_name


def resolve_adapter_output_dir(output_dir: str | Path, base_model_name: str) -> Path:
    base_path = Path(output_dir)
    if adapter_artifacts_exist(base_path) and adapter_matches_base_model(base_path, base_model_name):
        return base_path
    return base_path / _model_slug(base_model_name)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def _model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name.strip())
    slug = slug.strip("-").lower()
    return slug or "model"
