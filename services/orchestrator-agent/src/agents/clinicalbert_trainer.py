"""
ClinicalBERT fine-tuning logic.

Training data is injected via the `training_data` parameter.
Use the standalone train_triage_classifier.py at the project root to train.
"""

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from core.config import LABEL2ID, ID2LABEL
from log.logger import logger

_BASE_MODEL  = "emilyalsentzer/Bio_ClinicalBERT"
_MAX_LENGTH  = 128
_RANDOM_SEED = 42


class _TriageDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str]], tokenizer: AutoTokenizer) -> None:
        self.samples   = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(LABEL2ID[label], dtype=torch.long)
        return item


def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    float(np.mean(preds == labels)),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def train_and_save(
    output_dir: str,
    training_data: list[tuple[str, str]],
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 2e-5,
    val_split: float = 0.15,
) -> None:
    """Fine-tune Bio_ClinicalBERT on training_data and save to output_dir."""
    if not training_data:
        logger.error("[CLINICALBERT_TRAINER] No training data provided. Aborting.")
        return

    logger.info("[CLINICALBERT_TRAINER] Starting fine-tuning | base=%s | samples=%d | output=%s",
                _BASE_MODEL, len(training_data), output_dir)

    train_samples, val_samples = train_test_split(
        training_data,
        test_size=val_split,
        stratify=[label for _, label in training_data],
        random_state=_RANDOM_SEED,
    )
    logger.info("[CLINICALBERT_TRAINER] Train: %d | Val: %d", len(train_samples), len(val_samples))

    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = _TriageDataset(train_samples, tokenizer)
    val_dataset   = _TriageDataset(val_samples,   tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=_RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Log classification report
    preds_output = trainer.predict(val_dataset)
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    report = classification_report(labels, preds, target_names=list(LABEL2ID.keys()))
    logger.info("[CLINICALBERT_TRAINER] Validation results:\n%s", report)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("[CLINICALBERT_TRAINER] Model saved to: %s", output_dir)
