"""
Standalone script to fine-tune the ClinicalBERT specialist classifier.

Run this OUTSIDE the application - no running service required.
Both CSV files are external (not committed to git) and must be provided via
--classifier-csv and/or --disease-csv. At least one must be supplied.

The trained model is saved to --output-dir. Point the orchestrator service at
it via the CLINICALBERT_MODEL_DIR env var (default: ./clinicalbert_router).

Usage examples
--------------
# Both CSVs (recommended - 6000+ balanced examples)
python services/orchestrator-agent/src/training/train_clinicalbert_classifier.py \\
    --classifier-csv  /data/classifier_training_data.csv \\
    --disease-csv "/data/Disease and symptoms dataset.csv"

# Disease CSV only
python services/orchestrator-agent/src/training/train_clinicalbert_classifier.py \\
    --disease-csv "/data/Disease and symptoms dataset.csv"

# Classifier CSV only (small dataset, useful for quick smoke-test)
python services/orchestrator-agent/src/training/train_clinicalbert_classifier.py \\
    --classifier-csv /data/classifier_training_data.csv

# Full control
python services/orchestrator-agent/src/training/train_clinicalbert_classifier.py \\
    --classifier-csv    /data/classifier_training_data.csv \\
    --disease-csv   "/data/Disease and symptoms dataset.csv" \\
    --output-dir    services/orchestrator-agent/clinicalbert_router \\
    --max-per-label 2000 \\
    --epochs        10 \\
    --batch-size    16

Dependencies (pip install)
--------------------------
    torch transformers scikit-learn accelerate
"""

import argparse
import csv
import logging
import os
import random
import sys

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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("train_classifier")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_MODEL  = "emilyalsentzer/Bio_ClinicalBERT"
_MAX_LENGTH  = 128
_RANDOM_SEED = 42

# Label map — keep in sync with LABEL2ID in core/config.py (used by the runtime).
LABEL2ID: dict[str, int] = {
    "cardiology":         0,
    "neurology":          1,
    "cancer":             2,
    "pathology":          3,
    "gastroenterology":   4,
    "dermatology":        5,
    "orthopedics":        6,
    "pulmonology":        7,
    "urology":            8,
    "endocrinology":      9,
    "psychiatry":         10,
    "ophthalmology":      11,
    "rheumatology":       12,
    "nephrology":         13,
    "gynecology":         14,
    "hematology":         15,
    "infectious_disease": 16,
    "allergy":            17,
    "otolaryngology":     18,
    "unknown":            19,
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Disease → specialist label keyword rules
# Priority: cancer > neurology > cardiology > psychiatry > pulmonology >
#           endocrinology > gastroenterology > nephrology > urology >
#           dermatology > rheumatology > orthopedics > ophthalmology >
#           pathology (strict: blood/lab disorders only) > unknown
# New domains (gynecology, hematology, infectious_disease, allergy,
# otolaryngology) are intentionally left without keyword rules — the
# disease CSV loader will classify unmatched diseases as "unknown" and
# the model learns those labels from whatever examples exist in the data.
# ---------------------------------------------------------------------------

_CANCER_KW = [
    "cancer", "carcinoma", "lymphoma", "leukemia", "melanoma", "sarcoma",
    "malignant", "metastatic", "myeloma", "glioma", "blastoma", "kaposi",
    "ependymoma", "meningioma", "neurofibromatosis", "polycythemia vera",
    "myelodysplastic", "hydatidiform mole",
]
_NEURO_KW = [
    "neuro", "brain", "stroke", "epilepsy", "migraine", "alzheimer",
    "parkinson", "dementia", "multiple sclerosis", "meningitis", "encephalitis",
    "cerebral", "cranial nerve", "transient ischemic", "guillain barre",
    "amyotrophic lateral", "huntington", "friedreich", "spinocerebellar",
    "myasthenia", "myoclonus", "narcolepsy", "hydrocephalus", "moyamoya",
    "pseudotumor cerebri", "restless leg", "tourette", "syringomyelia",
    "tuberous sclerosis", "subarachnoid", "subdural hemorrhage",
    "intracerebral", "intracranial", "trigeminal", "bell palsy",
    "brachial neuritis", "cerebral palsy", "autonomic nervous",
    "normal pressure hydrocephalus", "concussion", "lewy body",
    "spinal cord", "tension headache", "essential tremor",
]
_CARDIO_KW = [
    "heart", "cardiac", "coronary", "arrhythmia", "atrial", "ventricular",
    "cardiomyopathy", "pericarditis", "endocarditis", "aortic", "angina",
    "hypertensive heart", "mitral", "tricuspid", "pulmonic valve",
    "heart failure", "heart attack", "heart block", "myocarditis",
    "paroxysmal supraventricular", "paroxysmal ventricular", "sick sinus",
    "sinus bradycardia", "premature atrial", "premature ventricular",
    "hypertrophic obstructive", "pulmonary hypertension",
    "ischemic heart", "central atherosclerosis", "peripheral arterial",
    "deep vein thrombosis", "pulmonary embolism", "cardiac arrest",
    "congestive",
]
_PSYCHIATRY_KW = [
    "psychiat", "mental health", "depression", "anxiety", "schizophrenia",
    "bipolar", "ptsd", "ocd", "panic disorder", "phobia", "eating disorder",
    "anorexia", "bulimia", "adhd", "autism", "personality disorder",
    "psychosis", "hallucination", "suicid", "insomnia", "sleep disorder",
    "dissociative", "somatoform", "substance abuse", "addiction",
]
_PULMO_KW = [
    "pulmon", "lung", "respiratory", "asthma", "copd", "bronch", "pneumon",
    "pleuris", "emphysema", "tuberculosis", "sleep apnea", "sarcoidosis",
    "interstitial lung", "pneumothorax", "cystic fibrosis", "bronchiectasis",
    "hypoxia", "wheez", "airway", "trachea", "larynx", "vocal cord",
]
_ENDOCRINE_KW = [
    "endocrin", "diabetes", "diabetic", "thyroid", "hypothyroid", "hyperthyroid",
    "adrenal", "pituitary", "cushing", "addison", "insulin resistance",
    "hyperglycemia", "hypoglycemia", "goiter", "hormone", "metabolic syndrome",
    "polycystic ovary", "pcos", "acromegaly", "hyperparathyroid",
    "hypoparathyroid", "obesity endocrine",
]
_GASTRO_KW = [
    "gastro", "stomach", "intestin", "bowel", "colon", "rectal", "hepat",
    "liver", "pancreatic", "gallbladder", "bile", "esophag", "peptic",
    "ulcer", "crohn", "colitis", "irritable bowel", "appendic",
    "gastrointestinal", "gerd", "reflux", "celiac", "diverticular",
    "hemorrhoid", "hernia", "nausea vomiting", "diarrhea", "constipation",
    "cholecystitis", "pancreatitis", "cirrhosis", "hepatitis",
]
_NEPHRO_KW = [
    "nephro", "chronic kidney", "glomerulo", "nephritis", "nephrotic",
    "dialysis", "hemodialysis", "peritoneal dialysis", "renal failure",
    "renal insufficiency", "proteinuria", "kidney disease",
]
_UROLOGY_KW = [
    "urol", "bladder", "prostate", "ureter", "urethra", "urinary",
    "incontinence", "hematuria", "urinary tract", "benign prostatic",
    "erectile", "testicular", "kidney stone", "renal calcul",
    "cystitis", "pyelonephritis", "overactive bladder",
]
_DERMATO_KW = [
    "dermat", "skin", "rash", "eczema", "psoriasis", "acne", "urticaria",
    "hives", "cellulitis", "impetigo", "ringworm", "scabies", "vitiligo",
    "alopecia", "seborrhea", "rosacea", "contact dermatitis", "atopic",
    "pemphigus", "shingles", "herpes zoster", "nail disorder",
    "hyperhidrosis", "pruritus", "wound infection",
]
_RHEUMATO_KW = [
    "rheumat", "lupus", "rheumatoid arthritis", "ankylosing",
    "sjogren", "scleroderma", "vasculitis", "polymyalgia", "raynaud",
    "antiphospholipid", "myositis", "dermatomyositis", "psoriatic arthritis",
    "reactive arthritis", "mixed connective tissue",
]
_ORTHO_KW = [
    "ortho", "fracture", "bone", "joint", "ligament", "tendon",
    "osteo", "scoliosis", "kyphosis", "lumbar", "cervical spine",
    "hip replacement", "knee", "shoulder pain", "elbow", "wrist",
    "ankle", "carpal tunnel", "fibromyalgia", "gout", "bursitis",
    "tendinitis", "dislocation", "sprain", "strain", "back pain",
    "neck pain", "disc herniation", "spinal stenosis",
]
_OPHTHALMO_KW = [
    "ophthalm", "eye", "vision", "glaucoma", "cataract", "retinal",
    "macular", "conjunctivit", "uveitis", "keratitis", "blepharitis",
    "strabismus", "amblyopia", "diabetic retinopathy", "optic neuritis",
    "dry eye", "corneal", "vitreous", "pterygium",
]
# Pathology is STRICT: only genuine blood/haematology/lab disorders
_PATHOLOGY_KW = [
    "anemia", "anaemia", "hemophilia", "haemophilia", "thrombocytopenia",
    "thrombocytosis", "leukocytosis", "thalassemia", "sickle cell",
    "blood disorder", "coagulation disorder", "clotting disorder",
    "hemolytic", "haemolytic", "aplastic", "myelofibrosis",
    "iron deficiency", "vitamin b12 deficiency", "folate deficiency",
    "pernicious", "hematologic", "haematologic", "blood dyscrasia",
    "sepsis", "septicemia", "bacteremia",
]

# Priority order: most specific → least specific
# New domains (gynecology, hematology, infectious_disease, allergy,
# otolaryngology) have no keyword rules — unmatched diseases fall to "unknown".
_DOMAIN_PRIORITY = [
    ("cancer",           _CANCER_KW),
    ("neurology",        _NEURO_KW),
    ("cardiology",       _CARDIO_KW),
    ("psychiatry",       _PSYCHIATRY_KW),
    ("pulmonology",      _PULMO_KW),
    ("endocrinology",    _ENDOCRINE_KW),
    ("gastroenterology", _GASTRO_KW),
    ("nephrology",       _NEPHRO_KW),
    ("urology",          _UROLOGY_KW),
    ("dermatology",      _DERMATO_KW),
    ("rheumatology",     _RHEUMATO_KW),
    ("orthopedics",      _ORTHO_KW),
    ("ophthalmology",    _OPHTHALMO_KW),
    ("pathology",        _PATHOLOGY_KW),
]


def _classify_disease(disease_name: str) -> str:
    d = disease_name.lower()
    for label, keywords in _DOMAIN_PRIORITY:
        if any(kw in d for kw in keywords):
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_synthetic_data(path: str) -> list[tuple[str, str]]:
    """Load the two-column (text, label) CSV."""
    if not os.path.isfile(path):
        log.warning("Synthetic data loader: CSV not found: %s", path)
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(row["text"].strip(), row["label"].strip()) for row in reader
                if row.get("label", "").strip() in LABEL2ID]
    log.info("Synthetic data loader: %d examples from %s", len(rows), path)
    return rows


def load_disease_csv(
    path: str,
    max_per_label: int = 1500,
    seed: int = _RANDOM_SEED,
) -> list[tuple[str, str]]:
    """
    Load the binary symptom/disease CSV and convert to (text, label) tuples.

    - Column 0: 'diseases'
    - Columns 1+: symptom names with 0/1 values

    Each present symptom (value == '1') becomes a comma-separated term in the
    output text. Up to max_per_label rows are sampled per specialist label.
    """
    if not path or not os.path.isfile(path):
        log.warning("Disease CSV not found or not provided: %s", path)
        return []

    buckets: dict[str, list[tuple[str, str]]] = {k: [] for k in LABEL2ID}
    bucket_cap = max_per_label * 5   # over-collect before sampling

    log.info("Loading disease CSV (this may take a moment for large files): %s", path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = _classify_disease(row["diseases"])
            if len(buckets[label]) >= bucket_cap:
                continue
            symptoms = [
                col.replace(".", " ").strip()
                for col, val in row.items()
                if col != "diseases" and val.strip() == "1"
            ]
            if not symptoms:
                continue
            text = "Patient presents with: " + ", ".join(symptoms)
            buckets[label].append((text, label))

    rng = random.Random(seed)
    result: list[tuple[str, str]] = []
    for label, samples in buckets.items():
        rng.shuffle(samples)
        result.extend(samples[:max_per_label])
        log.info("  Disease CSV - %s: %d samples (capped at %d)", label, min(len(samples), max_per_label), max_per_label)

    rng.shuffle(result)
    log.info("Disease CSV loaded: %d examples total", len(result))
    return result


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class _ClassifierDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, str]],
        tokenizer: AutoTokenizer,
        label2id: dict[str, int],
    ) -> None:
        self.samples   = samples
        self.tokenizer = tokenizer
        self.label2id  = label2id

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
        item["labels"] = torch.tensor(self.label2id[label], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    float(np.mean(preds == labels)),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_save(
    classifier_csv: str,
    output_dir: str,
    disease_csv: str = "",
    max_per_label: int = 1500,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 2e-5,
    val_split: float = 0.15,
) -> None:
    synthetic_data   = load_synthetic_data(classifier_csv)
    disease_data  = load_disease_csv(disease_csv, max_per_label=max_per_label)
    all_data      = synthetic_data + disease_data

    if not all_data:
        log.error("No training data found. Aborting.")
        sys.exit(1)

    # Log per-label counts before training
    from collections import Counter
    label_counts = Counter(lbl for _, lbl in all_data)
    log.info("Combined dataset: %d examples | %s", len(all_data),
             " | ".join(f"{k}={v}" for k, v in sorted(label_counts.items())))

    # Build an active label set from labels that actually have training data.
    # Labels declared in LABEL2ID but with 0 examples (e.g. new domains without
    # keyword rules yet) are excluded so they don't add untrained softmax neurons
    # that degrade accuracy for all other labels.
    active_labels = sorted(label_counts.keys(), key=lambda l: LABEL2ID[l])
    inactive = [l for l in LABEL2ID if l not in label_counts]
    if inactive:
        log.warning(
            "The following LABEL2ID entries have 0 training examples and will be "
            "excluded from this training run: %s", inactive
        )
    active_label2id = {lbl: idx for idx, lbl in enumerate(active_labels)}
    active_id2label = {idx: lbl for lbl, idx in active_label2id.items()}
    log.info("Active labels (%d): %s", len(active_label2id), list(active_label2id.keys()))

    rng = random.Random(_RANDOM_SEED)
    rng.shuffle(all_data)

    train_samples, val_samples = train_test_split(
        all_data,
        test_size=val_split,
        stratify=[lbl for _, lbl in all_data],
        random_state=_RANDOM_SEED,
    )
    log.info("Split - Train: %d | Val: %d", len(train_samples), len(val_samples))

    log.info("Loading base model: %s", _BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL,
        num_labels=len(active_label2id),
        id2label=active_id2label,
        label2id=active_label2id,
    )

    train_dataset = _ClassifierDataset(train_samples, tokenizer, active_label2id)
    val_dataset   = _ClassifierDataset(val_samples,   tokenizer, active_label2id)

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
        logging_steps=50,
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

    log.info("Training started...")
    trainer.train()

    # Classification report on validation set
    preds_output = trainer.predict(val_dataset)
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels_arr = preds_output.label_ids
    report = classification_report(labels_arr, preds, target_names=list(active_label2id.keys()))
    log.info("Validation classification report:\n%s", report)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Model saved to: %s", os.path.abspath(output_dir))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_output_dir() -> str:
    # src/training/ -> src/ -> orchestrator-agent/ -> clinicalbert_router/
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "clinicalbert_router",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune ClinicalBERT specialist classifier (standalone, no service required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--classifier-csv",
        default="",
        help="Path to classifier_training_data.csv (synthetic examples). External - not in git.",
    )
    parser.add_argument(
        "--disease-csv",
        default="",
        help="Path to 'Disease and symptoms dataset.csv'. External - not in git.",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument("--max-per-label", type=int,   default=1500,
                        help="Max rows sampled per label from the disease CSV")
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch-size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--val-split",     type=float, default=0.15)
    args = parser.parse_args()

    if not args.classifier_csv and not args.disease_csv:
        parser.error("At least one of --classifier-csv or --disease-csv must be provided.")

    train_and_save(
        classifier_csv = args.classifier_csv,
        output_dir    = args.output_dir,
        disease_csv   = args.disease_csv,
        max_per_label = args.max_per_label,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        val_split     = args.val_split,
    )


if __name__ == "__main__":
    main()
