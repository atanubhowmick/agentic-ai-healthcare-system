"""
Loads triage training data from triage_training_data.csv.

The CSV must have two columns (no header quoting needed):
    text,label

Labels: "cardiology" | "neurology" | "cancer" | "pathology"

To replace with real data: edit triage_training_data.csv only — no Python changes needed.
"""

import csv
import os

LABEL2ID: dict[str, int] = {
    "cardiology": 0,
    "neurology":  1,
    "cancer":     2,
    "pathology":  3,
}

ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

_CSV_PATH = os.path.join(os.path.dirname(__file__), "triage_training_data.csv")


def _load_csv(path: str) -> list[tuple[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(row["text"].strip(), row["label"].strip()) for row in reader]


TRIAGE_TRAINING_DATA: list[tuple[str, str]] = _load_csv(_CSV_PATH)
