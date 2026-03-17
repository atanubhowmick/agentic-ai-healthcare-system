"""
CLI wrapper for manually re-training the ClinicalBERT triage classifier.

The orchestrator auto-trains on first startup, so this script is only needed
when you want to re-train with updated data or changed hyperparameters.

Usage (from the orchestrator-agent service root):
    python train_clinicalbert.py
    python train_clinicalbert.py --output-dir ./clinicalbert_router --epochs 10 --batch-size 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agents.clinicalbert_trainer import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-train ClinicalBERT triage classifier")
    parser.add_argument("--output-dir", default="./clinicalbert_router")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--val-split",  type=float, default=0.15)
    args = parser.parse_args()

    train_and_save(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
