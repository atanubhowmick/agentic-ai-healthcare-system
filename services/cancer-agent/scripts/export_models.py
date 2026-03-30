"""
Export trained TF-IDF + classifier models to a pickle file
for use by the XAI validation service's SHAP explainer.

Usage:
    python export_models.py [output_path]

Default output: <project_root>/trained_model/cancer_agent_models.pkl
The XAI validation service reads from the same location automatically.

Models exported:
    severity       - LogisticRegression  (tfidf_doc, tfidf_icd, clf)
    emergency      - HistGBM + SVD       (tfidf_doc, tfidf_icd, tfidf_cc, svd, clf, threshold)
    hospitalization- HistGBM + SVD       (tfidf_doc, tfidf_icd, tfidf_cc, svd, clf, threshold)
    cancer_type    - CalibratedClassifierCV(LinearSVC)  (tfidf_doc, tfidf_icd, clf)
"""

import os
import pickle
import sys
from pathlib import Path

# Allow running from the scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

# Project root is 4 levels up from services/cancer-agent/scripts/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_OUTPUT = _PROJECT_ROOT / "trained_model" / "cancer_agent_models.pkl"

from rag.tfidf_predictor import _predictor  # triggers singleton creation
from log.logger import logger


def export_models(output_path: str) -> None:
    print(f"[EXPORT] Training models from MongoDB (this may take a few minutes)...")
    _predictor.ensure_trained()

    models: dict = {}

    if _predictor._clf_sev is not None:
        models["severity"] = {
            "tfidf_doc": _predictor._tfidf_doc_sev,
            "tfidf_icd": _predictor._tfidf_icd_sev,
            "clf":       _predictor._clf_sev,
        }
        print(f"[EXPORT] severity model ready | "
              f"vocab={len(_predictor._tfidf_doc_sev.vocabulary_)} tokens")

    if _predictor._clf_emg is not None:
        models["emergency"] = {
            "tfidf_doc": _predictor._tfidf_doc_emg,
            "tfidf_icd": _predictor._tfidf_icd_emg,
            "tfidf_cc":  _predictor._tfidf_cc_emg,
            "svd":       _predictor._svd_emg,
            "clf":       _predictor._clf_emg,
            "threshold": _predictor._thresh_emg,
        }
        print(f"[EXPORT] emergency model ready | threshold={_predictor._thresh_emg:.4f}")

    if _predictor._clf_hosp is not None:
        models["hospitalization"] = {
            "tfidf_doc": _predictor._tfidf_doc_hosp,
            "tfidf_icd": _predictor._tfidf_icd_hosp,
            "tfidf_cc":  _predictor._tfidf_cc_hosp,
            "svd":       _predictor._svd_hosp,
            "clf":       _predictor._clf_hosp,
            "threshold": _predictor._thresh_hosp,
        }
        print(f"[EXPORT] hospitalization model ready | threshold={_predictor._thresh_hosp:.4f}")

    if _predictor._clf_can is not None:
        models["cancer_type"] = {
            "tfidf_doc": _predictor._tfidf_doc_can,
            "tfidf_icd": _predictor._tfidf_icd_can,
            "clf":       _predictor._clf_can,
        }
        print(f"[EXPORT] cancer_type model ready")

    if not models:
        print("[EXPORT] ERROR: No models were trained — is MongoDB populated?")
        sys.exit(1)

    with open(output_path, "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[EXPORT] Saved {len(models)} models → {output_path} ({size_mb:.1f} MB)")
    print(f"[EXPORT] XAI validation service will load from: {output_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(_DEFAULT_OUTPUT)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    export_models(out)
