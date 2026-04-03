"""
Diagnosis explainability module.

SHAP-based implementation (active when cancer_agent_models.pkl is present):
  - Severity  (LogisticRegression): linear attribution — feature_value × coef
  - Emergency (HistGBM + TruncatedSVD): TreeSHAP values projected back through SVD
    to recover word-level importances
  - Cancer type (CalibratedClassifierCV→LinearSVC): linear attribution from
    the first fold's underlying LinearSVC coefs

Fallback to LLM-based explanation when:
  - Model file not found (xai-validation-service/models/cancer_agent_models.pkl)
  - Any SHAP computation error

Model file is produced by:
  services/cancer-agent/scripts/export_models.py
"""

import json
import os
import pickle
import threading
from pathlib import Path
from typing import Any

import shap
import numpy as np
from scipy.sparse import csr_matrix, hstack
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.config import OPENAI_MODEL
from log.logger import logger


# ---------------------------------------------------------------------------
# Model file location
# ---------------------------------------------------------------------------

# Project root is 4 levels up from xai-validation-service/src/explainers/
_MODEL_FILE = Path(__file__).resolve().parent.parent.parent.parent / "trained_model" / "cancer_agent_models.pkl"


# ---------------------------------------------------------------------------
# LLM fallback (unchanged from original implementation)
# ---------------------------------------------------------------------------

_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

_SYSTEM_PROMPT = """You are a clinical AI explainability specialist.
Given a patient's symptoms and the resulting diagnosis, identify the top 3 clinical
factors that most contributed to the diagnosis decision.

Respond ONLY with valid JSON (no markdown fences):
{
    "top_factors": [
        {"factor": "Cardiac arrest possibility", "importance": 0.92, "direction": "increases_risk"},
        {"factor": "Infection in the stomach", "importance": 0.78, "direction": "increases_risk"},
        {"factor": "Normal ECG baseline", "importance": 0.45, "direction": "decreases_risk"}
    ]
}
direction must be one of: increases_risk, decreases_risk, neutral
importance is a float between 0.0 and 1.0"""


def _llm_explain(symptoms: str, diagnosis_summary: str) -> list:
    """LLM-based fallback explanation."""
    try:
        result = _llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Patient symptoms: {symptoms}\nDiagnosis summary: {diagnosis_summary}"),
        ])
        content = result.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
        raw = json.loads(content.strip())
        return raw.get("top_factors", [])
    except Exception as e:
        logger.warning("[SHAP] LLM fallback error: %s", e)
        return []


# ---------------------------------------------------------------------------
# Model loader (singleton, thread-safe)
# ---------------------------------------------------------------------------

_models: dict[str, Any] | None = None
_models_lock = threading.Lock()
_models_loaded = False


def preload_models() -> None:
    """Eagerly load SHAP models at startup. Safe to call multiple times."""
    _load_models()


def _load_models() -> dict[str, Any] | None:
    global _models, _models_loaded
    if _models_loaded:
        return _models
    with _models_lock:
        if _models_loaded:
            return _models
        if not _MODEL_FILE.exists():
            logger.warning(
                "[SHAP] Model file not found at %s — falling back to LLM explainer. "
                "Run services/cancer-agent/scripts/export_models.py to generate it.",
                _MODEL_FILE,
            )
            _models_loaded = True
            _models = None
            return None
        try:
            with open(_MODEL_FILE, "rb") as f:
                _models = pickle.load(f)
            logger.info(
                "[SHAP] Loaded models from %s | tasks: %s",
                _MODEL_FILE, list(_models.keys()),
            )
        except Exception as exc:
            logger.warning("[SHAP] Failed to load model file: %s — using LLM fallback.", exc)
            _models = None
        _models_loaded = True
        return _models


# ---------------------------------------------------------------------------
# SHAP attribution helpers
# ---------------------------------------------------------------------------

def _top_features(
    shap_vals: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict]:
    """Return top_n features sorted by abs(shap_value), non-zero only."""
    pairs = [
        (name, float(val))
        for name, val in zip(feature_names, shap_vals)
        if abs(val) > 1e-6 and name.strip()
    ]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return [
        {
            "factor": name,
            "importance": round(min(abs(val), 1.0), 4),
            "direction": "increases_risk" if val > 0 else "decreases_risk",
        }
        for name, val in pairs[:top_n]
    ]


def _shap_severity(models: dict, symptoms: str) -> list[dict]:
    """Linear attribution for LogisticRegression severity model."""
    m = models.get("severity")
    if not m:
        return []
    tfidf_doc = m["tfidf_doc"]
    tfidf_icd = m["tfidf_icd"]
    clf = m["clf"]

    X_doc = tfidf_doc.transform([symptoms])
    X_icd = tfidf_icd.transform([""])           # ICD not available at XAI time
    X_icu = csr_matrix(np.array([0.0]).reshape(1, 1))
    X = hstack([X_doc, X_icd, X_icu])

    pred_class = clf.predict(X)[0]
    class_idx  = list(clf.classes_).index(pred_class)

    # Linear SHAP: shap_i = x_i * w_i  (equivalent to SHAP for linear models)
    coef = clf.coef_[class_idx]
    shap_vals = np.asarray(X.toarray()[0]) * coef

    feature_names = (
        list(tfidf_doc.get_feature_names_out()) +
        list(tfidf_icd.get_feature_names_out()) +
        ["has_icu_stay"]
    )
    factors = _top_features(shap_vals, feature_names)
    for f in factors:
        f["factor"] = f"[severity:{pred_class}] {f['factor']}"
    return factors


def _shap_emergency(models: dict, symptoms: str) -> list[dict]:
    """
    TreeSHAP for HistGBM emergency model.
    SHAP values are in SVD component space; projected back through SVD.components_
    to recover approximate word-level importances.
    """
    import shap as _shap

    m = models.get("emergency")
    if not m:
        return []

    tfidf_doc = m["tfidf_doc"]
    tfidf_icd = m["tfidf_icd"]
    tfidf_cc  = m["tfidf_cc"]
    svd       = m["svd"]
    clf       = m["clf"]

    X_doc = tfidf_doc.transform([symptoms])
    X_icd = tfidf_icd.transform([""])
    X_cc  = tfidf_cc.transform([symptoms])     # use symptoms as chief-complaint proxy
    X_icu = csr_matrix(np.array([0.0]).reshape(1, 1))
    X_sp  = hstack([X_doc, X_icd, X_cc, X_icu])
    X_dense = svd.transform(X_sp)              # shape: (1, n_components)

    explainer  = shap.TreeExplainer(clf)
    shap_svd   = explainer.shap_values(X_dense)[0]  # shape: (n_components,) for positive class

    # Project SVD SHAP values back to original TF-IDF feature space
    # svd.components_ shape: (n_components, n_original_features)
    word_shap = svd.components_.T @ shap_svd   # shape: (n_original_features,)

    # Build feature names (doc + icd + cc + icu_flag)
    feature_names = (
        list(tfidf_doc.get_feature_names_out()) +
        list(tfidf_icd.get_feature_names_out()) +
        list(tfidf_cc.get_feature_names_out()) +
        ["has_icu_stay"]
    )
    # Only keep doc and cc features (ICD not present at inference)
    doc_end = len(tfidf_doc.get_feature_names_out())
    icd_end = doc_end + len(tfidf_icd.get_feature_names_out())
    cc_end  = icd_end + len(tfidf_cc.get_feature_names_out())

    mask = np.zeros(len(feature_names), dtype=bool)
    mask[:doc_end] = True          # doc features
    mask[icd_end:cc_end] = True    # cc features

    filtered_vals  = word_shap * mask
    factors = _top_features(filtered_vals, feature_names)
    for f in factors:
        f["factor"] = f"[emergency] {f['factor']}"
    return factors


def _shap_cancer_type(models: dict, symptoms: str) -> list[dict]:
    """
    Linear attribution for CalibratedClassifierCV(LinearSVC) cancer-type model.
    Uses the first calibrated fold's underlying LinearSVC coefs.
    """
    m = models.get("cancer_type")
    if not m:
        return []

    tfidf_doc = m["tfidf_doc"]
    tfidf_icd = m["tfidf_icd"]
    clf       = m["clf"]       # CalibratedClassifierCV

    X_doc = tfidf_doc.transform([symptoms])
    X_icd = tfidf_icd.transform([""])
    X = hstack([X_doc, X_icd])

    pred_class = clf.predict(X)[0]

    # Extract coefs from the first fold's LinearSVC
    try:
        first_clf  = clf.calibrated_classifiers_[0].estimator
        class_idx  = list(first_clf.classes_).index(pred_class)
        coef       = first_clf.coef_[class_idx]
    except (AttributeError, ValueError, IndexError):
        return []

    shap_vals = np.asarray(X.toarray()[0]) * coef
    feature_names = (
        list(tfidf_doc.get_feature_names_out()) +
        list(tfidf_icd.get_feature_names_out())
    )
    factors = _top_features(shap_vals, feature_names)
    for f in factors:
        f["factor"] = f"[{pred_class}] {f['factor']}"
    return factors


# ---------------------------------------------------------------------------
# Public DiagnosisExplainer
# ---------------------------------------------------------------------------

class DiagnosisExplainer:
    """
    Provides explainability for AI diagnosis decisions.

    Primary path (when cancer_agent_models.pkl is present):
      - Severity: linear SHAP on LogisticRegression
      - Emergency: TreeSHAP on HistGBM, projected back to word space
      - Cancer type: linear SHAP on CalibratedClassifierCV(LinearSVC)

    Fallback (pkl not found or any SHAP error):
      - LLM-based explanation (original implementation)
    """

    def __init__(self):
        self.last_method: str = ""

    def explain_diagnosis(self, symptoms: str, diagnosis_summary: str) -> list:
        """Return top contributing clinical factors for the diagnosis."""
        models = _load_models()

        if models is not None:
            try:
                result = self._explain_with_shap(models, symptoms)
                self.last_method = "SHAP"
                return result
            except Exception as exc:
                logger.warning("[SHAP] SHAP computation failed: %s — using LLM fallback.", exc)

        self.last_method = "LLM_FALLBACK"
        return _llm_explain(symptoms, diagnosis_summary)

    def _explain_with_shap(self, models: dict, symptoms: str) -> list:
        """Compute SHAP explanations across all available models and return top 3."""
        all_factors: list[dict] = []

        # Severity (LR) — always fast
        try:
            all_factors.extend(_shap_severity(models, symptoms))
        except Exception as exc:
            logger.debug("[SHAP] Severity SHAP error: %s", exc)

        # Emergency (HistGBM + SVD)
        try:
            all_factors.extend(_shap_emergency(models, symptoms))
        except Exception as exc:
            logger.debug("[SHAP] Emergency SHAP error: %s", exc)

        # Cancer type (LinearSVC)
        try:
            all_factors.extend(_shap_cancer_type(models, symptoms))
        except Exception as exc:
            logger.debug("[SHAP] Cancer type SHAP error: %s", exc)

        if not all_factors:
            raise RuntimeError("All SHAP computations returned empty results.")

        # Deduplicate by factor name (keep highest importance), sort, return top 3
        seen: dict[str, dict] = {}
        for f in all_factors:
            key = f["factor"]
            if key not in seen or f["importance"] > seen[key]["importance"]:
                seen[key] = f

        ranked = sorted(seen.values(), key=lambda x: x["importance"], reverse=True)

        logger.info("[SHAP] %d factor(s) computed | top: %s (%.4f)",
                    len(ranked), ranked[0]["factor"] if ranked else "—",
                    ranked[0]["importance"] if ranked else 0.0)
        return ranked[:3]
