"""
TF-IDF Predictor — Cancer Agent Production Classifier
======================================================
Trains TF-IDF + classifier models on MIMIC-IV evaluation cases at startup
and exposes prediction for structured diagnosis fields, bypassing the LLM
for those fields to reduce cost and improve consistency.

Fields predicted (not requested from LLM):
  severity              LOW / HIGH / CRITICAL     (LR)
  severityConfidence    0-100                     (LR predict_proba)
  emergencyCareNeeded   YES / NO                  (HistGBM + SVD + Youden)
  emergencyCareConfidence 0-100                   (HistGBM predict_proba)
  hospitalizationNeeded YES / NO                  (HistGBM + SVD + Youden)
  suspectedCancerType   e.g. "Lung Cancer"        (LinearSVC)

At inference only patient symptom text is available (no ICD codes,
no has_icu_stay). ICD features default to empty string and ICU flag to 0.
"""

import threading

import numpy as np
from pymongo import MongoClient
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.svm import LinearSVC

from core.config import MONGO_DB, MONGO_EVAL_COLLECTION, MONGO_URI
from log.logger import logger


# ---------------------------------------------------------------------------
# Label helpers (inline — avoids cross-service import)
# ---------------------------------------------------------------------------

_EMERGENCY_ADMISSION_TYPES = frozenset({"EMERGENCY", "DIRECT EMER.", "URGENT"})
_NON_EMERGENCY_ADMISSION_TYPES = frozenset({
    "ELECTIVE", "AMBULATORY OBSERVATION", "DIRECT OBSERVATION",
    "EU OBSERVATION", "OBSERVATION ADMIT", "SURGICAL SAME DAY ADMISSION",
})

_CANCER_CATEGORY_MAP: dict[str, str] = {
    "lung": "Lung Cancer",          "bronch": "Lung Cancer",
    "breast": "Breast Cancer",
    "colon": "Colorectal Cancer",   "rectal": "Colorectal Cancer",
    "colorectal": "Colorectal Cancer",
    "prostate": "Prostate Cancer",
    "bladder": "Bladder Cancer",
    "kidney": "Kidney Cancer",      "renal cell": "Kidney Cancer",
    "leuk": "Leukaemia",
    "lymphoma": "Lymphoma",
    "melanoma": "Melanoma",
    "pancrea": "Pancreatic Cancer",
    "liver": "Liver Cancer",        "hepat": "Liver Cancer",
    "ovari": "Ovarian Cancer",
    "cervix": "Cervical Cancer",    "cervi": "Cervical Cancer",
    "uteri": "Uterine Cancer",
    "thyroid": "Thyroid Cancer",
    "brain": "Brain Cancer",        "glioma": "Brain Cancer",
    "stomach": "Gastric Cancer",    "gastric": "Gastric Cancer",
    "oesophag": "Oesophageal Cancer", "esophag": "Oesophageal Cancer",
    "myeloma": "Multiple Myeloma",
    "testicular": "Testicular Cancer",
    "head and neck": "Head and Neck Cancer",
    "oral": "Head and Neck Cancer", "pharynx": "Head and Neck Cancer",
    "larynx": "Head and Neck Cancer",
    "skin": "Skin Cancer",
}


def _label_emergency(admission_type: str, severity: str) -> int | None:
    a = (admission_type or "").upper().strip()
    if a in _EMERGENCY_ADMISSION_TYPES:
        return 1
    if a in _NON_EMERGENCY_ADMISSION_TYPES:
        return 0
    s = (severity or "").upper().strip()
    if s == "CRITICAL":
        return 1
    if s == "LOW":
        return 0
    return None


def _label_hospitalization(severity: str) -> int | None:
    s = (severity or "").upper().strip()
    if s in ("HIGH", "CRITICAL"):
        return 1
    if s == "LOW":
        return 0
    return None


def _normalise_cancer_type(raw: str) -> str | None:
    lower = (raw or "").lower()
    for kw, cat in _CANCER_CATEGORY_MAP.items():
        if kw in lower:
            return cat
    return None  # "Other Cancer" excluded from training


# ---------------------------------------------------------------------------
# TF-IDF factory helpers (mirrors evaluation-service implementation)
# ---------------------------------------------------------------------------

def _make_tfidf(n: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, 2),
        min_df=1 if n < 200 else 2, max_df=0.95, max_features=30_000,
    )


def _make_icd_tfidf(n: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, 1),
        min_df=1 if n < 200 else 2, max_features=5_000,
        token_pattern=r"[A-Za-z][0-9]+(?:\.[0-9]+)?",
    )


def _make_complaint_tfidf(n: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, 2),
        min_df=1 if n < 200 else 2, max_df=0.95, max_features=5_000,
    )


def _make_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, class_weight="balanced", random_state=42,
    )


def _youden_threshold(clf, X_train, y_train) -> float:
    scores = clf.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(list(y_train), scores)
    return float(thresholds[np.argmax(tpr - fpr)])


# ---------------------------------------------------------------------------
# Predictor — singleton, lazy-trained
# ---------------------------------------------------------------------------

class TfidfPredictor:
    """
    Singleton predictor trained on MIMIC evaluation data.
    Call ensure_trained() before predict(); training is thread-safe and
    happens at most once per process lifetime.
    """

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._trained = False

        # severity
        self._tfidf_doc_sev: TfidfVectorizer | None = None
        self._tfidf_icd_sev: TfidfVectorizer | None = None
        self._clf_sev: LogisticRegression | None     = None

        # emergency
        self._tfidf_doc_emg: TfidfVectorizer | None           = None
        self._tfidf_icd_emg: TfidfVectorizer | None           = None
        self._tfidf_cc_emg:  TfidfVectorizer | None           = None
        self._svd_emg:  TruncatedSVD | None                   = None
        self._clf_emg:  HistGradientBoostingClassifier | None  = None
        self._thresh_emg: float                               = 0.5

        # hospitalization
        self._tfidf_doc_hosp: TfidfVectorizer | None          = None
        self._tfidf_icd_hosp: TfidfVectorizer | None          = None
        self._tfidf_cc_hosp:  TfidfVectorizer | None          = None
        self._svd_hosp:  TruncatedSVD | None                  = None
        self._clf_hosp:  HistGradientBoostingClassifier | None = None
        self._thresh_hosp: float                              = 0.5

        # cancer type
        self._tfidf_doc_can: TfidfVectorizer | None           = None
        self._tfidf_icd_can: TfidfVectorizer | None           = None
        self._clf_can: CalibratedClassifierCV | None          = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def ensure_trained(self) -> None:
        if self._trained:
            return
        with self._lock:
            if self._trained:
                return
            self._train()

    def _train(self) -> None:
        logger.info("[TFIDF_PRED] Loading MIMIC evaluation records from MongoDB...")
        client  = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
        records = list(client[MONGO_DB][MONGO_EVAL_COLLECTION].find({}, {"_id": 0}))
        client.close()

        if not records:
            logger.warning("[TFIDF_PRED] No MIMIC records found — predictor will use defaults.")
            self._trained = True
            return

        logger.info("[TFIDF_PRED] Training on %d MIMIC records...", len(records))

        texts      = [r.get("document", "")                    for r in records]
        icd_texts  = [r.get("icd_codes", "").replace(",", " ") for r in records]
        complaints = [r.get("chief_complaint", "") or ""       for r in records]
        icu_flags  = [int(bool(r.get("has_icu_stay", 0)))       for r in records]

        self._train_severity(texts, icd_texts, icu_flags, records)
        self._train_emergency(texts, icd_texts, complaints, icu_flags, records)
        self._train_hospitalization(texts, icd_texts, complaints, icu_flags, records)
        self._train_cancer_type(texts, icd_texts, records)

        self._trained = True
        logger.info("[TFIDF_PRED] All models trained and ready.")

    def _train_severity(self, texts, icd_texts, icu_flags, records) -> None:
        data = [
            (texts[i], icd_texts[i], icu_flags[i], r["severity"].upper())
            for i, r in enumerate(records)
            if r.get("severity", "").upper() in ("LOW", "HIGH", "CRITICAL")
        ]
        if not data:
            return
        tx, ic, icu, y = zip(*data)
        self._tfidf_doc_sev = _make_tfidf(len(tx))
        self._tfidf_icd_sev = _make_icd_tfidf(len(ic))
        X = hstack([
            self._tfidf_doc_sev.fit_transform(tx),
            self._tfidf_icd_sev.fit_transform(ic),
            csr_matrix(np.array(icu, dtype=float).reshape(-1, 1)),
        ])
        self._clf_sev = LogisticRegression(C=1.0, max_iter=1_000, class_weight="balanced")
        self._clf_sev.fit(X, y)
        logger.info("[TFIDF_PRED] Severity model ready (%d samples)", len(y))

    def _train_emergency(self, texts, icd_texts, complaints, icu_flags, records) -> None:
        data = []
        for i, r in enumerate(records):
            label = _label_emergency(r.get("admission_type", ""), r.get("severity", ""))
            if label is not None:
                data.append((texts[i], icd_texts[i], complaints[i], icu_flags[i], label))
        if not data:
            return
        tx, ic, cc, icu, y = zip(*data)
        self._tfidf_doc_emg = _make_tfidf(len(tx))
        self._tfidf_icd_emg = _make_icd_tfidf(len(ic))
        self._tfidf_cc_emg  = _make_complaint_tfidf(len(cc))
        X_sp = hstack([
            self._tfidf_doc_emg.fit_transform(tx),
            self._tfidf_icd_emg.fit_transform(ic),
            self._tfidf_cc_emg.fit_transform(cc),
            csr_matrix(np.array(icu, dtype=float).reshape(-1, 1)),
        ])
        n_comp = min(300, X_sp.shape[1] - 1)
        self._svd_emg = TruncatedSVD(n_components=n_comp, random_state=42)
        X_dense = self._svd_emg.fit_transform(X_sp)
        self._clf_emg = _make_gbm()
        self._clf_emg.fit(X_dense, y)
        self._thresh_emg = _youden_threshold(self._clf_emg, X_dense, y)
        logger.info("[TFIDF_PRED] Emergency model ready (%d samples) | threshold=%.4f",
                    len(y), self._thresh_emg)

    def _train_hospitalization(self, texts, icd_texts, complaints, icu_flags, records) -> None:
        data = []
        for i, r in enumerate(records):
            label = _label_hospitalization(r.get("severity", ""))
            if label is not None:
                data.append((texts[i], icd_texts[i], complaints[i], icu_flags[i], label))
        if not data:
            return
        tx, ic, cc, icu, y = zip(*data)
        self._tfidf_doc_hosp = _make_tfidf(len(tx))
        self._tfidf_icd_hosp = _make_icd_tfidf(len(ic))
        self._tfidf_cc_hosp  = _make_complaint_tfidf(len(cc))
        X_sp = hstack([
            self._tfidf_doc_hosp.fit_transform(tx),
            self._tfidf_icd_hosp.fit_transform(ic),
            self._tfidf_cc_hosp.fit_transform(cc),
            csr_matrix(np.array(icu, dtype=float).reshape(-1, 1)),
        ])
        n_comp = min(300, X_sp.shape[1] - 1)
        self._svd_hosp = TruncatedSVD(n_components=n_comp, random_state=42)
        X_dense = self._svd_hosp.fit_transform(X_sp)
        self._clf_hosp = _make_gbm()
        self._clf_hosp.fit(X_dense, y)
        self._thresh_hosp = _youden_threshold(self._clf_hosp, X_dense, y)
        logger.info("[TFIDF_PRED] Hospitalization model ready (%d samples) | threshold=%.4f",
                    len(y), self._thresh_hosp)

    def _train_cancer_type(self, texts, icd_texts, records) -> None:
        data = []
        for i, r in enumerate(records):
            cat = _normalise_cancer_type(r.get("cancer_type", ""))
            if cat:
                data.append((texts[i], icd_texts[i], cat))
        if not data:
            return
        tx, ic, y = zip(*data)
        self._tfidf_doc_can = _make_tfidf(len(tx))
        self._tfidf_icd_can = _make_icd_tfidf(len(ic))
        X = hstack([
            self._tfidf_doc_can.fit_transform(tx),
            self._tfidf_icd_can.fit_transform(ic),
        ])
        self._clf_can = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2_000, class_weight="balanced"), cv=3,
        )
        self._clf_can.fit(X, y)
        logger.info("[TFIDF_PRED] Cancer type model ready (%d samples)", len(y))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, symptoms: str) -> dict:
        """
        Predict structured fields from patient symptom text.
        ICD codes and has_icu_stay are not available at inference time
        and default to empty string / 0.
        """
        self.ensure_trained()

        # Safe defaults (conservative: assume hospitalisation needed, uncertain emergency)
        result: dict = {
            "severity":              "HIGH",
            "severityConfidence":    50,
            "emergencyCareNeeded":   "NO",
            "emergencyCareConfidence": 50,
            "hospitalizationNeeded": "YES",
            "suspectedCancerType":   "Unknown",
        }

        text = symptoms or ""
        icd  = ""    # not available at inference time
        cc   = text  # use symptoms text as chief-complaint proxy

        # --- Severity ---
        if self._clf_sev is not None:
            try:
                X = hstack([
                    self._tfidf_doc_sev.transform([text]),
                    self._tfidf_icd_sev.transform([icd]),
                    csr_matrix(np.array([0], dtype=float).reshape(1, 1)),
                ])
                pred  = self._clf_sev.predict(X)[0]
                conf  = int(round(max(self._clf_sev.predict_proba(X)[0]) * 100))
                result["severity"]          = str(pred)
                result["severityConfidence"] = conf
            except Exception as exc:
                logger.warning("[TFIDF_PRED] Severity predict error: %s", exc)

        # --- Emergency ---
        if self._clf_emg is not None:
            try:
                X_sp = hstack([
                    self._tfidf_doc_emg.transform([text]),
                    self._tfidf_icd_emg.transform([icd]),
                    self._tfidf_cc_emg.transform([cc]),
                    csr_matrix(np.array([0], dtype=float).reshape(1, 1)),
                ])
                score = float(self._clf_emg.predict_proba(self._svd_emg.transform(X_sp))[0, 1])
                result["emergencyCareNeeded"]     = "YES" if score >= self._thresh_emg else "NO"
                result["emergencyCareConfidence"] = int(round(score * 100))
            except Exception as exc:
                logger.warning("[TFIDF_PRED] Emergency predict error: %s", exc)

        # --- Hospitalization ---
        if self._clf_hosp is not None:
            try:
                X_sp = hstack([
                    self._tfidf_doc_hosp.transform([text]),
                    self._tfidf_icd_hosp.transform([icd]),
                    self._tfidf_cc_hosp.transform([cc]),
                    csr_matrix(np.array([0], dtype=float).reshape(1, 1)),
                ])
                score = float(self._clf_hosp.predict_proba(self._svd_hosp.transform(X_sp))[0, 1])
                result["hospitalizationNeeded"] = "YES" if score >= self._thresh_hosp else "NO"
            except Exception as exc:
                logger.warning("[TFIDF_PRED] Hospitalization predict error: %s", exc)

        # --- Cancer type ---
        if self._clf_can is not None:
            try:
                X = hstack([
                    self._tfidf_doc_can.transform([text]),
                    self._tfidf_icd_can.transform([icd]),
                ])
                result["suspectedCancerType"] = str(self._clf_can.predict(X)[0])
            except Exception as exc:
                logger.warning("[TFIDF_PRED] Cancer type predict error: %s", exc)

        logger.debug(
            "[TFIDF_PRED] Predicted | severity=%s(%d) | emergency=%s(%d) | hosp=%s | cancer=%s",
            result["severity"], result["severityConfidence"],
            result["emergencyCareNeeded"], result["emergencyCareConfidence"],
            result["hospitalizationNeeded"], result["suspectedCancerType"],
        )
        return result


# ---------------------------------------------------------------------------
# Module-level singleton + public API
# ---------------------------------------------------------------------------

_predictor = TfidfPredictor()


def predict_diagnosis_fields(symptoms: str) -> dict:
    """Predict structured diagnosis fields from patient symptom text."""
    return _predictor.predict(symptoms)


def warm_up() -> None:
    """Pre-train models at service startup (call from main.py)."""
    _predictor.ensure_trained()
