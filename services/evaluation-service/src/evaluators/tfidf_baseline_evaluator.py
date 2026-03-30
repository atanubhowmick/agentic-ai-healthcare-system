"""
TF-IDF Baseline Evaluator — Cancer Agent Standalone Comparison
===============================================================
Trains TF-IDF + classifier models on MIMIC-IV evaluation cases and
computes the same metrics as CancerAgentEvaluator, providing a classical
ML baseline for direct comparison.

Research framing
----------------
"Does clinical reasoning from an LLM outperform vocabulary-based
classification trained on the same dataset?"

Evaluation design
-----------------
- Loads all available MIMIC-IV evaluation cases from MongoDB.
- Stratified 80/20 train/test split (per task, stratified on task label).
- Fits one combined-feature classifier per task on the train split.
- Reports metrics on the held-out test split using the same metric
  functions as CancerAgentEvaluator so results are directly comparable.

Features used
-------------
- TF-IDF on document text  (triage complaint + chief complaint + HPI)
- TF-IDF on icd_codes      (structured ICD-10 code tokens)
- has_icu_stay binary flag  (severity task only — direct CRITICAL signal)

Classifiers
-----------
- Emergency (binary):     HistGradientBoosting (non-linear interactions, sparse-native)
- Severity (3-class):     Logistic Regression  (handles class imbalance)
- Cancer type (multi):    LinearSVC            (outperforms LR on balanced text)

NOTE: The TF-IDF classifier is trained on MIMIC data (supervised),
      whereas the LLM agent is zero-shot. This gives TF-IDF a training
      advantage — that asymmetry is intentional and should be reported.

Tasks
-----
emergency_care_needed  binary   admission_type → YES / NO
severity               3-class  LOW / HIGH / CRITICAL
cancer_type            match    normalised ICD category accuracy
"""

from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from core.mongo_client import load_evaluation_cases, save_tfidf_report
from evaluators.label_mapper import MimicLabelMapper
from evaluators.metrics_calculator import AgentEvaluator
from log.logger import logger


# ---------------------------------------------------------------------------
# Feature builder helpers
# ---------------------------------------------------------------------------

def _make_tfidf(train_size: int) -> TfidfVectorizer:
    """TF-IDF for clinical document text (unigrams + bigrams)."""
    min_df = 1 if train_size < 200 else 2
    return TfidfVectorizer(
        sublinear_tf = True,
        ngram_range  = (1, 2),
        min_df       = min_df,
        max_df       = 0.95,
        max_features = 30_000,
        analyzer     = "word",
    )


def _make_icd_tfidf(train_size: int) -> TfidfVectorizer:
    """
    TF-IDF for ICD-10 code strings.
    Uses a custom token pattern to capture full codes including the decimal
    (e.g. 'C34.1', 'C50', 'D37.0').  Unigrams only — codes are atomic tokens.
    """
    min_df = 1 if train_size < 200 else 2
    return TfidfVectorizer(
        sublinear_tf  = True,
        ngram_range   = (1, 1),
        min_df        = min_df,
        max_features  = 5_000,
        analyzer      = "word",
        token_pattern = r"[A-Za-z][0-9]+(?:\.[0-9]+)?",  # C34.1, C50, D37.0
    )


def _make_complaint_tfidf(train_size: int) -> TfidfVectorizer:
    """
    TF-IDF for chief/triage complaint text.
    Kept separate from the full document so the emergency-presenting
    vocabulary gets its own dedicated feature space (not diluted by HPI /
    discharge summary text).  Unigrams + bigrams, small vocabulary.
    """
    min_df = 1 if train_size < 200 else 2
    return TfidfVectorizer(
        sublinear_tf = True,
        ngram_range  = (1, 2),
        min_df       = min_df,
        max_df       = 0.95,
        max_features = 5_000,
        analyzer     = "word",
    )


def _make_lr() -> LogisticRegression:
    """Logistic Regression — severity and imbalanced multi-class tasks."""
    return LogisticRegression(C=1.0, max_iter=1_000, class_weight="balanced")


def _make_gbm() -> HistGradientBoostingClassifier:
    """
    HistGradientBoostingClassifier — emergency binary task.
    Captures non-linear interactions between TF-IDF text, ICD codes,
    chief complaint, and has_icu_stay that LR cannot model.
    Supports scipy sparse input natively (sklearn >= 1.1).
    """
    return HistGradientBoostingClassifier(
        max_iter        = 300,
        learning_rate   = 0.05,
        max_depth       = 6,
        min_samples_leaf= 20,
        class_weight    = "balanced",
        random_state    = 42,
    )


def _make_svm() -> CalibratedClassifierCV:
    """
    LinearSVC wrapped for predict_proba support.
    Used for cancer type — balanced multi-class, text-heavy task.
    """
    return CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2_000, class_weight="balanced"),
        cv=3,
    )


def _build_features(
    X_text_tr: list, X_icd_tr: list,
    X_text_te: list, X_icd_te: list,
    icu_tr: list = None, icu_te: list = None,
    X_complaint_tr: list = None, X_complaint_te: list = None,
):
    """
    Build combined sparse feature matrices for train and test sets.

    Combines:
      - TF-IDF(document text)
      - TF-IDF(ICD codes)
      - TF-IDF(chief/triage complaint)  (only when X_complaint_tr/te provided)
      - has_icu_stay binary column       (only when icu_tr / icu_te are provided)

    Returns:
        (X_tr_combined, X_te_combined)  — scipy sparse matrices
    """
    tfidf_text = _make_tfidf(len(X_text_tr))
    tfidf_icd  = _make_icd_tfidf(len(X_icd_tr))

    X_tr_text = tfidf_text.fit_transform(X_text_tr)
    X_te_text = tfidf_text.transform(X_text_te)

    X_tr_icd  = tfidf_icd.fit_transform(X_icd_tr)
    X_te_icd  = tfidf_icd.transform(X_icd_te)

    parts_tr = [X_tr_text, X_tr_icd]
    parts_te = [X_te_text, X_te_icd]

    if X_complaint_tr is not None and X_complaint_te is not None:
        tfidf_complaint = _make_complaint_tfidf(len(X_complaint_tr))
        parts_tr.append(tfidf_complaint.fit_transform(X_complaint_tr))
        parts_te.append(tfidf_complaint.transform(X_complaint_te))

    if icu_tr is not None and icu_te is not None:
        parts_tr.append(csr_matrix(np.array(icu_tr, dtype=float).reshape(-1, 1)))
        parts_te.append(csr_matrix(np.array(icu_te, dtype=float).reshape(-1, 1)))

    return hstack(parts_tr), hstack(parts_te)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class TfidfBaselineEvaluator:
    """
    TF-IDF + structured feature baseline evaluator for Cancer Agent tasks.
    """

    def __init__(self, test_size: float = 0.20, random_state: int = 42):
        self._test_size    = test_size
        self._random_state = random_state
        self._mapper       = MimicLabelMapper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_evaluation(self, max_cases: int = 0) -> dict:
        """
        Full TF-IDF baseline evaluation.

        Args:
            max_cases: Cap at N records (0 = all available in MongoDB).

        Returns:
            Evaluation report dict (also persisted to MongoDB).
        """
        records = load_evaluation_cases(max_cases=max_cases)
        if not records:
            logger.warning("[TFIDF] No evaluation cases in MongoDB — aborting.")
            return {"error": "no_evaluation_cases_in_mongodb"}

        logger.info(
            "[TFIDF] Loaded %d records | test_size=%.0f%% | random_state=%d",
            len(records), self._test_size * 100, self._random_state,
        )

        # ---- Extract feature columns ------------------------------------
        texts      = [r.get("document", "")                          for r in records]
        icd_texts  = [r.get("icd_codes", "").replace(",", " ")       for r in records]
        icu_flags  = [int(bool(r.get("has_icu_stay", 0)))             for r in records]
        complaints = [r.get("chief_complaint", "") or ""             for r in records]


        # ---- Derive labels for each task --------------------------------
        emerg_idx, emerg_y   = [], []
        hosp_idx,  hosp_y    = [], []
        sev_idx,   sev_y     = [], []
        cancer_idx, cancer_y = [], []

        for i, r in enumerate(records):
            raw_sev    = r.get("severity", "")
            adm_type   = r.get("admission_type", "")
            chief_cc   = r.get("chief_complaint", "")
            cancer_raw = r.get("cancer_type", "")

            e = self._mapper.map_emergency(adm_type, raw_sev, chief_cc)
            h = self._mapper.map_hospitalization(raw_sev)
            s = self._mapper.map_severity(raw_sev)
            c = self._mapper.normalise_cancer_type(cancer_raw) if cancer_raw else None

            if e is not None:
                emerg_idx.append(i)
                emerg_y.append(self._mapper.encode_binary(e))
            if h is not None:
                hosp_idx.append(i)
                hosp_y.append(self._mapper.encode_binary(h))
            if s is not None:
                sev_idx.append(i)
                sev_y.append(s)
            if c and c != "Other Cancer":
                cancer_idx.append(i)
                cancer_y.append(c)

        logger.info(
            "[TFIDF] Label counts | emergency: %d | hospitalization: %d | severity: %d | cancer_type: %d",
            len(emerg_idx), len(hosp_idx), len(sev_idx), len(cancer_idx),
        )

        metrics: dict = {}

        # ---- Task 1: Emergency — text + ICD + complaint + ICU -----------
        metrics.update(self._binary_task(
            X_text      = [texts[i]      for i in emerg_idx],
            X_icd       = [icd_texts[i]  for i in emerg_idx],
            X_complaint = [complaints[i] for i in emerg_idx],
            X_icu       = [icu_flags[i]  for i in emerg_idx],
            y           = emerg_y,
            label       = "emergency_care_needed",
        ))

        # ---- Task 2: Hospitalization (binary) — text + ICD + ICU --------
        metrics.update(self._binary_task(
            X_text      = [texts[i]      for i in hosp_idx],
            X_icd       = [icd_texts[i]  for i in hosp_idx],
            X_complaint = [complaints[i] for i in hosp_idx],
            X_icu       = [icu_flags[i]  for i in hosp_idx],
            y           = hosp_y,
            label       = "hospitalization_needed",
        ))

        # ---- Task 3: Severity (3-class) — text + ICD + has_icu_stay -----
        metrics.update(self._multiclass_task(
            X_text = [texts[i]     for i in sev_idx],
            X_icd  = [icd_texts[i] for i in sev_idx],
            X_icu  = [icu_flags[i] for i in sev_idx],
            y      = sev_y,
            labels = ["LOW", "HIGH", "CRITICAL"],
            task_name = "severity",
        ))

        # ---- Task 3: Cancer type — text + ICD ---------------------------
        metrics["cancer_type"] = self._cancer_type_task(
            X_text = [texts[i]     for i in cancer_idx],
            X_icd  = [icd_texts[i] for i in cancer_idx],
            y      = cancer_y,
        )

        report = {
            "approach": "tfidf_baseline",
            "summary": {
                "total_records":           len(records),
                "test_fraction":           self._test_size,
                "random_state":            self._random_state,
                "emergency_labeled":       len(emerg_idx),
                "hospitalization_labeled": len(hosp_idx),
                "severity_labeled":        len(sev_idx),
                "cancer_type_labeled":     len(cancer_idx),
            },
            "metrics": metrics,
            "note": (
                "Emergency/Hospitalization features: TF-IDF(document) + TF-IDF(ICD) + TF-IDF(chief_complaint) "
                "+ has_icu_stay → TruncatedSVD(300) → HistGBM + Youden threshold. "
                "Severity features: same text + ICD + has_icu_stay → LR. "
                "Cancer type: text + ICD → LinearSVC. "
                "TF-IDF trained on train split (supervised); LLM agent is zero-shot."
            ),
        }

        save_tfidf_report(report)
        logger.info("[TFIDF] Report saved to MongoDB.")
        return report

    # ------------------------------------------------------------------
    # Task runners
    # ------------------------------------------------------------------

    def _binary_task(
        self,
        X_text: list, X_icd: list, y: list, label: str,
        X_complaint: list = None,
        X_icu: list = None,
    ) -> dict:
        """
        Binary classification: TF-IDF(text) + TF-IDF(ICD)
                              + TF-IDF(chief complaint)  [optional]
                              + has_icu_stay binary      [optional]
                              → SVD → HistGBM with Youden-optimal threshold.
        """
        if len(X_text) < 20 or len(set(y)) < 2:
            return {label: {"error": "insufficient_data", "n": len(X_text)}}

        idx = list(range(len(X_text)))
        try:
            idx_tr, idx_te, y_tr, y_te = train_test_split(
                idx, y,
                test_size    = self._test_size,
                random_state = self._random_state,
                stratify     = y,
            )
        except ValueError as exc:
            return {label: {"error": f"split_failed: {exc}", "n": len(X_text)}}

        X_tr_feat, X_te_feat = _build_features(
            [X_text[i] for i in idx_tr], [X_icd[i] for i in idx_tr],
            [X_text[i] for i in idx_te], [X_icd[i] for i in idx_te],
            icu_tr         = [X_icu[i]       for i in idx_tr] if X_icu       else None,
            icu_te         = [X_icu[i]       for i in idx_te] if X_icu       else None,
            X_complaint_tr = [X_complaint[i] for i in idx_tr] if X_complaint else None,
            X_complaint_te = [X_complaint[i] for i in idx_te] if X_complaint else None,
        )

        # -- TruncatedSVD: reduce sparse TF-IDF → dense for GBM ----------
        # HistGradientBoostingClassifier requires dense input. Full .toarray()
        # on 22k×33k would use ~6 GB; SVD reduces to 300 latent components
        # (~54 MB) while retaining the most discriminative variance.
        n_components = min(300, X_tr_feat.shape[1] - 1)
        logger.info(
            "[GBM] Applying TruncatedSVD | sparse_features: %d → dense_components: %d",
            X_tr_feat.shape[1], n_components,
        )
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_tr_dense = svd.fit_transform(X_tr_feat)
        X_te_dense = svd.transform(X_te_feat)

        clf = _make_gbm()
        logger.info(
            "[GBM] Training started | task: %s | train_samples: %d | components: %d | max_iter: %d",
            label, X_tr_dense.shape[0], X_tr_dense.shape[1], clf.max_iter,
        )
        clf.fit(X_tr_dense, y_tr)
        logger.info(
            "[GBM] Training complete | task: %s | iterations_run: %d",
            label, clf.n_iter_,
        )

        # -- Youden-optimal threshold from training ROC curve --------------
        y_tr_scores = clf.predict_proba(X_tr_dense)[:, 1]
        fpr_tr, tpr_tr, thresh_tr = roc_curve(y_tr, y_tr_scores)
        j_scores   = tpr_tr - fpr_tr
        best_idx   = int(np.argmax(j_scores))
        best_thresh = float(thresh_tr[best_idx])

        y_scores = clf.predict_proba(X_te_dense)[:, 1]
        y_pred   = (y_scores >= best_thresh).astype(int)

        yt = np.array(y_te)
        ys = np.array(y_scores)
        yp = np.array(y_pred)

        try:
            base = AgentEvaluator.calculate_agent_metrics(yt, ys, yp)
            tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            return {
                label: {
                    "n":                  len(y_te),
                    "train_n":            len(y_tr),
                    "roc_auc":            round(base["roc_auc"],  4),
                    "pr_auc":             round(base["pr_auc"],   4),
                    "f1_score":           round(base["f1_score"], 4),
                    "accuracy":           round(base["accuracy"], 4),
                    "specificity":        round(specificity, 4),
                    "sensitivity_recall": round(sensitivity, 4),
                    "decision_threshold": round(best_thresh, 4),
                    "confusion_matrix":   {
                        "tn": int(tn), "fp": int(fp),
                        "fn": int(fn), "tp": int(tp),
                    },
                }
            }
        except Exception as exc:
            return {label: {"error": str(exc), "n": len(y_te)}}

    def _multiclass_task(
        self,
        X_text: list, X_icd: list, X_icu: list,
        y: list, labels: list, task_name: str,
    ) -> dict:
        """Multi-class classification: TF-IDF(text) + TF-IDF(ICD) + has_icu_stay → LR."""
        if len(X_text) < 20 or len(set(y)) < 2:
            return {task_name: {"error": "insufficient_data", "n": len(X_text)}}

        idx = list(range(len(X_text)))
        try:
            idx_tr, idx_te, y_tr, y_te = train_test_split(
                idx, y,
                test_size    = self._test_size,
                random_state = self._random_state,
                stratify     = y,
            )
        except ValueError as exc:
            return {task_name: {"error": f"split_failed: {exc}", "n": len(X_text)}}

        X_tr_feat, X_te_feat = _build_features(
            [X_text[i] for i in idx_tr], [X_icd[i] for i in idx_tr],
            [X_text[i] for i in idx_te], [X_icd[i] for i in idx_te],
            icu_tr = [X_icu[i] for i in idx_tr],
            icu_te = [X_icu[i] for i in idx_te],
        )

        clf = _make_lr()
        clf.fit(X_tr_feat, y_tr)
        y_pred = clf.predict(X_te_feat)

        try:
            f1_w   = f1_score(y_te, y_pred, labels=labels, average="weighted", zero_division=0)
            f1_m   = f1_score(y_te, y_pred, labels=labels, average="macro",    zero_division=0)
            acc    = accuracy_score(y_te, y_pred)
            pc_raw = classification_report(
                y_te, y_pred, labels=labels, zero_division=0, output_dict=True,
            )
            per_class = {
                k: {
                    "precision": round(v["precision"], 4),
                    "recall":    round(v["recall"],    4),
                    "f1_score":  round(v["f1-score"],  4),
                }
                for k, v in pc_raw.items()
                if k in labels
            }
            return {
                task_name: {
                    "n":           len(y_te),
                    "train_n":     len(y_tr),
                    "accuracy":    round(acc,  4),
                    "f1_weighted": round(f1_w, 4),
                    "f1_macro":    round(f1_m, 4),
                    "per_class":   per_class,
                }
            }
        except Exception as exc:
            return {task_name: {"error": str(exc), "n": len(y_te)}}

    def _cancer_type_task(self, X_text: list, X_icd: list, y: list) -> dict:
        """Cancer type match accuracy: TF-IDF(text) + TF-IDF(ICD) → LinearSVC."""
        if len(X_text) < 20:
            return {"error": "insufficient_data", "n": len(X_text)}

        # Drop rare classes (< 2 samples) — would break stratified split
        counts  = Counter(y)
        mask    = [counts[lbl] >= 2 for lbl in y]
        X_tf    = [x for x, m in zip(X_text, mask) if m]
        X_if    = [x for x, m in zip(X_icd,  mask) if m]
        y_f     = [lbl for lbl, m in zip(y, mask) if m]
        dropped = len(y) - len(y_f)

        if len(X_tf) < 20 or len(set(y_f)) < 2:
            return {"error": "insufficient_data_after_rare_class_drop", "n": len(X_tf)}

        if dropped:
            logger.debug("[TFIDF] cancer_type: dropped %d samples with rare classes", dropped)

        idx = list(range(len(X_tf)))
        try:
            idx_tr, idx_te, y_tr, y_te = train_test_split(
                idx, y_f,
                test_size    = self._test_size,
                random_state = self._random_state,
                stratify     = y_f,
            )
        except ValueError as exc:
            return {"error": f"split_failed: {exc}", "n": len(X_tf)}

        X_tr_feat, X_te_feat = _build_features(
            [X_tf[i] for i in idx_tr], [X_if[i] for i in idx_tr],
            [X_tf[i] for i in idx_te], [X_if[i] for i in idx_te],
        )

        clf = _make_svm()
        clf.fit(X_tr_feat, y_tr)
        y_pred  = clf.predict(X_te_feat)

        labels = sorted(set(y_f))

        try:
            from sklearn.metrics import roc_auc_score
            y_proba = clf.predict_proba(X_te_feat)
            roc_auc_ovr_weighted = round(
                roc_auc_score(y_te, y_proba, multi_class="ovr",
                              average="weighted", labels=labels), 4,
            )
            roc_auc_ovr_macro = round(
                roc_auc_score(y_te, y_proba, multi_class="ovr",
                              average="macro", labels=labels), 4,
            )
        except Exception:
            roc_auc_ovr_weighted = None
            roc_auc_ovr_macro    = None

        f1_w   = round(f1_score(y_te, y_pred, labels=labels, average="weighted", zero_division=0), 4)
        f1_m   = round(f1_score(y_te, y_pred, labels=labels, average="macro",    zero_division=0), 4)
        acc    = round(accuracy_score(y_te, y_pred), 4)

        pc_raw = classification_report(
            y_te, y_pred, labels=labels, zero_division=0, output_dict=True,
        )
        per_class = {
            k: {
                "precision": round(v["precision"], 4),
                "recall":    round(v["recall"],    4),
                "f1_score":  round(v["f1-score"],  4),
                "support":   int(v["support"]),
            }
            for k, v in pc_raw.items() if k in labels
        }

        cm = confusion_matrix(y_te, y_pred, labels=labels).tolist()

        return {
            "n":                     len(y_te),
            "train_n":               len(y_tr),
            "match_accuracy":        round(accuracy_score(y_te, y_pred), 4),
            "f1_weighted":           f1_w,
            "f1_macro":              f1_m,
            "roc_auc_ovr_weighted":  roc_auc_ovr_weighted,
            "roc_auc_ovr_macro":     roc_auc_ovr_macro,
            "rare_dropped":          dropped,
            "per_class":             per_class,
            "confusion_matrix":      {
                "labels": labels,
                "matrix": cm,
            },
        }
