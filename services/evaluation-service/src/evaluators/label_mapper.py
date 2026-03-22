"""
MIMIC-IV → Cancer Agent Label Mapper
======================================
Maps MIMIC-IV evaluation set ground truth fields to the Cancer Agent's DiagnosisResult
schema so they can be compared for evaluation.

Ground truth derivation rules
------------------------------
severity (LOW / HIGH / CRITICAL / UNKNOWN)
    Inherited directly from load_mimic_data._infer_severity():
        DIED / EXPIRED / HOSPICE   → CRITICAL
        REHAB / SKILLED / ACUTE /
        FACILITY / TRANSFER        → HIGH
        everything else            → LOW
        missing / unknown          → UNKNOWN  (excluded from metrics)

hospitalizationNeeded (YES / NO)
    Derived from ICU/admission-based severity (set during extraction):
    CRITICAL or HIGH → YES
    LOW              → NO
    UNKNOWN          → None  (excluded)

emergencyCareNeeded (YES / NO)
    Primary:  admission_type in _EMERGENCY_ADMISSION_TYPES     → YES
              admission_type in _NON_EMERGENCY_ADMISSION_TYPES → NO
              (covers all 9 MIMIC-IV v3.1 admission_type values)
    Fallback (admission_type missing or unrecognised):
              CRITICAL                        → YES
              LOW                             → NO
              HIGH + emergency keyword in CC  → YES
              HIGH, no keyword                → NO
    UNKNOWN                                  → None (excluded)

cancer_type accuracy
    Normalised category match (broad ICD title → simplified label)
    OR fuzzy token similarity ≥ threshold 0.30 (difflib.SequenceMatcher, stdlib only)
"""

import re
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# MIMIC-IV v3.1 admission_type values that indicate emergency presentation.
# Covers all distinct values in physionet-data.mimiciv_3_1_hosp.admissions.
# ---------------------------------------------------------------------------

_EMERGENCY_ADMISSION_TYPES: frozenset[str] = frozenset({
    "EMERGENCY",
    "DIRECT EMER.",
    "URGENT",
})

_NON_EMERGENCY_ADMISSION_TYPES: frozenset[str] = frozenset({
    "ELECTIVE",
    "AMBULATORY OBSERVATION",
    "DIRECT OBSERVATION",
    "EU OBSERVATION",
    "OBSERVATION ADMIT",
    "SURGICAL SAME DAY ADMISSION",
})

# ---------------------------------------------------------------------------
# Emergency keyword set  (aligned with xai-validation-service/medical_rules.py)
# ---------------------------------------------------------------------------

_EMERGENCY_KEYWORDS: frozenset[str] = frozenset({
    "cardiac arrest", "heart attack", "myocardial infarction", " mi ",
    "stroke", "aneurysm", "sepsis", "respiratory failure",
    "pulmonary embolism", "chest pain", "difficulty breathing",
    "shortness of breath", " sob ", "severe headache", "sudden weakness",
    "confusion", "seizure", "haemoptysis", "hemoptysis",
    "bowel obstruction", "spinal cord compression",
})

# ---------------------------------------------------------------------------
# Broad cancer category normalisation map
# ICD long titles contain verbose phrases; map to simplified labels that
# overlap with the cancer agent's suspectedCancerType output.
# ---------------------------------------------------------------------------

_CANCER_CATEGORY_MAP: dict[str, str] = {
    "lung":          "Lung Cancer",
    "bronch":        "Lung Cancer",
    "breast":        "Breast Cancer",
    "colon":         "Colorectal Cancer",
    "rectal":        "Colorectal Cancer",
    "colorectal":    "Colorectal Cancer",
    "prostate":      "Prostate Cancer",
    "bladder":       "Bladder Cancer",
    "kidney":        "Kidney Cancer",
    "renal cell":    "Kidney Cancer",
    "leuk":          "Leukaemia",
    "lymphoma":      "Lymphoma",
    "melanoma":      "Melanoma",
    "pancrea":       "Pancreatic Cancer",
    "liver":         "Liver Cancer",
    "hepat":         "Liver Cancer",
    "ovari":         "Ovarian Cancer",
    "cervix":        "Cervical Cancer",
    "cervi":         "Cervical Cancer",
    "uteri":         "Uterine Cancer",
    "thyroid":       "Thyroid Cancer",
    "brain":         "Brain Cancer",
    "glioma":        "Brain Cancer",
    "stomach":       "Gastric Cancer",
    "gastric":       "Gastric Cancer",
    "oesophag":      "Oesophageal Cancer",
    "esophag":       "Oesophageal Cancer",
    "myeloma":       "Multiple Myeloma",
    "testicular":    "Testicular Cancer",
    "head and neck": "Head and Neck Cancer",
    "oral":          "Head and Neck Cancer",
    "pharynx":       "Head and Neck Cancer",
    "larynx":        "Head and Neck Cancer",
    "skin":          "Skin Cancer",
}


class MimicLabelMapper:
    """
    Stateless converter: MIMIC-IV evaluation record → ground truth labels
    matching the Cancer Agent's DiagnosisResult field names.
    """

    @staticmethod
    def map_severity(severity: str) -> str | None:
        """Pass-through for severity. Returns None for UNKNOWN (excluded from metrics)."""
        s = (severity or "").upper().strip()
        return s if s in ("LOW", "HIGH", "CRITICAL") else None

    @staticmethod
    def map_hospitalization(severity: str) -> str | None:
        """HIGH or CRITICAL → YES (admitted). LOW → NO (home). UNKNOWN → None (excluded)."""
        s = (severity or "").upper().strip()
        if s in ("HIGH", "CRITICAL"):
            return "YES"
        if s == "LOW":
            return "NO"
        return None

    @staticmethod
    def map_emergency(admission_type: str, severity: str, chief_complaint: str) -> str | None:
        """
        Primary: admission_type=EMERGENCY → YES; URGENT/ELECTIVE → NO.
        Fallback (admission_type missing): CRITICAL → YES; LOW → NO;
            HIGH + emergency keyword → YES, else NO. UNKNOWN → None.
        """
        a = (admission_type or "").upper().strip()
        if a in _EMERGENCY_ADMISSION_TYPES:
            return "YES"
        if a in _NON_EMERGENCY_ADMISSION_TYPES:
            return "NO"

        # Fallback to severity + keyword approach when admission_type is absent
        s = (severity or "").upper().strip()
        if s == "CRITICAL":
            return "YES"
        if s == "LOW":
            return "NO"
        if s == "HIGH":
            cc = " " + re.sub(r"\s+", " ", (chief_complaint or "").lower()) + " "
            for kw in _EMERGENCY_KEYWORDS:
                if kw in cc:
                    return "YES"
            return "NO"
        return None

    @staticmethod
    def normalise_cancer_type(raw_label: str) -> str:
        """Map verbose MIMIC ICD long titles to a simplified category label."""
        lower = (raw_label or "").lower()
        for keyword, category in _CANCER_CATEGORY_MAP.items():
            if keyword in lower:
                return category
        return "Other Cancer"

    @staticmethod
    def cancer_type_similarity(predicted: str, ground_truth: str) -> float:
        """Fuzzy similarity score 0.0 – 1.0 using stdlib SequenceMatcher."""
        p = (predicted or "").lower().strip()
        g = (ground_truth or "").lower().strip()
        if not p or not g:
            return 0.0
        return SequenceMatcher(None, p, g).ratio()

    @staticmethod
    def is_cancer_type_correct(
        predicted: str,
        ground_truth: str,
        similarity_threshold: float = 0.30,
    ) -> int:
        """Binary correctness (1/0): category match OR fuzzy similarity ≥ threshold."""
        pred_cat = MimicLabelMapper.normalise_cancer_type(predicted)
        gt_cat   = MimicLabelMapper.normalise_cancer_type(ground_truth)
        if pred_cat == gt_cat and pred_cat != "Other Cancer":
            return 1
        return 1 if MimicLabelMapper.cancer_type_similarity(predicted, ground_truth) >= similarity_threshold else 0

    @staticmethod
    def encode_binary(value: str) -> int | None:
        """YES → 1, NO → 0, anything else → None (excluded from metrics)."""
        v = (value or "").upper().strip()
        if v == "YES":
            return 1
        if v == "NO":
            return 0
        return None
