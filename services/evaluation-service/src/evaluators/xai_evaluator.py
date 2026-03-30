"""
XAI Validation Service Evaluator.

Original options:
  Option 1 — Validation Decision Accuracy
  Option 2 — Safety Net Effectiveness
  Option 4 — Rule Engine Coverage
  Option 6 — Over-rejection Rate

New XAI quality metrics:
  Fidelity      — Does the explanation faithfully reflect the decision?
                  Tested by perturbing severity and checking whether both
                  the decision AND the explanation change accordingly.
  Stability     — Does the same input always produce the same recommendation?
                  Tested by sending identical payloads 3× and measuring agreement.
  Consistency   — Do near-identical inputs (paraphrased symptoms) produce the
                  same recommendation?
  Sparsity      — How focused is the explanation? (avg key_concerns count,
                  avg summary word count — collected for free during Option 1 calls)
  Interpretability — How readable is the validation_summary?
                  Measured via Flesch Reading Ease (computed inline, no extra calls).
"""

from __future__ import annotations

import datetime
import re
import time
from typing import Any

import httpx

from core.config import XAI_SERVICE_URL
from core.mongo_client import (
    load_evaluation_cases,
    save_xai_report,
)
from log.logger import logger


# ---------------------------------------------------------------------------
# Rule replication (mirrors xai-validation-service/src/validators/medical_rules.py)
# ---------------------------------------------------------------------------

_CRITICAL_SYMPTOM_KEYWORDS = [
    # Full phrases (natural language)
    "cardiac arrest", "heart attack", "myocardial infarction", "stroke",
    "aneurysm", "sepsis", "septic shock", "pulmonary embolism",
    "respiratory failure", "loss of consciousness", "unresponsive",
    "unconscious", "not breathing",
    # MIMIC clinical abbreviations
    "mi", "stemi", "nstemi", "cva", "tia", "pe ", "ards",
    "septic", "cardiac arrest",
]

_EMERGENCY_SYMPTOM_KEYWORDS = [
    # Full phrases (natural language)
    "chest pain", "difficulty breathing", "shortness of breath",
    "severe headache", "sudden weakness", "confusion", "severe bleeding",
    "high fever", "seizure", "paralysis", "severe chest",
    # MIMIC clinical abbreviations
    "cp ", "sob", "doe", "dyspnea", "syncope", "altered mental",
    "ams", "altered ms", "hypotension", "hypoxia", "hemorrhage",
    "bleeding", "fall", "chest tightness",
]


def _rule_triggered(symptoms: str, severity: str, emergency_care: str) -> bool:
    """Return True if the deterministic rule engine would flag this combination."""
    s = symptoms.lower()
    has_critical = any(kw in s for kw in _CRITICAL_SYMPTOM_KEYWORDS)
    has_emergency = any(kw in s for kw in _EMERGENCY_SYMPTOM_KEYWORDS)
    em = emergency_care.upper()
    sv = severity.upper()
    if has_critical and em != "YES":
        return True
    if sv == "CRITICAL" and em != "YES":
        return True
    if has_emergency and sv == "LOW":
        return True
    return False


# ---------------------------------------------------------------------------
# Readability helpers (Flesch Reading Ease — no external library needed)
# ---------------------------------------------------------------------------

def _count_syllables(word: str) -> int:
    """Crude syllable counter based on vowel groups."""
    word = word.lower().strip(".,!?;:")
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Silent 'e' at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_reading_ease(text: str) -> float:
    """
    Flesch Reading Ease score (0–100).
    90–100: very easy | 60–70: standard | 30–50: difficult | <30: very difficult.
    Medical professional reports typically score 20–50.
    """
    sentences = max(1, len(re.split(r"[.!?]+", text.strip())))
    words = text.split()
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
    return round(max(0.0, min(100.0, score)), 2)


def _compute_text_metrics(result: dict) -> dict:
    """Extract sparsity and interpretability metrics from one XAI result."""
    summary = result.get("validation_summary", "") or ""
    concerns = result.get("key_concerns", []) or []
    return {
        "key_concerns_count": len(concerns),
        "summary_word_count": len(summary.split()),
        "flesch_reading_ease": _flesch_reading_ease(summary),
    }


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

def _build_diagnosis_details(symptoms: str, severity: str, emergency_care: str) -> str:
    """
    Construct a minimal but realistic diagnosisDetails string so the XAI LLM
    receives clinical context rather than a raw dict fallback.
    """
    em_text = (
        "Emergency care is indicated"
        if emergency_care.upper() == "YES"
        else "No emergency care required"
    )
    return (
        f"Patient presented with: {symptoms}. "
        f"Clinical assessment indicates {severity.lower()} severity oncological concern. "
        f"{em_text} based on current clinical indicators."
    )


def _paraphrase_symptoms(symptoms: str) -> str:
    """Minimal paraphrase for consistency testing — same meaning, different wording."""
    s = symptoms.strip().rstrip(".")
    return f"Patient presenting with {s.lower()}."


def _perturb_severity(severity: str) -> str:
    """Flip severity to a different value for fidelity perturbation tests."""
    mapping = {"LOW": "CRITICAL", "HIGH": "LOW", "CRITICAL": "LOW"}
    return mapping.get(severity.upper(), "HIGH")


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _call_xai(
    patient_id: str,
    symptoms: str,
    severity: str,
    emergency_care: str,
    hospitalization_needed: str,
    timeout: float = 30.0,
) -> dict[str, Any] | None:
    """
    POST to /xai-validator/validate-diagnosis.
    Returns the parsed result dict or None on failure.
    """
    payload = {
        "patient_id": patient_id,
        "symptoms": symptoms,
        "specialist_agent": "Cancer_Oncology_Specialist",
        "diagnosis": {
            "diagnosisDetails": _build_diagnosis_details(symptoms, severity, emergency_care),
            "severity": severity,
            "emergencyCareNeeded": emergency_care,
            "hospitalizationNeeded": hospitalization_needed,
        },
    }
    url = f"{XAI_SERVICE_URL.rstrip('/')}/xai-validator/validate-diagnosis"
    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("is_success") and data.get("payload"):
            return data["payload"].get("result")
    except Exception as exc:
        logger.warning("[XAI_EVAL] HTTP call failed for %s: %s", patient_id, exc)
    return None


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _label_emergency(row: dict) -> str | None:
    adm = str(row.get("admission_type", "")).upper()
    if adm == "EMERGENCY":
        return "YES"
    if adm in ("ELECTIVE", "OBSERVATION ADMIT", "EU OBSERVATION",
               "AMBULATORY OBSERVATION", "DIRECT EMER."):
        return "NO"
    return None


def _label_severity(row: dict) -> str | None:
    loc = str(row.get("discharge_location", "")).upper()
    has_icu = bool(row.get("has_icu_stay", False))
    if "DIED" in loc or "HOSPICE" in loc or has_icu:
        return "CRITICAL"
    if "HOME" in loc or "REHAB" in loc or "SNF" in loc or "CHRONIC" in loc:
        return "LOW"
    return "HIGH"


def _label_hospitalization(severity: str | None) -> str | None:
    if severity in ("CRITICAL", "HIGH"):
        return "YES"
    if severity == "LOW":
        return "NO"
    return None


def _has_keywords(text: str) -> bool:
    s = text.lower()
    return (
        any(kw in s for kw in _CRITICAL_SYMPTOM_KEYWORDS) or
        any(kw in s for kw in _EMERGENCY_SYMPTOM_KEYWORDS)
    )


def _safe_cases(cases: list[dict]) -> list[tuple[dict, str, str, str]]:
    """
    Filter and label cases, keeping only text-safe ones (no emergency/critical keywords).
    Returns list of (row, symptoms, severity, emergency).
    """
    out = []
    for row in cases:
        severity = _label_severity(row)
        emergency = _label_emergency(row)
        if severity is None or emergency is None:
            continue
        symptoms = str(row.get("chief_complaint", "") or row.get("diagnoses_text", ""))
        if not _has_keywords(symptoms):
            out.append((row, symptoms, severity, emergency))
    return out


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class XaiEvaluator:
    """
    Evaluate XAI validation service across Options 1, 2, 4, 6 and
    new XAI quality metrics: Fidelity, Stability, Consistency, Sparsity,
    Interpretability.
    """

    def __init__(
        self,
        max_cases: int = 0,
        max_correct_cases: int = 150,
        max_undertriage_cases: int = 50,
        max_stability_cases: int = 30,
        max_fidelity_cases: int = 30,
        max_consistency_cases: int = 30,
    ) -> None:
        self.max_cases = max_cases
        self.max_correct_cases = max_correct_cases
        self.max_undertriage_cases = max_undertriage_cases
        self.max_stability_cases = max_stability_cases
        self.max_fidelity_cases = max_fidelity_cases
        self.max_consistency_cases = max_consistency_cases

    # ------------------------------------------------------------------
    def run_evaluation(self) -> None:
        logger.info("[XAI_EVAL] Starting XAI evaluation | max_cases=%s", self.max_cases or "all")
        started = time.time()

        cases = load_evaluation_cases(max_cases=self.max_cases)
        logger.info("[XAI_EVAL] Loaded %d cases from MongoDB", len(cases))

        # Options 1 & 6 + Sparsity + Interpretability (collected during same calls)
        correct_cases = cases[:self.max_correct_cases] if self.max_correct_cases else cases
        opt1_results = self._eval_correct_diagnoses(correct_cases)

        # Option 2
        opt2_results = self._eval_undertriage(cases)

        # Option 4 (no HTTP calls)
        opt4_results = self._eval_rule_coverage(cases)

        # New XAI quality metrics
        stability_results     = self._eval_stability(cases)
        fidelity_results      = self._eval_fidelity(cases)
        consistency_results   = self._eval_consistency(cases)

        elapsed = round(time.time() - started, 1)
        logger.info("[XAI_EVAL] Done in %.1f s", elapsed)

        report = {
            "run_at":                      datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_cases_loaded":          len(cases),
            "elapsed_seconds":             elapsed,
            "option_1_decision_accuracy":  opt1_results["opt1"],
            "option_6_over_rejection_rate": opt1_results["opt6"],
            "option_2_safety_net_effectiveness": opt2_results,
            "option_4_rule_engine_coverage": opt4_results,
            "xai_sparsity":               opt1_results["sparsity"],
            "xai_interpretability":       opt1_results["interpretability"],
            "xai_stability":              stability_results,
            "xai_fidelity":               fidelity_results,
            "xai_consistency":            consistency_results,
        }
        save_xai_report(report)
        logger.info("[XAI_EVAL] Report saved.")

    # ------------------------------------------------------------------
    # Option 1 + 6 + Sparsity + Interpretability
    # ------------------------------------------------------------------
    def _eval_correct_diagnoses(self, cases: list[dict]) -> dict:
        logger.info("[XAI_EVAL] Option 1/6 + Sparsity + Interpretability …")

        approved = over_rejected = skipped = keyword_excluded = called = 0
        text_metrics_list: list[dict] = []

        for i, row in enumerate(cases):
            severity  = _label_severity(row)
            emergency = _label_emergency(row)
            if severity is None or emergency is None:
                skipped += 1
                continue

            symptoms = str(row.get("chief_complaint", "") or row.get("diagnoses_text", ""))
            if _has_keywords(symptoms):
                keyword_excluded += 1
                continue

            hospitalization = _label_hospitalization(severity)
            patient_id = f"eval_correct_{row.get('subject_id', i)}"

            result = _call_xai(patient_id, symptoms, severity, emergency, hospitalization or "NO")
            if result is None:
                skipped += 1
                continue
            called += 1

            recommendation = result.get("recommendation", "").upper()
            if recommendation == "APPROVE":
                approved += 1
            else:
                over_rejected += 1

            # Collect text metrics for Sparsity + Interpretability (free, no extra calls)
            text_metrics_list.append(_compute_text_metrics(result))

            if called % 50 == 0:
                logger.info("[XAI_EVAL] Correct-diagnosis progress: %d called", called)

        logger.info(
            "[XAI_EVAL] Option 1/6 complete | called=%d keyword_excluded=%d skipped=%d",
            called, keyword_excluded, skipped,
        )

        total = approved + over_rejected
        approval_accuracy   = round(approved       / total, 4) if total else 0.0
        over_rejection_rate = round(over_rejected  / total, 4) if total else 0.0

        # Sparsity
        sparsity = self._aggregate_sparsity(text_metrics_list)

        # Interpretability
        interpretability = self._aggregate_interpretability(text_metrics_list)

        opt1 = {
            "total_tested":        total,
            "correctly_approved":  approved,
            "incorrectly_flagged": over_rejected,
            "approval_accuracy":   approval_accuracy,
            "keyword_excluded":    keyword_excluded,
            "skipped":             skipped,
            "note": (
                "Only text-safe cases (no emergency/critical keywords) are tested. "
                "Any REJECT/REVIEW is a genuine false positive from the LLM path."
            ),
        }
        opt6 = {
            "total_correct_diagnoses_tested":   total,
            "approved":                         approved,
            "incorrectly_rejected_or_reviewed": over_rejected,
            "over_rejection_rate":              over_rejection_rate,
            "specificity":                      approval_accuracy,
            "keyword_excluded":                 keyword_excluded,
            "skipped":                          skipped,
        }

        return {"opt1": opt1, "opt6": opt6, "sparsity": sparsity, "interpretability": interpretability}

    # ------------------------------------------------------------------
    # Option 2
    # ------------------------------------------------------------------
    def _eval_undertriage(self, cases: list[dict]) -> dict:
        logger.info("[XAI_EVAL] Option 2: under-triage detection …")

        severe_cases = [
            row for row in cases if _label_severity(row) in ("HIGH", "CRITICAL")
        ]
        if self.max_undertriage_cases and len(severe_cases) > self.max_undertriage_cases:
            severe_cases = severe_cases[:self.max_undertriage_cases]

        logger.info("[XAI_EVAL] Under-triage: %d severe cases selected", len(severe_cases))

        detected = missed = skipped = called = 0

        for i, row in enumerate(severe_cases):
            symptoms   = str(row.get("chief_complaint", "") or row.get("diagnoses_text", ""))
            patient_id = f"eval_undertriage_{row.get('subject_id', i)}"

            result = _call_xai(patient_id, symptoms, "LOW", "NO", "NO")
            if result is None:
                skipped += 1
                continue
            called += 1

            if result.get("recommendation", "").upper() in ("REJECT", "REVIEW"):
                detected += 1
            else:
                missed += 1

            if (i + 1) % 20 == 0:
                logger.info("[XAI_EVAL] Under-triage progress: %d/%d called", called, i + 1)

        sensitivity = round(detected / (detected + missed), 4) if (detected + missed) else 0.0
        miss_rate   = round(missed   / (detected + missed), 4) if (detected + missed) else 0.0

        logger.info(
            "[XAI_EVAL] Under-triage complete | detected=%d missed=%d sensitivity=%.4f",
            detected, missed, sensitivity,
        )

        return {
            "severe_cases_selected":    len(severe_cases),
            "total_undertriage_tested": called,
            "correctly_detected":       detected,
            "missed_undertriages":      missed,
            "sensitivity":              sensitivity,
            "miss_rate":                miss_rate,
            "skipped":                  skipped,
            "note": (
                "CRITICAL/HIGH case reported as severity=LOW, emergencyCareNeeded=NO. "
                "Sensitivity = fraction correctly rejected/reviewed."
            ),
        }

    # ------------------------------------------------------------------
    # Option 4
    # ------------------------------------------------------------------
    def _eval_rule_coverage(self, cases: list[dict]) -> dict:
        logger.info("[XAI_EVAL] Option 4: rule engine coverage …")

        total = rule_hit = rule_miss = skipped = 0
        llm_path_has_critical = llm_path_has_emergency = llm_path_text_safe = 0
        llm_path_keyword_severity_mismatch = 0

        for row in cases:
            severity  = _label_severity(row)
            emergency = _label_emergency(row)
            if severity is None or emergency is None:
                skipped += 1
                continue

            symptoms = str(row.get("chief_complaint", "") or row.get("diagnoses_text", ""))
            s_lower  = symptoms.lower()
            total   += 1

            has_critical  = any(kw in s_lower for kw in _CRITICAL_SYMPTOM_KEYWORDS)
            has_emergency = any(kw in s_lower for kw in _EMERGENCY_SYMPTOM_KEYWORDS)

            if _rule_triggered(symptoms, severity, emergency):
                rule_hit += 1
            else:
                rule_miss += 1
                if has_critical:
                    llm_path_has_critical += 1
                    if severity == "LOW":
                        llm_path_keyword_severity_mismatch += 1
                elif has_emergency:
                    llm_path_has_emergency += 1
                    if severity == "LOW":
                        llm_path_keyword_severity_mismatch += 1
                else:
                    llm_path_text_safe += 1

        rule_coverage = round(rule_hit  / total, 4) if total else 0.0
        llm_path_rate = round(rule_miss / total, 4) if total else 0.0
        mismatch_rate = round(llm_path_keyword_severity_mismatch / rule_miss, 4) if rule_miss else 0.0

        logger.info("[XAI_EVAL] Rule coverage: hit=%d miss=%d coverage=%.4f", rule_hit, rule_miss, rule_coverage)

        return {
            "total_cases_evaluated":     total,
            "rule_engine_hit":           rule_hit,
            "rule_engine_miss_llm_path": rule_miss,
            "rule_coverage_rate":        rule_coverage,
            "llm_path_rate":             llm_path_rate,
            "skipped":                   skipped,
            "llm_path_breakdown": {
                "text_safe_no_keywords":          llm_path_text_safe,
                "has_emergency_keywords":         llm_path_has_emergency,
                "has_critical_keywords":          llm_path_has_critical,
                "keyword_severity_mismatch":      llm_path_keyword_severity_mismatch,
                "keyword_severity_mismatch_rate": mismatch_rate,
            },
            "note": (
                "Rule coverage = fraction caught by deterministic rules (no LLM needed). "
                "LLM path = cases forwarded to LLM. "
                "keyword_severity_mismatch = emergency/critical keyword text but severity=LOW "
                "that the rules missed — these are the highest-risk cases for the LLM."
            ),
        }

    # ------------------------------------------------------------------
    # Stability: send identical payload 3× — measure recommendation agreement
    # ------------------------------------------------------------------
    def _eval_stability(self, cases: list[dict]) -> dict:
        logger.info("[XAI_EVAL] Stability: sending identical payloads 3× …")

        safe = _safe_cases(cases)
        if self.max_stability_cases:
            safe = safe[:self.max_stability_cases]

        logger.info("[XAI_EVAL] Stability: %d cases selected", len(safe))

        stable = unstable = skipped = 0
        REPEATS = 3

        for idx, (row, symptoms, severity, emergency) in enumerate(safe):
            hospitalization = _label_hospitalization(severity) or "NO"
            patient_id = f"stab_{row.get('subject_id', idx)}"

            recommendations = []
            for r in range(REPEATS):
                result = _call_xai(f"{patient_id}_r{r}", symptoms, severity, emergency, hospitalization)
                if result is None:
                    break
                recommendations.append(result.get("recommendation", "").upper())

            if len(recommendations) < REPEATS:
                skipped += 1
                continue

            if len(set(recommendations)) == 1:
                stable += 1
            else:
                unstable += 1
                logger.debug(
                    "[XAI_EVAL] Unstable case %s: %s", patient_id, recommendations
                )

            if (idx + 1) % 10 == 0:
                logger.info("[XAI_EVAL] Stability progress: %d/%d", idx + 1, len(safe))

        total = stable + unstable
        stability_rate = round(stable / total, 4) if total else 0.0
        logger.info("[XAI_EVAL] Stability complete | stable=%d unstable=%d rate=%.4f", stable, unstable, stability_rate)

        return {
            "cases_tested":           total,
            "stable_cases":           stable,
            "unstable_cases":         unstable,
            "stability_rate":         stability_rate,
            "repeats_per_case":       REPEATS,
            "skipped":                skipped,
            "note": (
                f"Each case sent {REPEATS}× with identical payload. "
                "Stability rate = fraction where all repeats gave the same recommendation. "
                "High stability expected since LLM uses temperature=0."
            ),
        }

    # ------------------------------------------------------------------
    # Fidelity: perturb severity, check decision + explanation change
    # ------------------------------------------------------------------
    def _eval_fidelity(self, cases: list[dict]) -> dict:
        logger.info("[XAI_EVAL] Fidelity: perturbation tests …")

        safe = _safe_cases(cases)
        if self.max_fidelity_cases:
            safe = safe[:self.max_fidelity_cases]

        logger.info("[XAI_EVAL] Fidelity: %d cases selected", len(safe))

        decision_changed      = 0   # recommendation changed after perturbation (good)
        decision_unchanged    = 0   # recommendation did not change (bad — model ignores severity)
        explanation_faithful  = 0   # explanation references severity change when decision changed
        explanation_silent    = 0   # explanation does not mention severity change
        skipped = 0

        severity_terms = {"severity", "critical", "low", "high", "serious", "mild", "urgent", "acute"}

        for idx, (row, symptoms, severity, emergency) in enumerate(safe):
            hospitalization = _label_hospitalization(severity) or "NO"
            perturbed_sev   = _perturb_severity(severity)
            perturbed_hosp  = _label_hospitalization(perturbed_sev) or "NO"
            patient_id      = f"fid_{row.get('subject_id', idx)}"

            # Original call
            r_orig = _call_xai(f"{patient_id}_orig", symptoms, severity, emergency, hospitalization)
            # Perturbed call (severity flipped)
            r_pert = _call_xai(f"{patient_id}_pert", symptoms, perturbed_sev, emergency, perturbed_hosp)

            if r_orig is None or r_pert is None:
                skipped += 1
                continue

            rec_orig = r_orig.get("recommendation", "").upper()
            rec_pert = r_pert.get("recommendation", "").upper()

            if rec_orig != rec_pert:
                decision_changed += 1
                # Check explanation faithfulness: does the perturbed explanation mention severity?
                pert_summary  = (r_pert.get("validation_summary", "") or "").lower()
                pert_concerns = " ".join(r_pert.get("key_concerns", [])).lower()
                combined      = pert_summary + " " + pert_concerns
                if any(term in combined for term in severity_terms):
                    explanation_faithful += 1
                else:
                    explanation_silent += 1
            else:
                decision_unchanged += 1

            if (idx + 1) % 10 == 0:
                logger.info("[XAI_EVAL] Fidelity progress: %d/%d", idx + 1, len(safe))

        total = decision_changed + decision_unchanged
        decision_sensitivity = round(decision_changed    / total, 4) if total else 0.0
        explanation_fidelity = round(explanation_faithful / decision_changed, 4) if decision_changed else 0.0

        logger.info(
            "[XAI_EVAL] Fidelity complete | changed=%d unchanged=%d exp_faithful=%d",
            decision_changed, decision_unchanged, explanation_faithful,
        )

        return {
            "cases_tested":                total,
            "decision_changed_on_perturbation": decision_changed,
            "decision_unchanged":          decision_unchanged,
            "decision_sensitivity_rate":   decision_sensitivity,
            "explanation_faithful":        explanation_faithful,
            "explanation_silent":          explanation_silent,
            "explanation_fidelity_rate":   explanation_fidelity,
            "skipped":                     skipped,
            "note": (
                "Severity is flipped (LOW→CRITICAL, HIGH→LOW, CRITICAL→LOW) per case. "
                "decision_sensitivity_rate = fraction where recommendation changed (model responds to severity). "
                "explanation_fidelity_rate = fraction of changed decisions where explanation "
                "references severity — measures whether the explanation reflects the actual decision trigger."
            ),
        }

    # ------------------------------------------------------------------
    # Consistency: paraphrase symptoms — check recommendation agreement
    # ------------------------------------------------------------------
    def _eval_consistency(self, cases: list[dict]) -> dict:
        logger.info("[XAI_EVAL] Consistency: paraphrase tests …")

        safe = _safe_cases(cases)
        if self.max_consistency_cases:
            safe = safe[:self.max_consistency_cases]

        logger.info("[XAI_EVAL] Consistency: %d cases selected", len(safe))

        consistent = inconsistent = skipped = 0

        for idx, (row, symptoms, severity, emergency) in enumerate(safe):
            hospitalization = _label_hospitalization(severity) or "NO"
            paraphrased     = _paraphrase_symptoms(symptoms)
            patient_id      = f"cons_{row.get('subject_id', idx)}"

            r_orig = _call_xai(f"{patient_id}_orig", symptoms,    severity, emergency, hospitalization)
            r_para = _call_xai(f"{patient_id}_para", paraphrased, severity, emergency, hospitalization)

            if r_orig is None or r_para is None:
                skipped += 1
                continue

            rec_orig = r_orig.get("recommendation", "").upper()
            rec_para = r_para.get("recommendation", "").upper()

            if rec_orig == rec_para:
                consistent += 1
            else:
                inconsistent += 1
                logger.debug(
                    "[XAI_EVAL] Inconsistent case %s: orig=%s para=%s",
                    patient_id, rec_orig, rec_para,
                )

            if (idx + 1) % 10 == 0:
                logger.info("[XAI_EVAL] Consistency progress: %d/%d", idx + 1, len(safe))

        total = consistent + inconsistent
        consistency_rate = round(consistent / total, 4) if total else 0.0
        logger.info(
            "[XAI_EVAL] Consistency complete | consistent=%d inconsistent=%d rate=%.4f",
            consistent, inconsistent, consistency_rate,
        )

        return {
            "cases_tested":      total,
            "consistent":        consistent,
            "inconsistent":      inconsistent,
            "consistency_rate":  consistency_rate,
            "skipped":           skipped,
            "note": (
                "Each case tested with original symptoms and a rephrased version "
                "(e.g. 'WEAKNESS' → 'Patient presenting with weakness.'). "
                "Consistency rate = fraction where both phrasings give the same recommendation."
            ),
        }

    # ------------------------------------------------------------------
    # Sparsity aggregation (from text_metrics collected during Option 1 calls)
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_sparsity(metrics: list[dict]) -> dict:
        if not metrics:
            return {"note": "No data collected."}

        counts = [m["key_concerns_count"] for m in metrics]
        words  = [m["summary_word_count"]  for m in metrics]
        avg_concerns   = round(sum(counts) / len(counts), 2)
        avg_word_count = round(sum(words)  / len(words),  2)
        dist = {1: 0, 2: 0, 3: 0}
        for c in counts:
            if c <= 1:   dist[1] += 1
            elif c == 2: dist[2] += 1
            else:        dist[3] += 1

        return {
            "samples":                         len(metrics),
            "avg_key_concerns_per_response":   avg_concerns,
            "avg_summary_word_count":          avg_word_count,
            "key_concerns_distribution": {
                "1_concern":     dist[1],
                "2_concerns":    dist[2],
                "3plus_concerns": dist[3],
            },
            "note": (
                "Sparsity measures how focused explanations are. "
                "Ideal: 1–2 key concerns, summary under 80 words. "
                "Collected from Option 1 calls at no extra cost."
            ),
        }

    # ------------------------------------------------------------------
    # Interpretability aggregation (Flesch Reading Ease)
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_interpretability(metrics: list[dict]) -> dict:
        if not metrics:
            return {"note": "No data collected."}

        scores = [m["flesch_reading_ease"] for m in metrics]
        avg_score = round(sum(scores) / len(scores), 2)
        # Classify
        buckets = {"very_easy_90_100": 0, "standard_60_89": 0,
                   "difficult_30_59": 0, "very_difficult_0_29": 0}
        for s in scores:
            if s >= 90:   buckets["very_easy_90_100"]       += 1
            elif s >= 60: buckets["standard_60_89"]         += 1
            elif s >= 30: buckets["difficult_30_59"]        += 1
            else:         buckets["very_difficult_0_29"]    += 1

        return {
            "samples":                  len(metrics),
            "avg_flesch_reading_ease":  avg_score,
            "score_distribution":       buckets,
            "note": (
                "Flesch Reading Ease (0–100): 90–100=very easy, 60–70=standard, "
                "30–50=difficult, <30=very difficult. "
                "Medical professional reports typically score 20–50. "
                "Collected from Option 1 calls at no extra cost."
            ),
        }
