"""
Rule-based medical safety checks — fast, deterministic, no LLM required.

These act as a lightweight pre-filter and are also exposed as LangChain @tools
so the DeepAgent can call them explicitly during reasoning.

Improvement 1: Expanded keyword lists and SpO2 pattern detection.
The check_emergency_consistency function now covers a wider range of acute
presentations, reducing false negatives on the safety net metric.
"""

import re

# ---------------------------------------------------------------------------
# SpO2 pattern (shared with rule_engine.py)
# ---------------------------------------------------------------------------

_SPO2_PATTERN = re.compile(
    r'(?:spo2|o2\s*sat(?:uration)?|oxygen\s*sat(?:uration)?|sats)\s*(?:of\s*|:\s*|=\s*)?<?(\d{1,3})\s*%?',
    re.IGNORECASE,
)


def _has_low_spo2(text: str) -> bool:
    """Return True if SpO2/O2 saturation below 90% is found in text."""
    for m in _SPO2_PATTERN.finditer(text):
        try:
            if int(m.group(1)) < 90:
                return True
        except ValueError:
            pass
    return False


# ---------------------------------------------------------------------------
# Keyword lists — expanded to improve safety net coverage
# ---------------------------------------------------------------------------

_CRITICAL_SYMPTOM_KEYWORDS = [
    # Cardiac
    "cardiac arrest", "heart attack", "myocardial infarction", "stemi", "nstemi",
    "acs", "acute coronary syndrome", "ventricular fibrillation", "vf arrest",
    "pulseless", "asystole", "aortic dissection",
    # Neurological
    "stroke", "cva", "cerebrovascular accident", "ischaemic stroke", "ischemic stroke",
    "haemorrhagic stroke", "subarachnoid haemorrhage", "sah", "status epilepticus",
    "meningococcal", "bacterial meningitis",
    # Respiratory / circulatory
    "respiratory failure", "ards", "acute respiratory distress syndrome",
    "pulmonary embolism", "massive pe",
    # Haemodynamic
    "septic shock", "cardiogenic shock", "haemodynamic instability",
    "hemodynamic instability", "haemodynamically unstable",
    # Consciousness
    "loss of consciousness", "unresponsive", "unconscious", "comatose",
    "cardiac tamponade", "aneurysm rupture", "aortic aneurysm rupture",
    # Obstetric
    "eclampsia",
]

_EMERGENCY_SYMPTOM_KEYWORDS = [
    # Chest
    "chest pain", "chest tightness", "crushing chest", "chest pressure",
    "severe chest",
    # Respiratory
    "difficulty breathing", "shortness of breath", "severe dyspnoea",
    "severe breathlessness",
    # Neurological
    "severe headache", "thunderclap headache", "sudden weakness", "facial droop",
    "confusion", "altered consciousness", "decreased level of consciousness",
    "seizure", "convulsion", "paralysis",
    # Circulatory
    "severe bleeding", "haemorrhage", "massive haemorrhage", "active bleeding",
    "sepsis", "high fever with rigors",
    # Other acute
    "syncope", "near syncope", "sudden collapse",
]

_VALID_SEVERITIES = {"LOW", "HIGH", "CRITICAL"}


# ---------------------------------------------------------------------------
# Public rule checks
# ---------------------------------------------------------------------------

def check_emergency_consistency(
    symptoms: str, severity: str, emergency_care: str
) -> tuple[bool, str]:
    """
    Verify that the emergency care decision is consistent with the symptoms and severity.

    Checks:
      1. Critical symptom keywords → emergency care must be YES.
      2. CRITICAL severity → emergency care must be YES.
      3. Emergency-level symptoms → severity must not be LOW.
      4. SpO2 below 90% detected → severity must not be LOW.

    Returns (is_consistent, message).
    """
    symptoms_lower = symptoms.lower()
    emergency_upper = emergency_care.upper()
    severity_upper = severity.upper()

    has_critical = any(kw in symptoms_lower for kw in _CRITICAL_SYMPTOM_KEYWORDS)
    has_emergency = any(kw in symptoms_lower for kw in _EMERGENCY_SYMPTOM_KEYWORDS)

    if has_critical and emergency_upper != "YES":
        return False, (
            "Critical symptom keywords detected in patient description but emergency care "
            "was not recommended. This is a clinical safety concern."
        )

    if severity_upper == "CRITICAL" and emergency_upper != "YES":
        return False, "Severity is CRITICAL but emergency care is not flagged — inconsistent."

    if has_emergency and severity_upper == "LOW":
        return False, (
            "Emergency-level symptoms are present but severity is marked LOW — "
            "possible undertriage."
        )

    if severity_upper == "LOW" and _has_low_spo2(symptoms):
        return False, (
            "SpO2 below 90% detected in patient symptoms but severity is LOW — "
            "significant hypoxia is inconsistent with LOW severity."
        )

    return True, "Emergency care decision is consistent with symptoms."


def check_severity_validity(severity: str) -> tuple[bool, str]:
    """Validate that severity is a recognised value (LOW/HIGH/CRITICAL)."""
    if severity.upper() not in _VALID_SEVERITIES:
        return False, f"Unknown severity value '{severity}'. Expected: LOW, HIGH, or CRITICAL."
    return True, "Severity value is valid."
