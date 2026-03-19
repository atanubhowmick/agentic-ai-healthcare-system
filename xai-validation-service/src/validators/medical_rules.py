"""
Rule-based medical safety checks - fast, deterministic, no LLM required.
These act as a pre-filter before the LLM validation step.
"""

_CRITICAL_SYMPTOM_KEYWORDS = [
    "cardiac arrest", "heart attack", "myocardial infarction", "stroke",
    "aneurysm", "sepsis", "septic shock", "pulmonary embolism",
    "respiratory failure", "loss of consciousness", "unresponsive",
    "unconscious", "not breathing",
]

_EMERGENCY_SYMPTOM_KEYWORDS = [
    "chest pain", "difficulty breathing", "shortness of breath",
    "severe headache", "sudden weakness", "confusion", "severe bleeding",
    "high fever", "seizure", "paralysis", "severe chest",
]

_VALID_SEVERITIES = {"LOW", "HIGH", "CRITICAL"}


def check_emergency_consistency(symptoms: str, severity: str, emergency_care: str) -> tuple[bool, str]:
    """
    Verify that the emergency care decision is consistent with the symptoms and severity.
    Returns (is_consistent, message).
    """
    symptoms_lower = symptoms.lower()

    has_critical = any(kw in symptoms_lower for kw in _CRITICAL_SYMPTOM_KEYWORDS)
    has_emergency = any(kw in symptoms_lower for kw in _EMERGENCY_SYMPTOM_KEYWORDS)
    emergency_upper = emergency_care.upper()
    severity_upper = severity.upper()

    if has_critical and emergency_upper != "YES":
        return False, (
            "Critical symptom keywords detected in patient description but emergency care "
            "was not recommended. This is a clinical safety concern."
        )

    if severity_upper == "CRITICAL" and emergency_upper != "YES":
        return False, "Severity is CRITICAL but emergency care is not flagged - inconsistent."

    if has_emergency and severity_upper == "LOW":
        return False, (
            "Emergency-level symptoms are present but severity is marked LOW - "
            "possible under-triage."
        )

    return True, "Emergency care decision is consistent with symptoms."


def check_severity_validity(severity: str) -> tuple[bool, str]:
    """Validate that severity is a recognised value."""
    if severity.upper() not in _VALID_SEVERITIES:
        return False, f"Unknown severity value '{severity}'. Expected: LOW, HIGH, or CRITICAL."
    return True, "Severity value is valid."
