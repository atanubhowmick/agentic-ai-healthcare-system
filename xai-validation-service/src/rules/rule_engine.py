# Evaluates deterministic safety rules before the LLM validation step.
# REJECT rules take precedence over REVIEW rules.

import re
from log.logger import logger

# Matches: "SpO2 88%", "O2 sat 82", "oxygen saturation: 78%", "sats 86", "O2 sats 84"
_SPO2_PATTERN = re.compile(
    r'(?:spo2|o2\s*sat(?:uration)?|oxygen\s*sat(?:uration)?|sats)\s*(?:of\s*|:\s*|=\s*)?<?(\d{1,3})\s*%?',
    re.IGNORECASE,
)


def _has_low_spo2(text: str) -> bool:
    """Return True if the text contains an SpO2/O2 saturation reading below 90%."""
    for m in _SPO2_PATTERN.finditer(text):
        try:
            if int(m.group(1)) < 90:
                return True
        except ValueError:
            pass
    return False



def _field_text(field: str, symptoms: str, diagnosis_text: str, treatment_text: str) -> str:
    return {
        "symptoms": symptoms,
        "diagnosis": diagnosis_text,
        "treatment": treatment_text,
    }.get(field, "")


def _keywords_in_fields(
    keywords: list[str],
    fields: list[str],
    symptoms: str,
    diagnosis_text: str,
    treatment_text: str,
) -> bool:
    """Return True if ANY keyword is found in the combined text of the specified fields."""
    if not keywords:
        return True  # empty keyword list = condition always met
    combined = " ".join(
        _field_text(f, symptoms, diagnosis_text, treatment_text) for f in fields
    ).lower()
    return any(kw.lower() in combined for kw in keywords)



def _rule_fires(
    rule: dict,
    symptoms: str,
    severity: str,
    emergency_care: str,
    diagnosis_text: str,
    treatment_text: str,
) -> bool:
    """Return True if the rule fires for the given request inputs."""
    sev_upper = severity.upper()
    emg_upper = emergency_care.upper()

    # -- SpO2 special rule --------------------------------------------------
    if rule.get("_spo2_rule"):
        combined = f"{symptoms} {diagnosis_text}"
        if _has_low_spo2(combined):
            forbidden_sev = rule.get("severity_is", [])
            return sev_upper in forbidden_sev if forbidden_sev else True
        return False

    # -- Primary keyword condition ------------------------------------------
    keywords = rule.get("keywords", [])
    fields = rule.get("fields", [])
    if keywords:
        if not _keywords_in_fields(keywords, fields, symptoms, diagnosis_text, treatment_text):
            return False
    # No keywords (structural rule): primary condition is unconditionally met.

    # -- Secondary (AND) keyword condition ----------------------------------
    and_keywords = rule.get("and_keywords", [])
    if and_keywords:
        and_fields = rule.get("and_fields", [])
        if not _keywords_in_fields(and_keywords, and_fields, symptoms, diagnosis_text, treatment_text):
            return False

    # -- Auto-fire: trigger as soon as keyword conditions are satisfied ------
    if rule.get("auto_fire"):
        return True

    # -- Constraint evaluation ----------------------------------------------
    forbidden_severity = rule.get("severity_is", [])
    required_emergency = rule.get("emergency_must_be")
    requires_both = rule.get("constraint_requires_both", False)

    sev_violated = bool(forbidden_severity) and sev_upper in [s.upper() for s in forbidden_severity]
    emg_violated = required_emergency is not None and emg_upper != required_emergency.upper()

    if requires_both:
        return sev_violated and emg_violated
    else:
        return sev_violated or emg_violated



def evaluate(
    symptoms: str,
    severity: str,
    emergency_care: str,
    diagnosis_text: str = "",
    treatment_text: str = "",
    rules: list[dict] | None = None,
) -> tuple[bool, str, str, list[str]]:
    """Evaluate all active clinical safety rules. Returns (passed, action, reason, triggered_rule_ids).
    passed=True → proceed to LLM. passed=False → action is REJECT (hard stop) or REVIEW (inject concern)."""
    if rules is None:
        from rules.rule_repository import load_rules
        rules = load_rules()

    triggered_rejects: list[tuple[str, str]] = []  # (rule_id, reason)
    triggered_reviews: list[tuple[str, str]] = []

    for rule in rules:
        if not rule.get("active", True):
            continue
        try:
            if _rule_fires(rule, symptoms, severity, emergency_care, diagnosis_text, treatment_text):
                entry = (rule["rule_id"], rule["reason"])
                if rule.get("action") == "REJECT":
                    triggered_rejects.append(entry)
                else:
                    triggered_reviews.append(entry)
        except Exception as exc:
            logger.warning("[RULE_ENGINE] Rule %s evaluation error: %s", rule.get("rule_id"), exc)

    if triggered_rejects:
        rule_id, reason = triggered_rejects[0]
        all_ids = [r[0] for r in triggered_rejects + triggered_reviews]
        logger.info("[RULE_ENGINE] REJECT | rule: %s | %s", rule_id, reason)
        return False, "REJECT", reason, all_ids

    if triggered_reviews:
        rule_id, reason = triggered_reviews[0]
        all_ids = [r[0] for r in triggered_reviews]
        logger.info("[RULE_ENGINE] REVIEW | rule: %s | %s", rule_id, reason)
        return False, "REVIEW", reason, all_ids

    logger.debug("[RULE_ENGINE] All rules passed | severity=%s emergency=%s", severity, emergency_care)
    return True, "", "", []
