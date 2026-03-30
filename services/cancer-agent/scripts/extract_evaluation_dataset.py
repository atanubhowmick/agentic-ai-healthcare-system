"""
MIMIC-IV Cancer Cases - Evaluation Dataset Extractor
======================================================
Extracts an evaluation test set from MIMIC-IV v3.1 for evaluating the Cancer Agent.

Loads all available cancer cases (no subject_id split filter).

The training loader (load_mimic_data.py) does not apply a split filter, so there
is potential overlap for subject_ids divisible by 5.  For a clean split, re-run
load_mimic_data.py with the additional WHERE clause:
    AND MOD(CAST(d.subject_id AS INT64), 5) != 0

Output: MongoDB collection  (agentic_ai_healthcare_db.mimic_evaluation_cases)
Each record mirrors the ChromaDB metadata structure so it can be consumed
directly by the evaluation pipeline.

Usage
-----
# Extract up to 2000 evaluation cases:
python extract_evaluation_dataset.py --project MY-GCP-PROJECT --limit 2000

# Dry-run (fetch 20 rows, print, do not save):
python extract_evaluation_dataset.py --project MY-GCP-PROJECT --dry-run
"""

import argparse
import os
import sys

from google.cloud import bigquery

# Allow running from scripts/ or repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

# Allow importing evaluation-service mongo_client
_eval_service_src = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../evaluation-service/src")
)
sys.path.insert(0, _eval_service_src)

from core.mongo_client import save_evaluation_cases  # type: ignore[import]  # runtime sys.path
from log.logger import logger
from load_mimic_data import _process_row  # re-use existing row processor


def _process_evaluation_row(row: dict) -> dict | None:
    """
    Extend _process_row with admission_type and ICU-based severity.

    Severity derivation (priority order):
      1. ICU admission (has_icu_stay=1)       → CRITICAL
      2. EMERGENCY or URGENT admission_type   → HIGH
      3. ELECTIVE admission_type              → LOW
      4. Fall back to discharge_location      → from _process_row()
    """
    rec = _process_row(row)
    if rec is None:
        return None

    admission_type = (row.get("admission_type") or "").upper().strip()
    has_icu_stay   = bool(row.get("has_icu_stay", 0))

    # Override severity with ICU/admission-based derivation (more reliable
    # than discharge_location for ground truth evaluation).
    # Covers all 9 MIMIC-IV v3.1 admission_type values.
    if has_icu_stay:
        rec["severity"] = "CRITICAL"
    elif admission_type in ("EMERGENCY", "URGENT", "DIRECT EMER."):
        rec["severity"] = "HIGH"
    elif admission_type in (
        "ELECTIVE",
        "OBSERVATION ADMIT",
        "EU OBSERVATION",
        "AMBULATORY OBSERVATION",
        "DIRECT OBSERVATION",
        "SURGICAL SAME DAY ADMISSION",
    ):
        rec["severity"] = "LOW"
    # else: keep discharge_location-based severity already set by _process_row()

    rec["admission_type"] = admission_type
    rec["has_icu_stay"]   = has_icu_stay
    return rec


# ---------------------------------------------------------------------------
# BigQuery SQL — identical JOIN structure to load_mimic_data.py but with
# the MOD filter to isolate the evaluation split.
# ---------------------------------------------------------------------------

_EVALUATION_SQL = """
SELECT
    d.subject_id,
    d.hadm_id,
    STRING_AGG(DISTINCT diag.long_title ORDER BY diag.long_title LIMIT 5) AS cancer_diagnoses,
    STRING_AGG(DISTINCT d.icd_code    ORDER BY d.icd_code    LIMIT 10) AS icd_codes,
    ANY_VALUE(adm.discharge_location)  AS discharge_location,
    ANY_VALUE(adm.admission_type)      AS admission_type,
    MAX(CASE WHEN icu.hadm_id IS NOT NULL THEN 1 ELSE 0 END) AS has_icu_stay,
    ANY_VALUE(t.chiefcomplaint)        AS triage_complaint,
    ANY_VALUE(n.text)                  AS discharge_notes
FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`        AS d
JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses`      AS diag
    ON d.icd_code = diag.icd_code AND d.icd_version = diag.icd_version
LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions`      AS adm
    ON d.subject_id = adm.subject_id AND d.hadm_id = adm.hadm_id
LEFT JOIN `physionet-data.mimiciv_note.discharge`            AS n
    ON d.subject_id = n.subject_id  AND d.hadm_id = n.hadm_id
LEFT JOIN `physionet-data.mimiciv_ed.edstays`               AS es
    ON d.subject_id = es.subject_id AND d.hadm_id = es.hadm_id
LEFT JOIN `physionet-data.mimiciv_ed.triage`                AS t
    ON es.subject_id = t.subject_id AND es.stay_id = t.stay_id
LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays`         AS icu
    ON d.subject_id = icu.subject_id AND d.hadm_id = icu.hadm_id
WHERE
    d.icd_version = 10
    AND REGEXP_CONTAINS(d.icd_code, r'^C|^D[0-4][0-9]')
    AND (t.chiefcomplaint IS NOT NULL OR n.text IS NOT NULL)
GROUP BY d.subject_id, d.hadm_id
LIMIT {limit}
"""


# ---------------------------------------------------------------------------
# BigQuery loader (own copy so load_mimic_data.py is not modified)
# ---------------------------------------------------------------------------

def _load_evaluation_rows(project_id: str, limit: int) -> list[dict]:
    """Fetch evaluation rows from MIMIC-IV BigQuery."""
    client = bigquery.Client(project=project_id)
    query = _EVALUATION_SQL.format(limit=limit)
    logger.info("[EVAL] Submitting BigQuery evaluation query (limit=%d)...", limit)
    job = client.query(query)
    logger.info("[EVAL] job_id: %s | waiting for results...", job.job_id)

    rows: list[dict] = []
    for row in job.result():
        rows.append(dict(row))
        if len(rows) % 200 == 0:
            logger.info("[EVAL] Fetched %d rows so far...", len(rows))

    logger.info("[EVAL] BigQuery fetch complete | total rows: %d", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_evaluation_set(
    project_id: str,
    limit: int,
    dry_run: bool = False,
) -> list[dict]:
    """
    Extract evaluation records from MIMIC-IV and persist to MongoDB.

    Args:
        project_id:  GCP billing project (not physionet-data).
        limit:       Maximum number of evaluation cases to fetch.
        dry_run:     If True, fetch only 20 rows and print without saving.

    Returns:
        List of processed record dicts.
    """
    effective_limit = 20 if dry_run else limit
    raw_rows = _load_evaluation_rows(project_id, effective_limit)

    records: list[dict] = []
    skipped = 0
    for row in raw_rows:
        rec = _process_evaluation_row(row)
        if rec:
            records.append(rec)
        else:
            skipped += 1

    logger.info(
        "[EVAL] Processed %d records | skipped %d (no usable symptom text)",
        len(records), skipped,
    )

    if dry_run:
        logger.info("[DRY RUN] === first %d records ===", min(5, len(records)))
        for i, rec in enumerate(records[:5], 1):
            logger.info(
                "[DRY RUN] %d | Cancer: %s | ICD: %s | Severity: %s",
                i, rec["cancer_type"], rec["icd_codes"], rec["severity"],
            )
            logger.info("[DRY RUN]   Document   : %.120s", rec["document"])
            logger.info("[DRY RUN]   Treatment  : %.80s", rec["treatment_summary"] or "(none)")
        logger.info("[DRY RUN] %d records ready (not saved — dry-run mode).", len(records))
        return records

    if not records:
        logger.warning("[EVAL] No records to save. Exiting.")
        return records

    saved = save_evaluation_cases(records)
    logger.info("[EVAL] Saved %d evaluation records → MongoDB (%s)", saved, "mimic_evaluation_cases")
    return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MIMIC-IV evaluation test set for Cancer Agent evaluation"
    )
    parser.add_argument(
        "--project", required=True,
        help="GCP billing project ID (your own, not physionet-data)",
    )
    parser.add_argument(
        "--limit", type=int, default=2000,
        help="Maximum evaluation cases to extract (default: 2000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch 20 rows, print them, do not write to MongoDB",
    )
    args = parser.parse_args()

    extract_evaluation_set(
        project_id=args.project,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
