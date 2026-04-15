# Extracts MIMIC-IV cancer cases from BigQuery and saves them to MongoDB (mimic_iv_records).
# The collection is consumed by two pipelines in the evaluation-service:
#   - Cancer Agent evaluator     : trains HistGBM / LogisticRegression / LinearSVC classifiers
#   - XAI evaluator              : sends cases through the XAI validation service for metrics
#
# SQL includes admission_type and has_icu_stay (via ICU join), which load_mimic_data.py omits.
# These extra columns enable more reliable ground truth severity labels for both pipelines.
#
# Note: load_mimic_data.py excludes subject_ids where MOD(subject_id, 5) = 0.
# This script has no split filter, so those subject_ids appear only here.
# Re-enable the MOD filter in load_mimic_data.py for a clean train/eval split.
#
# Usage:
#   python load_mimic_mongo.py --project MY-GCP-PROJECT --limit 2000
#   python load_mimic_mongo.py --project MY-GCP-PROJECT --dry-run

import argparse
import os
import sys

import pymongo
from google.cloud import bigquery
from pymongo import MongoClient

# Allow running from scripts/ or repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from core.config import MONGO_URI, MONGO_DB, MONGO_MIMIC_COLLECTION  # cancer-agent config
from log.logger import logger
from load_mimic_data import _process_row  # re-use existing row processor

_mongo_client: MongoClient | None = None


def _get_mongo_collection():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
        logger.info("[MONGO] Connected to %s / %s", MONGO_URI, MONGO_DB)
    return _mongo_client[MONGO_DB][MONGO_MIMIC_COLLECTION]


def save_evaluation_cases(records: list[dict]) -> int:
    """Bulk-upsert evaluation records on (subject_id, hadm_id). Returns upserted count."""
    if not records:
        logger.warning("[MONGO] save_evaluation_cases called with empty list — nothing saved.")
        return 0

    col = _get_mongo_collection()
    col.create_index(
        [("subject_id", pymongo.ASCENDING), ("hadm_id", pymongo.ASCENDING)],
        unique=True,
        background=True,
    )
    ops = [
        pymongo.UpdateOne(
            {"subject_id": rec.get("subject_id"), "hadm_id": rec.get("hadm_id")},
            {"$set": rec},
            upsert=True,
        )
        for rec in records
    ]
    result = col.bulk_write(ops, ordered=False)
    logger.info(
        "[MONGO] Upserted %d | modified %d | total processed %d",
        result.upserted_count, result.modified_count, len(records),
    )
    return result.upserted_count + result.modified_count


def _process_mimic_row(row: dict) -> dict | None:
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
    # than discharge_location for ground truth labels).
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


_MIMIC_SQL = """
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


def _load_mimic_rows(project_id: str, limit: int) -> list[dict]:
    """Fetch MIMIC-IV rows from BigQuery."""
    client = bigquery.Client(project=project_id)
    query = _MIMIC_SQL.format(limit=limit)
    logger.info("[MIMIC] Submitting BigQuery query (limit=%d)...", limit)
    job = client.query(query)
    logger.info("[MIMIC] job_id: %s | waiting for results...", job.job_id)

    rows: list[dict] = []
    for row in job.result():
        rows.append(dict(row))
        if len(rows) % 200 == 0:
            logger.info("[MIMIC] Fetched %d rows so far...", len(rows))

    logger.info("[MIMIC] BigQuery fetch complete | total rows: %d", len(rows))
    return rows


def load_mimic_cases(
    project_id: str,
    limit: int,
    dry_run: bool = False,
) -> list[dict]:
    """
    Load MIMIC-IV cancer cases from BigQuery and persist to MongoDB.

    Args:
        project_id:  GCP billing project (not physionet-data).
        limit:       Maximum number of cases to fetch.
        dry_run:     If True, fetch only 20 rows and print without saving.

    Returns:
        List of processed record dicts.
    """
    effective_limit = 20 if dry_run else limit
    raw_rows = _load_mimic_rows(project_id, effective_limit)

    records: list[dict] = []
    skipped = 0
    for row in raw_rows:
        rec = _process_mimic_row(row)
        if rec:
            records.append(rec)
        else:
            skipped += 1

    logger.info(
        "[MIMIC] Processed %d records | skipped %d (no usable symptom text)",
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
        logger.warning("[MIMIC] No records to save. Exiting.")
        return records

    saved = save_evaluation_cases(records)
    logger.info("[MIMIC] Saved %d records → MongoDB (%s)", saved, "mimic_iv_records")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load MIMIC-IV cancer cases into MongoDB"
    )
    parser.add_argument(
        "--project", required=True,
        help="GCP billing project ID (your own, not physionet-data)",
    )
    parser.add_argument(
        "--limit", type=int, default=2000,
        help="Maximum cases to load (default: 2000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch 20 rows, print them, do not write to MongoDB",
    )
    args = parser.parse_args()

    load_mimic_cases(
        project_id=args.project,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
