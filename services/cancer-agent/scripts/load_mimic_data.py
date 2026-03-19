"""
MIMIC-IV v3.1 Cancer Cases - Data Loader
==========================================
Ingests oncology cases from MIMIC-IV v3.1 into the ChromaDB vector store used by
the Cancer Agent for RAG (Retrieval-Augmented Generation).

Source: Google BigQuery (requires GCP credentials)

MIMIC-IV v3.1 BigQuery schemas (physionet-data project):
  physionet-data.mimiciv_3_1_hosp   - hospital module (diagnoses, admissions)
  physionet-data.mimiciv_note       - clinical notes (discharge summaries)
  physionet-data.mimiciv_ed         - ED module (triage chief complaints)

NOTE: --project is YOUR OWN GCP billing project ID (not physionet-data).
      The data lives in physionet-data; your project is used for query billing.

ICD-10 codes extracted: C00-C97 (malignant neoplasms) + D00-D49 (in-situ / benign)

Document text priority (for semantic embedding):
  1. ED triage chief complaint  - patient's own words (best match for user queries)
  2. Discharge note Chief Complaint + HPI - clinical free text (good match)
  3. ICD long title + lay synonym  - structured text (fallback)

Usage
-----
# From BigQuery (replace MY-GCP-PROJECT with your own GCP project ID):
python load_mimic_data.py --project MY-GCP-PROJECT --limit 50000

# Dry-run (print first 3 records, no write):
python load_mimic_data.py --project MY-GCP-PROJECT --dry-run
"""

import argparse
import re
import sys
import os

# allow running from repo root or scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from log.logger import logger


# -- Clinical note section extractors -----------------------------------------

def _clean_text(text: str) -> str:
    """
    Normalise raw MIMIC discharge note text:
      - Replace anonymisation placeholders (___) with a generic token
      - Collapse newlines and runs of whitespace into a single space
    """
    if not text:
        return ""
    text = re.sub(r"_{2,}", " ", text)   # ___ -> space
    text = re.sub(r"\s+", " ", text)     # \n, \t, multiple spaces -> single space
    return text.strip()


def _extract_section(note: str, headers: list[str], max_chars: int = 600) -> str:
    """Extract the first matching section from a clinical discharge note."""
    if not note:
        return ""
    note = _clean_text(note)
    for header in headers:
        pattern = re.compile(
            rf"{re.escape(header)}[:\s]*(.*?)(?=[A-Z][A-Za-z /]+:|$)",
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(note)
        if m:
            return m.group(1).strip()[:max_chars]
    return ""


def _extract_chief_complaint(note: str) -> str:
    return _extract_section(note, ["Chief Complaint", "CHIEF COMPLAINT"], max_chars=2000)


def _extract_hpi(note: str) -> str:
    return _extract_section(
        note,
        ["History of Present Illness", "HISTORY OF PRESENT ILLNESS", "HPI"],
        max_chars=2000,
    )


def _extract_assessment(note: str) -> str:
    return _extract_section(
        note,
        ["Assessment and Plan", "ASSESSMENT AND PLAN", "Assessment", "ASSESSMENT", "Plan", "PLAN"],
        max_chars=2000,
    )


def _infer_severity(discharge_location) -> str:
    """Map MIMIC discharge disposition to LOW / HIGH / CRITICAL."""
    if not discharge_location or not isinstance(discharge_location, str):
        return "UNKNOWN"
    loc = discharge_location.upper()
    if any(k in loc for k in ("DIED", "EXPIRED", "HOSPICE")):
        return "CRITICAL"
    if any(k in loc for k in ("REHAB", "SKILLED", "ACUTE", "FACILITY", "TRANSFER")):
        return "HIGH"
    return "LOW"


def _build_document_text(triage_complaint: str, chief_complaint: str, hpi: str) -> str:
    """
    Build the embedding document - text used for similarity search against user queries.

    Only symptom/presentation text is embedded. The cancer diagnosis is stored
    in metadata only so it does not dilute symptom-based similarity scores.

    Priority:
      1. triage_complaint : patient's own words from ED visit (best semantic match)
      2. chief_complaint + hpi : clinical free text from discharge note
    """
    parts = []

    if triage_complaint and triage_complaint.strip():
        parts.append(triage_complaint.strip())
    if chief_complaint:
        parts.append(chief_complaint)
    if hpi:
        parts.append(hpi)

    return " ".join(parts) if parts else ""


# -- BigQuery source -----------------------------------------------------------

BIGQUERY_SQL = """
SELECT
    d.subject_id,
    d.hadm_id,
    STRING_AGG(DISTINCT diag.long_title ORDER BY diag.long_title LIMIT 5) AS cancer_diagnoses,
    STRING_AGG(DISTINCT d.icd_code    ORDER BY d.icd_code    LIMIT 10) AS icd_codes,
    ANY_VALUE(adm.discharge_location)  AS discharge_location,
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
WHERE
    d.icd_version = 10
    AND REGEXP_CONTAINS(d.icd_code, r'^C|^D[0-4][0-9]')
    AND (t.chiefcomplaint IS NOT NULL OR n.text IS NOT NULL)
GROUP BY d.subject_id, d.hadm_id
LIMIT {limit}
"""


def _load_from_bigquery(project_id: str, limit: int):
    """Return a list of raw row dicts from BigQuery."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    query = BIGQUERY_SQL.format(limit=limit)
    logger.info("[LOADER] Submitting BigQuery query (limit=%d)...", limit)
    job = client.query(query)
    logger.info("[LOADER] Query submitted | job_id: %s | waiting for results...", job.job_id)

    rows = []
    log_interval = 200
    for row in job.result():
        rows.append(dict(row))
        n = len(rows)
        if n % log_interval == 0:
            logger.info("[LOADER] Fetched %d rows so far...", n)

    logger.info("[LOADER] BigQuery fetch complete | total rows: %d", len(rows))
    return rows


# -- Record processor ----------------------------------------------------------

def _process_row(row: dict) -> dict | None:
    """Transform a raw MIMIC row into a ChromaDB-ready record."""
    note             = row.get("discharge_notes") or ""
    triage_complaint = row.get("triage_complaint") or ""
    cancer_type      = row.get("cancer_diagnoses", "Unknown neoplasm")
    icd_codes        = row.get("icd_codes", "")
    discharge_loc    = row.get("discharge_location", "")

    chief_complaint   = _extract_chief_complaint(note)
    hpi               = _extract_hpi(note)
    treatment_summary = _extract_assessment(note)
    document_text     = _build_document_text(triage_complaint, chief_complaint, hpi)

    # Last resort fallback: no symptom text available - use ICD diagnosis title
    if not document_text.strip():
        if cancer_type and cancer_type != "Unknown neoplasm":
            document_text = cancer_type
            if icd_codes:
                document_text += f" {icd_codes}"
        else:
            return None   # nothing useful to embed

    return {
        "document":          document_text,
        "cancer_type":       str(cancer_type)[:500],
        "icd_codes":         str(icd_codes)[:200],
        "chief_complaint":   triage_complaint or chief_complaint,
        "treatment_summary": treatment_summary,
        "lab_findings":      "",
        "severity":          _infer_severity(discharge_loc),
        "source":            "MIMIC-IV",
        "subject_id":        str(row.get("subject_id", "")),
        "hadm_id":           str(row.get("hadm_id", "")),
    }


# -- ChromaDB writer -----------------------------------------------------------

def _write_to_chroma(records: list[dict], batch_size: int = 200) -> int:
    """Embed and store processed records in ChromaDB. Returns count written."""
    import chromadb
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from core.config import CHROMA_HOST, CHROMA_PORT, MIMIC_COLLECTION_NAME

    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma(
        client=chroma_client,
        collection_name=MIMIC_COLLECTION_NAME,
        embedding_function=embeddings,
    )

    texts, metadatas, ids = [], [], []
    written = 0

    for rec in records:
        doc_id = f"mimic_{rec['subject_id']}_{rec['hadm_id']}"
        texts.append(rec["document"])
        metadatas.append({
            "cancer_type":       rec["cancer_type"],
            "icd_codes":         rec["icd_codes"],
            "chief_complaint":   rec["chief_complaint"],
            "treatment_summary": rec["treatment_summary"],
            "lab_findings":      rec["lab_findings"],
            "severity":          rec["severity"],
            "source":            rec["source"],
        })
        ids.append(doc_id)

        if len(texts) >= batch_size:
            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            written += len(texts)
            logger.info("[LOADER] Written %d / %d records...", written, len(records))
            texts, metadatas, ids = [], [], []

    if texts:
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        written += len(texts)

    logger.info("[LOADER] Done. Total records written to ChromaDB: %d", written)
    return written


# -- CLI entry point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Load MIMIC-IV cancer cases into ChromaDB")
    parser.add_argument("--project", required=True,
                        help="GCP billing project ID (your own, not physionet-data)")
    parser.add_argument("--limit", type=int, default=50000,
                        help="Maximum number of cases to load (default: 50000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first 3 processed records without writing to ChromaDB")
    args = parser.parse_args()

    # Dry-run caps the fetch at 20 rows so it completes quickly
    limit = 10 if args.dry_run else args.limit

    # Load raw rows
    raw_rows = _load_from_bigquery(args.project, limit)

    # Process rows
    records = []
    skipped = 0
    for row in raw_rows:
        processed = _process_row(row)
        if processed:
            records.append(processed)
        else:
            skipped += 1

    logger.info("[LOADER] Processed %d records | skipped %d (no usable text)", len(records), skipped)

    if args.dry_run:
        logger.info("[DRY RUN] === first %d records ===", len(records[:10]))
        for i, rec in enumerate(records[:10], start=1):
            logger.info(
                "[DRY RUN] Record %d | Cancer: %s | ICD: %s | Severity: %s",
                i, rec['cancer_type'], rec['icd_codes'], rec['severity'],
            )
            logger.info("[DRY RUN]   Document  : %s", rec['document'])
            logger.info("[DRY RUN]   Treatment : %s", rec['treatment_summary'] or "(none)")
        logger.info("[DRY RUN] Total records ready to write: %d", len(records))
        return

    # Remove these logs
    for i, rec in enumerate(records[:10], start=1):
        logger.info(
            "[Data Load] Record %d | Cancer: %s | ICD: %s | Severity: %s",
            i, rec['cancer_type'], rec['icd_codes'], rec['severity'],
        )
        logger.info("[Data Load] Document  : %s", rec['document'])
        logger.info("[Data Load] Treatment : %s", rec['treatment_summary'] or "(none)")
    logger.info("[Data Load] Total records ready to write: %d", len(records))
        
    # Write to ChromaDB
    if not records:
        logger.warning("[LOADER] No records to write. Exiting.")
        return

    written = _write_to_chroma(records)
    logger.info("[LOADER] Successfully loaded %d MIMIC-IV cancer cases into ChromaDB.", written)


if __name__ == "__main__":
    main()
