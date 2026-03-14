"""
MIMIC-IV v3.1 Cancer Cases - Data Loader
==========================================
Ingests oncology cases from MIMIC-IV v3.1 into the ChromaDB vector store used by
the Cancer Agent for RAG (Retrieval-Augmented Generation).

Supports two source modes:
  --source bigquery   : query via Google BigQuery (requires GCP credentials)
  --source local      : read from locally extracted MIMIC-IV CSV/gz files

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
python load_mimic_data.py --source bigquery --project MY-GCP-PROJECT --limit 50000

# From local CSV files (path to the extracted MIMIC-IV zip):
python load_mimic_data.py --source local --mimic-dir /path/to/mimic-iv --limit 5000

# Dry-run (print first 3 records, no write):
python load_mimic_data.py --source bigquery --project MY-GCP-PROJECT --dry-run
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
    return _extract_section(note, ["Chief Complaint", "CHIEF COMPLAINT"], max_chars=300)


def _extract_hpi(note: str) -> str:
    return _extract_section(
        note,
        ["History of Present Illness", "HISTORY OF PRESENT ILLNESS", "HPI"],
        max_chars=600,
    )


def _extract_assessment(note: str) -> str:
    return _extract_section(
        note,
        ["Assessment and Plan", "ASSESSMENT AND PLAN", "Assessment", "ASSESSMENT", "Plan", "PLAN"],
        max_chars=600,
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
    Build the embedding document — text used for similarity search against user queries.

    Priority:
      1. triage_complaint : patient's own words from ED visit (best semantic match)
      2. chief_complaint + hpi : clinical free text from discharge note
    """
    # ED triage complaint is closest to natural patient language
    if triage_complaint and triage_complaint.strip():
        parts = [f"Patient complaint: {triage_complaint.strip()}"]
        if chief_complaint:
            parts.append(f"Chief Complaint: {chief_complaint}")
        if hpi:
            parts.append(f"Presentation: {hpi}")
        return " | ".join(parts)

    # Fall back to discharge note sections
    parts = []
    if chief_complaint:
        parts.append(f"Chief Complaint: {chief_complaint}")
    if hpi:
        parts.append(f"Presentation: {hpi}")
    return " | ".join(parts) if parts else ""


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
    logger.info("[LOADER] Running BigQuery query (limit=%d)...", limit)
    rows = list(client.query(query).result())
    logger.info("[LOADER] BigQuery returned %d rows", len(rows))
    return [dict(r) for r in rows]


# -- Local CSV source ----------------------------------------------------------

def _load_from_local(mimic_dir: str, limit: int):
    """
    Read and join MIMIC-IV local CSV/gz files.

    Expected file layout (matches MIMIC-IV official release):
      <mimic_dir>/hosp/diagnoses_icd.csv.gz
      <mimic_dir>/hosp/d_icd_diagnoses.csv.gz
      <mimic_dir>/hosp/admissions.csv.gz
      <mimic_dir>/note/discharge.csv.gz    - preferred for clinical notes
      <mimic_dir>/hosp/discharge.csv.gz   - fallback
      <mimic_dir>/ed/triage.csv.gz        - ED triage chief complaints (preferred)

    If no discharge notes or triage file is found the loader proceeds without
    natural language text; records will fall back to ICD titles.
    """
    import pandas as pd
    from pathlib import Path

    base = Path(mimic_dir)

    def _read(rel_path, **kwargs):
        full = base / Path(rel_path)
        if not full.exists():
            full = full.with_suffix("")   # try uncompressed
        logger.info("[LOADER] Reading %s", full)
        return pd.read_csv(full, **kwargs)

    def _read_optional(candidates: list, cols: list):
        """Try each path in order, return DataFrame or None."""
        for rel in candidates:
            for path in (base / rel, (base / rel).with_suffix("")):
                if path.exists():
                    logger.info("[LOADER] Reading %s", path)
                    return pd.read_csv(path, dtype=str, usecols=cols)
        return None

    diag  = _read("hosp/diagnoses_icd.csv.gz", dtype=str)
    d_icd = _read("hosp/d_icd_diagnoses.csv.gz", dtype=str)
    adm   = _read("hosp/admissions.csv.gz", dtype=str,
                  usecols=["subject_id", "hadm_id", "discharge_location"])

    notes = _read_optional(
        ["note/discharge.csv.gz", "hosp/discharge.csv.gz"],
        ["subject_id", "hadm_id", "text"],
    )
    if notes is None:
        logger.warning("[LOADER] No discharge notes file found - proceeding without notes.")

    triage = _read_optional(["ed/triage.csv.gz"], ["subject_id", "stay_id", "chiefcomplaint"])
    edstays = _read_optional(["ed/edstays.csv.gz"], ["subject_id", "hadm_id", "stay_id"])
    if triage is None:
        logger.warning("[LOADER] No ED triage file found - proceeding without triage complaints.")

    # Filter ICD-10 cancer codes
    cancer_mask = diag["icd_version"].eq("10") & (
        diag["icd_code"].str.match(r"^C") |
        diag["icd_code"].str.match(r"^D[0-4][0-9]")
    )
    diag = diag[cancer_mask]

    # Join ICD descriptions
    diag = diag.merge(d_icd[["icd_code", "icd_version", "long_title"]],
                      on=["icd_code", "icd_version"], how="left")

    # Aggregate to admission level
    diag_agg = (
        diag.groupby(["subject_id", "hadm_id"])
        .agg(
            cancer_diagnoses=("long_title", lambda x: "; ".join(x.dropna().unique()[:5])),
            icd_codes=("icd_code", lambda x: ", ".join(x.dropna().unique()[:10])),
        )
        .reset_index()
    )

    merged = diag_agg.merge(adm, on=["subject_id", "hadm_id"], how="left")

    # Join discharge notes
    if notes is not None:
        merged = merged.merge(
            notes.rename(columns={"text": "discharge_notes"}),
            on=["subject_id", "hadm_id"], how="left",
        )
    else:
        merged["discharge_notes"] = ""

    # Join ED triage chief complaint via edstays
    if triage is not None and edstays is not None:
        stay_complaint = edstays.merge(
            triage[["subject_id", "stay_id", "chiefcomplaint"]],
            on=["subject_id", "stay_id"], how="left",
        )[["subject_id", "hadm_id", "chiefcomplaint"]]
        merged = merged.merge(stay_complaint, on=["subject_id", "hadm_id"], how="left")
    else:
        merged["chiefcomplaint"] = None

    # Keep only rows that have at least one text source
    has_text = (
        merged["chiefcomplaint"].notna() |
        merged["discharge_notes"].notna()
    )
    merged = merged[has_text].head(limit)

    logger.info("[LOADER] Local CSV returned %d rows after join", len(merged))
    return merged.rename(columns={"chiefcomplaint": "triage_complaint"}).to_dict(orient="records")


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

    # Last resort fallback: use ICD diagnosis text
    if not document_text.strip():
        if cancer_type and cancer_type != "Unknown neoplasm":
            document_text = f"Oncology patient. Diagnosis: {cancer_type}."
            if icd_codes:
                document_text += f" ICD-10 codes: {icd_codes}."
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
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from core.config import CHROMA_PERSIST_DIR, MIMIC_COLLECTION_NAME

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma(
        collection_name=MIMIC_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
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
    parser.add_argument("--source", choices=["bigquery", "local"], required=True,
                        help="Data source: bigquery or local CSV files")
    parser.add_argument("--project", default=None,
                        help="GCP project ID (required for --source bigquery)")
    parser.add_argument("--mimic-dir", default=None,
                        help="Path to extracted MIMIC-IV directory (required for --source local)")
    parser.add_argument("--limit", type=int, default=50000,
                        help="Maximum number of cases to load (default: 50000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first 3 processed records without writing to ChromaDB")
    args = parser.parse_args()

    # Load raw rows
    if args.source == "bigquery":
        if not args.project:
            parser.error("--project is required when --source=bigquery")
        raw_rows = _load_from_bigquery(args.project, args.limit)
    else:
        if not args.mimic_dir:
            parser.error("--mimic-dir is required when --source=local")
        raw_rows = _load_from_local(args.mimic_dir, args.limit)

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
        print("\n=== DRY RUN - first 3 records ===")
        for rec in records[:3]:
            print(f"\nDocument  : {rec['document'][:300]}")
            print(f"Cancer    : {rec['cancer_type']}")
            print(f"ICD codes : {rec['icd_codes']}")
            print(f"Severity  : {rec['severity']}")
            print(f"Treatment : {rec['treatment_summary'][:200]}")
        print(f"\nTotal records ready to write: {len(records)}")
        return

    # Write to ChromaDB
    if not records:
        logger.warning("[LOADER] No records to write. Exiting.")
        return

    written = _write_to_chroma(records)
    print(f"\nSuccessfully loaded {written} MIMIC-IV cancer cases into ChromaDB.")


if __name__ == "__main__":
    main()
