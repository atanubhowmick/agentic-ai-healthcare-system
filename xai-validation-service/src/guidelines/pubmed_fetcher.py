"""
PubMed E-utilities fetcher for clinical guidelines.

Fetches real guideline abstracts from PubMed (NCBI) at startup.
Rate limit: 3 requests/second (no API key required).

API used:
  esearch: search PubMed for relevant PMIDs
  efetch:  fetch abstracts for those PMIDs (XML)

The fetched abstracts are stored in ChromaDB (clinical_guidelines collection)
and retrieved via semantic search during constitutional guard critique.
"""

import time
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from log.logger import logger

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# 3 req/s without API key → ~0.35s between requests (with buffer)
_REQUEST_INTERVAL = 0.35
_REQUEST_TIMEOUT = 10  # seconds per HTTP call

# Curated PubMed search queries with specialty/source metadata
_GUIDELINE_QUERIES = [
    {
        "query": "NCCN guidelines cancer treatment oncology[MeSH] clinical practice",
        "source": "NCCN/PubMed", "specialty": "oncology", "topic": "cancer_treatment", "max_results": 3,
    },
    {
        "query": "lymphoma treatment guidelines chemotherapy clinical practice",
        "source": "PubMed", "specialty": "oncology", "topic": "lymphoma", "max_results": 2,
    },
    {
        "query": "colorectal cancer treatment guidelines surgery chemotherapy",
        "source": "PubMed", "specialty": "oncology", "topic": "colorectal", "max_results": 2,
    },
    {
        "query": "AHA ACC STEMI myocardial infarction treatment guidelines PCI",
        "source": "AHA/PubMed", "specialty": "cardiology", "topic": "stemi", "max_results": 3,
    },
    {
        "query": "heart failure treatment guidelines ACE inhibitor beta blocker",
        "source": "AHA/PubMed", "specialty": "cardiology", "topic": "heart_failure", "max_results": 2,
    },
    {
        "query": "atrial fibrillation anticoagulation treatment guidelines",
        "source": "AHA/PubMed", "specialty": "cardiology", "topic": "atrial_fibrillation", "max_results": 2,
    },
    {
        "query": "ischemic stroke thrombolysis tPA treatment guidelines",
        "source": "AHA/PubMed", "specialty": "neurology", "topic": "stroke", "max_results": 3,
    },
    {
        "query": "epilepsy seizure treatment antiepileptic drug guidelines",
        "source": "WHO/PubMed", "specialty": "neurology", "topic": "epilepsy", "max_results": 2,
    },
    {
        "query": "sepsis septic shock treatment guidelines surviving sepsis campaign",
        "source": "SSC/PubMed", "specialty": "emergency", "topic": "sepsis", "max_results": 3,
    },
    {
        "query": "COPD exacerbation treatment guidelines oxygen bronchodilator",
        "source": "GOLD/PubMed", "specialty": "respiratory", "topic": "copd", "max_results": 2,
    },
    {
        "query": "community acquired pneumonia treatment guidelines antibiotic",
        "source": "WHO/PubMed", "specialty": "respiratory", "topic": "pneumonia", "max_results": 2,
    },
    {
        "query": "diabetic ketoacidosis DKA treatment guidelines insulin",
        "source": "ADA/PubMed", "specialty": "endocrinology", "topic": "dka", "max_results": 2,
    },
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _search_pmids(query: str, max_results: int) -> list[str]:
    """Run esearch to get PMIDs for a query. Returns list of PMID strings."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        logger.info("[PUBMED] esearch → topic: '%s' | query: '%s'", query.split()[0], query[:80])
        resp = requests.get(_ESEARCH_URL, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        logger.info("[PUBMED] esearch ✓ | PMIDs found: %d | %s", len(pmids), pmids)
        return pmids
    except Exception as exc:
        logger.warning("[PUBMED] esearch ✗ | query: '%s' | error: %s", query[:60], exc)
        return []


def _fetch_abstracts(pmids: list[str]) -> list[dict]:
    """
    Run efetch to get abstracts for a list of PMIDs.
    Returns list of dicts: {pmid, title, abstract}
    """
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    try:
        logger.info("[PUBMED] efetch → fetching abstracts for PMIDs: %s", pmids)
        resp = requests.get(_EFETCH_URL, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        articles = _parse_efetch_xml(resp.text)
        logger.info("[PUBMED] efetch ✓ | abstracts parsed: %d", len(articles))
        return articles
    except Exception as exc:
        logger.warning("[PUBMED] efetch ✗ | PMIDs: %s | error: %s", pmids, exc)
        return []


def _parse_efetch_xml(xml_text: str) -> list[dict]:
    """Parse PubMed efetch XML and extract title + abstract per article."""
    articles = []
    try:
        root = ET.fromstring(xml_text)
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abstract_els = article.findall(".//AbstractText")

            pmid = pmid_el.text.strip() if pmid_el is not None else ""
            title = title_el.text.strip() if title_el is not None and title_el.text else ""

            # AbstractText may have multiple sections (background, methods, etc.)
            abstract_parts = []
            for el in abstract_els:
                label = el.get("Label", "")
                text = el.text.strip() if el.text else ""
                if text:
                    abstract_parts.append(f"{label}: {text}" if label else text)
            abstract = " ".join(abstract_parts)

            if abstract:
                articles.append({"pmid": pmid, "title": title, "abstract": abstract})
    except ET.ParseError as exc:
        logger.warning("[PUBMED] XML parse error: %s", exc)
    return articles


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_guidelines() -> list[dict]:
    """
    Fetch clinical guideline abstracts from PubMed for all configured queries.

    Returns list of dicts:
        {text, source, specialty, topic, pmid}

    Respects 3 req/s rate limit (no API key needed).
    Gracefully skips failed queries — caller receives partial results.
    """
    results: list[dict] = []
    total_requests = 0

    for q in _GUIDELINE_QUERIES:
        # esearch call
        if total_requests > 0:
            time.sleep(_REQUEST_INTERVAL)
        pmids = _search_pmids(q["query"], q["max_results"])
        total_requests += 1

        if not pmids:
            continue

        # efetch call
        time.sleep(_REQUEST_INTERVAL)
        articles = _fetch_abstracts(pmids)
        total_requests += 1

        for art in articles:
            text = f"{art['title']}. {art['abstract']}" if art["title"] else art["abstract"]
            results.append({
                "text": text[:1500],  # cap to keep ChromaDB metadata compact
                "source": q["source"],
                "specialty": q["specialty"],
                "topic": q["topic"],
                "pmid": art["pmid"],
            })

        logger.debug(
            "[PUBMED] Query '%s' → %d abstract(s) fetched",
            q["topic"], len(articles),
        )

    logger.info("[PUBMED] Fetched %d guideline abstract(s) from PubMed (%d API calls)", len(results), total_requests)
    return results
