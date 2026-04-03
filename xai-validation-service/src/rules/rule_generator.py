"""
LLM-based clinical safety rule generator.

Reads guideline abstracts from ChromaDB and uses an LLM to extract
structured clinical safety rules in the standard rule schema format.
Generated rules are stored in MongoDB with source prefixed by the
guideline origin (e.g. "AHA/PubMed").

Called once at startup (after ChromaDB guideline seeding) via seed_llm_rules().
Skips generation if LLM-generated rules already exist in MongoDB.

Step-by-step flow:
  1. Check MongoDB — skip if LLM_G* rules already exist.
  2. Search ChromaDB with broad clinical queries to retrieve guideline abstracts.
  3. For each abstract: call LLM with a structured extraction prompt.
  4. Validate extracted rules against the rule schema.
  5. Upsert valid rules to MongoDB and refresh the in-memory cache.
"""

from __future__ import annotations

import json

import pymongo
from pymongo import MongoClient

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from core.config import MONGODB_URI, MONGODB_DB_NAME, OPENAI_MODEL
from guidelines.guideline_client import search_guidelines
from rules.rule_repository import load_rules
from log.logger import logger

# Guideline search queries used to retrieve relevant abstracts from ChromaDB
_SEED_QUERIES = [
    "emergency triage rules cardiac chest pain severity",
    "sepsis treatment guidelines emergency care",
    "stroke tPA thrombolysis treatment protocol",
    "anticoagulation bleeding contraindications",
    "cancer oncology treatment safety guidelines",
    "respiratory failure oxygen therapy emergency",
    "renal failure drug dosing contraindication",
    "septic shock vasopressor treatment guidelines",
    "pulmonary embolism anticoagulation emergency",
    "meningitis antibiotic treatment urgent",
]

_EXTRACT_SYSTEM = """You are a clinical safety rule extractor.
Extract deterministic clinical safety rules from a medical guideline abstract.

Each rule must follow this exact JSON schema:
{
  "rule_id": "LLM_G001",
  "category": "emergency",
  "description": "brief description",
  "keywords": ["clinical keyword 1", "clinical keyword 2"],
  "fields": ["symptoms", "diagnosis"],
  "and_keywords": [],
  "and_fields": [],
  "severity_is": ["LOW"],
  "emergency_must_be": null,
  "constraint_requires_both": false,
  "auto_fire": false,
  "action": "REJECT",
  "reason": "brief clinical justification",
  "source": "AHA/PubMed",
  "speciality": "cardiology",
  "active": true
}

Field rules:
- category: emergency | severity_consistency | diagnosis_alignment | treatment_safety | medication_safety
- fields / and_fields: choose from ["symptoms", "diagnosis", "treatment"]
- severity_is: list of severity values that trigger the rule (e.g. ["LOW"]) — empty if not severity-based
- emergency_must_be: "YES" or null
- constraint_requires_both: true only when BOTH severity_is AND emergency_must_be must be violated
- auto_fire: true for treatment/medication interaction rules (fire on keyword match alone)
- action: "REJECT" for absolute contraindications; "REVIEW" for caution flags

Output a JSON array of up to 3 rules. No markdown, no explanation — valid JSON only."""


def _extract_rules(
    llm: ChatOpenAI,
    guideline_text: str,
    source: str,
    id_offset: int,
) -> list[dict]:
    """Extract up to 3 structured rules from a single guideline abstract."""
    query = (
        f"Source: {source}\n\n"
        f"Guideline abstract:\n{guideline_text[:1200]}\n\n"
        "Extract clinical safety rules as a JSON array."
    )
    try:
        result = llm.invoke([
            SystemMessage(content=_EXTRACT_SYSTEM),
            HumanMessage(content=query),
        ])
        content = result.content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        raw_rules: list = json.loads(content)
        if not isinstance(raw_rules, list):
            return []

        validated: list[dict] = []
        for i, rule in enumerate(raw_rules[:3]):
            if not isinstance(rule, dict):
                continue
            rule["rule_id"] = f"LLM_G{id_offset + len(validated) + 1:03d}"
            rule.setdefault("active", True)
            rule.setdefault("auto_fire", False)
            rule.setdefault("and_keywords", [])
            rule.setdefault("and_fields", [])
            rule.setdefault("severity_is", [])
            rule.setdefault("emergency_must_be", None)
            rule.setdefault("constraint_requires_both", False)
            # Accept only rules with valid action and at least some matching criteria
            if (
                rule.get("action") in ("REJECT", "REVIEW")
                and isinstance(rule.get("keywords"), list)
            ):
                validated.append(rule)
        return validated
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("[RULE_GEN] Extraction failed for '%s': %s", source[:60], exc)
        return []


def seed_llm_rules() -> int:
    """
    Generate clinical safety rules from ChromaDB guidelines and store in MongoDB.

    Skips generation if LLM-generated rules already exist.

    Returns:
        Number of rules generated and stored (0 if skipped or failed).
    """
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        col = client[MONGODB_DB_NAME]["xai_validation_rules"]

        existing = col.count_documents({"rule_id": {"$regex": "^LLM_G"}})
        if existing > 0:
            logger.info("[RULE_GEN] %d LLM rules already exist in MongoDB — skipping generation", existing)
            return existing

    except Exception as exc:
        logger.warning("[RULE_GEN] MongoDB check failed — skipping LLM rule generation: %s", exc)
        return 0

    # Retrieve guideline abstracts from ChromaDB
    guidelines: list[dict] = []
    seen_texts: set[str] = set()
    for query in _SEED_QUERIES:
        try:
            results = search_guidelines(query, k=2)
            for g in results:
                text_key = g["text"][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    guidelines.append(g)
        except Exception as exc:
            logger.warning("[RULE_GEN] ChromaDB query failed: %s", exc)

    if not guidelines:
        logger.warning("[RULE_GEN] No guidelines retrieved from ChromaDB — skipping LLM rule generation")
        return 0

    logger.info("[RULE_GEN] Generating rules from %d unique guideline abstract(s)...", len(guidelines))

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    all_rules: list[dict] = []

    for g in guidelines[:10]:  # cap at 10 to control LLM call cost
        rules = _extract_rules(llm, g["text"], g.get("source", "PubMed"), len(all_rules))
        all_rules.extend(rules)
        logger.debug("[RULE_GEN] Extracted %d rule(s) from '%s'", len(rules), g.get("source", "?"))

    if not all_rules:
        logger.warning("[RULE_GEN] No valid rules extracted from guidelines")
        return 0

    try:
        ops = [
            pymongo.UpdateOne(
                {"rule_id": rule["rule_id"]},
                {"$set": rule},
                upsert=True,
            )
            for rule in all_rules
        ]
        col.bulk_write(ops, ordered=False)
        logger.info("[RULE_GEN] Stored %d LLM-generated rules in MongoDB", len(all_rules))

        # Refresh cache to include new rules
        load_rules(refresh=True)

        return len(all_rules)
    except Exception as exc:
        logger.warning("[RULE_GEN] Failed to store LLM rules: %s", exc)
        return 0
