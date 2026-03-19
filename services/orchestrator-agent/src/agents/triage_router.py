"""
4-Tier Hybrid Triage Router

Decision cascade (each tier escalates only when confidence is insufficient):

  Tier 1 - Rule Router              : keyword dominance >= RULE_DOMINANCE_RATIO           (~0ms,   free)
  Tier 2 - BioBERT Zero-Shot NLI    : top label score   >= BIOBERT_CONFIDENCE_THRESHOLD   (~150ms, free)
  Tier 3 - ClinicalBERT Classifier  : softmax probability >= CLINICAL_CONFIDENCE_THRESHOLD (~120ms, free)
  Tier 4 - LLM Fallback             : gpt-5.2 (always returns)                            (~600ms, paid)

Models:
  Tier 2: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
          Used as zero-shot-classification pipeline (NLI entailment scoring).
          No exemplar sentences needed - the model scores each candidate label directly.
  Tier 3: Fine-tuned emilyalsentzer/Bio_ClinicalBERT                (AutoModelForSequenceClassification)
          Loaded from CLINICALBERT_MODEL_DIR (produced by train_clinicalbert.py).
          If the model directory does not exist, Tier 3 is skipped gracefully.
  Tier 4: ChatOpenAI(model="gpt-5.2", temperature=0)

secondary_check_needed is determined by a pure keyword rule at every tier - no LLM required.
"""

import json
import asyncio
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from core.config import (
    RULE_DOMINANCE_RATIO,
    RULE_MIN_KEYWORD_HITS,
    BIOBERT_CONFIDENCE_THRESHOLD,
    CLINICAL_CONFIDENCE_THRESHOLD,
    CLINICALBERT_MODEL_DIR,
)
from log.logger import logger


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
_BIOBERT_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
_BIOBERT_HYPOTHESIS_TEMPLATE = "This patient requires {} specialist care."

# Tier 2 - BioBERT zero-shot NLI pipeline singleton
_biobert_zsc = None   # transformers zero-shot-classification pipeline

# Tier 3 - Fine-tuned ClinicalBERT classifier singletons
_clinicalbert_tokenizer: AutoTokenizer | None = None
_clinicalbert_clf:       AutoModelForSequenceClassification | None = None
_clinicalbert_available: bool = False   # True once the fine-tuned model is loaded

SPECIALISTS = ["cardiology", "neurology", "cancer", "pathology"]

# ---------------------------------------------------------------------------
# LLM (Tier 4 fallback) - mirrors nodes.py setup
# ---------------------------------------------------------------------------
_llm = ChatOpenAI(model="gpt-5.2", temperature=0)

_TRIAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the Master Orchestrator for an Agentic Healthcare Framework.
Triage patient symptoms and route to the most appropriate medical specialist.

Available specialists:
- cardiology  : heart/cardiovascular symptoms (chest pain, palpitations, hypertension, shortness of breath with cardiac indicators, arrhythmia)
- neurology   : neurological symptoms (headaches, seizures, memory loss, tremors, numbness, paralysis, dizziness, cognitive decline)
- cancer      : oncology symptoms (unexplained weight loss, persistent lumps/masses, abnormal bleeding, chronic fatigue with suspected malignancy, elevated tumour markers)
- pathology   : lab/test-result queries, blood disorders, metabolic issues, infections without a clear cardiac, neurological, or oncological focus

Also determine whether a secondary pathology cross-check is warranted:
Set secondary_check_needed=true when the case involves cardiology, neurology, or cancer AND
the symptoms suggest significant lab abnormalities (e.g. elevated enzymes, abnormal CBC, high tumour markers).

Respond ONLY with valid JSON (no markdown):
{{
    "specialist": "cardiology",
    "secondary_check_needed": false,
    "reasoning": "one-sentence rationale"
}}
specialist must be one of: cardiology, neurology, cancer, pathology, unknown"""),
    ("human", "Patient symptoms: {symptoms}"),
])

_triage_chain = _TRIAGE_PROMPT | _llm


# ---------------------------------------------------------------------------
# Keyword banks (Tier 1)
# Derived directly from the LLM prompt's specialist descriptions.
# ---------------------------------------------------------------------------
_KEYWORD_MAP: dict[str, list[str]] = {
    "cardiology": [
        "chest pain", "tightness", "palpitation", "palpitations",
        "difficulties in breathing", "difficulty in breathing",
        "arrhythmia", "arrhythmias", "hypertension", "heart attack",
        "myocardial", "infarction", "angina", "cardiac", "cardio",
        "shortness of breath", "dyspnea", "edema", "bradycardia",
        "tachycardia", "atrial fibrillation", "afib", "heart failure",
        "coronary", "aortic", "valve", "pericarditis",
        "endocarditis", "cardiomyopathy", "svt", "ecg abnormal",
    ],
    "neurology": [
        "headache", "migraine", "seizure", "seizures", "epilepsy",
        "memory loss", "dementia", "alzheimer", "tremor", "tremors",
        "parkinson", "numbness", "tingling", "paralysis", "weakness",
        "dizziness", "vertigo", "stroke", "tia", "transient ischemic",
        "cognitive decline", "confusion", "aphasia", "neuropathy",
        "multiple sclerosis", "meningitis", "encephalitis",
        "concussion", "spinal cord",
    ],
    "cancer": [
        "weight loss", "unexplained weight", "lump", "lumps", "mass",
        "masses", "tumour", "tumor", "malignancy", "malignant",
        "cancer", "carcinoma", "sarcoma", "lymphoma", "leukemia",
        "melanoma", "metastasis", "metastatic", "oncology", "biopsy",
        "abnormal bleeding", "haemoptysis", "hemoptysis", "chronic fatigue",
        "night sweats", "enlarged lymph", "lymph node",
        "tumour marker", "tumor marker", "ca-125", "psa elevated", "cea",
    ],
    "pathology": [
        "blood test", "lab result", "laboratory", "blood count", "cbc",
        "wbc", "rbc", "haemoglobin", "hemoglobin", "platelet",
        "creatinine", "liver function", "kidney function", "urinalysis",
        "infection", "bacteria", "bacterial", "viral", "fungal",
        "metabolic", "diabetes", "thyroid", "anaemia", "anemia",
        "inflammation", "crp", "esr", "electrolyte", "sodium", "potassium",
        "albumin", "enzyme", "elevated enzyme", "troponin",
    ],
}

# Keywords that indicate significant lab abnormalities → secondary_check_needed
_LAB_INDICATOR_KEYWORDS: list[str] = [
    "elevated", "enzyme", "enzymes", "troponin", "cbc", "wbc", "rbc",
    "hemoglobin", "haemoglobin", "platelet", "tumour marker", "tumor marker",
    "ca-125", "psa", "cea", "ldh", "creatinine", "albumin", "marker",
    "blood count", "anaemia", "anemia", "leukocyte", "lymphocyte",
    "lab", "laboratory", "blood test",
]

_SECONDARY_CHECK_SPECIALISTS = {"cardiology", "neurology", "cancer"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0]
    return json.loads(content.strip())


def _secondary_check(specialist: str, symptoms_lower: str) -> bool:
    """Pure rule: secondary pathology check needed when specialist is
    cardiology/neurology/cancer AND lab-indicator keywords are present."""
    if specialist not in _SECONDARY_CHECK_SPECIALISTS:
        return False
    return any(kw in symptoms_lower for kw in _LAB_INDICATOR_KEYWORDS)


# ---------------------------------------------------------------------------
# Tier 2 - BioBERT Zero-Shot NLI: loading
# ---------------------------------------------------------------------------

def _load_biobert_zsc():
    """Load the BioBERT zero-shot-classification pipeline (blocking)."""
    global _biobert_zsc
    logger.info("[TRIAGE_ROUTER] Loading BioBERT zero-shot-classification pipeline: %s", _BIOBERT_MODEL_NAME)
    _biobert_zsc = pipeline(
        "zero-shot-classification",
        model=_BIOBERT_MODEL_NAME,
        device=-1,   # CPU; set device=0 for GPU
    )
    logger.info("[TRIAGE_ROUTER] BioBERT zero-shot pipeline ready.")


async def _get_biobert_zsc():
    """Lazy-load the BioBERT pipeline; returns the singleton."""
    global _biobert_zsc
    if _biobert_zsc is not None:
        return _biobert_zsc
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_biobert_zsc)
    return _biobert_zsc


# ---------------------------------------------------------------------------
# Tier 3 - Fine-tuned ClinicalBERT Classifier: loading
# ---------------------------------------------------------------------------

def _load_clinicalbert_classifier(model_dir: str) -> bool:
    """
    Load fine-tuned ClinicalBERT tokenizer + model from model_dir.
    If model_dir does not exist, fine-tuning is triggered automatically first.
    Returns True on success, False only if training itself fails.
    """
    global _clinicalbert_tokenizer, _clinicalbert_clf, _clinicalbert_available
    if not os.path.isdir(model_dir):
        logger.warning(
            "[TRIAGE_ROUTER] ClinicalBERT model not found at '%s'. "
            "Run train_triage_classifier.py to generate it. Tier 3 will be skipped.",
            model_dir,
        )
        return False
    logger.info("[TRIAGE_ROUTER] Loading fine-tuned ClinicalBERT from: %s", model_dir)
    _clinicalbert_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _clinicalbert_clf = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _clinicalbert_clf.eval()
    _clinicalbert_available = True
    logger.info("[TRIAGE_ROUTER] ClinicalBERT classifier loaded | labels: %s",
                list(_clinicalbert_clf.config.id2label.values()))
    return True


async def _ensure_clinicalbert_loaded() -> bool:
    """Lazy loader - returns True if the classifier is available."""
    global _clinicalbert_available
    if _clinicalbert_available:
        return True
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _load_clinicalbert_classifier, CLINICALBERT_MODEL_DIR
    )


# ---------------------------------------------------------------------------
# Startup warm-up
# ---------------------------------------------------------------------------

async def warm_up_models() -> None:
    """Pre-load models before the first request.
    Call from FastAPI lifespan to eliminate cold-start latency."""
    await _get_biobert_zsc()
    await _ensure_clinicalbert_loaded()
    logger.info("[TRIAGE_ROUTER] All models warm and ready.")


# ---------------------------------------------------------------------------
# Tier 1 - Rule Router
# ---------------------------------------------------------------------------

def _rule_route(symptoms_lower: str) -> tuple[str | None, float]:
    """
    Returns (specialist, confidence) when:
      - total keyword hits across all domains >= RULE_MIN_KEYWORD_HITS, AND
      - winning domain's share >= RULE_DOMINANCE_RATIO
    else (None, confidence) to signal escalation to Tier 2.

    Requiring at least RULE_MIN_KEYWORD_HITS prevents a single-keyword match
    (e.g. "lump" → cancer conf=1.00) from bypassing the ML tiers.
    """
    counts: dict[str, int] = {
        s: sum(1 for kw in kws if kw in symptoms_lower)
        for s, kws in _KEYWORD_MAP.items()
    }
    total = sum(counts.values())
    logger.info("[TRIAGE_ROUTER] Tier 1 (Rule) keyword hits | total=%d (min=%d) | %s",
                total, RULE_MIN_KEYWORD_HITS,
                ", ".join(f"{s}={c}" for s, c in counts.items()))
    if total < RULE_MIN_KEYWORD_HITS:
        return None, 0.0
    winner = max(counts, key=counts.__getitem__)
    confidence = counts[winner] / total
    if confidence >= RULE_DOMINANCE_RATIO:
        return winner, confidence
    return None, confidence


# ---------------------------------------------------------------------------
# Tier 2 - BioBERT Zero-Shot NLI Router
# ---------------------------------------------------------------------------

async def _biobert_route(symptoms: str) -> tuple[str, float]:
    """Zero-shot NLI: score each specialist label via BioBERT entailment.
    Returns (top_specialist, entailment_score)."""
    zsc = await _get_biobert_zsc()
    loop = asyncio.get_event_loop()
    run_fn = lambda: zsc(
        symptoms,
        candidate_labels=SPECIALISTS,
        hypothesis_template=_BIOBERT_HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    result = await loop.run_in_executor(None, run_fn)
    specialist = result["labels"][0]
    score = float(result["scores"][0])
    return specialist, score


# ---------------------------------------------------------------------------
# Tier 3 - Fine-tuned ClinicalBERT Classifier
# ---------------------------------------------------------------------------

def _classify_with_clinicalbert(symptoms: str) -> tuple[str, float]:
    """
    Run fine-tuned ClinicalBERT inference synchronously.
    Returns (specialist, softmax_probability).
    Must only be called when _clinicalbert_available is True.
    """
    inputs = _clinicalbert_tokenizer(
        symptoms,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = _clinicalbert_clf(**inputs).logits          # (1, num_labels)
    probs = torch.softmax(logits, dim=-1).squeeze()          # (num_labels,)
    confidence = float(probs.max())
    idx = int(probs.argmax())
    # Use the label map stored in the model config (set during fine-tuning)
    label = _clinicalbert_clf.config.id2label.get(idx, SPECIALISTS[idx])
    return label, confidence


async def _clinical_route(symptoms: str) -> tuple[str, float]:
    """Async wrapper for ClinicalBERT inference."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _classify_with_clinicalbert, symptoms)


# ---------------------------------------------------------------------------
# Tier 4 - LLM Fallback
# ---------------------------------------------------------------------------

async def _llm_route(symptoms: str) -> tuple[str, bool, str]:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, _triage_chain.invoke, {"symptoms": symptoms}
    )
    raw = _parse_json(result.content)
    specialist = raw.get("specialist", "unknown").lower().strip()
    secondary  = bool(raw.get("secondary_check_needed", False))
    reasoning  = raw.get("reasoning", "")
    if specialist not in (*SPECIALISTS, "unknown"):
        specialist = "unknown"
    return specialist, secondary, reasoning


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def route_symptoms(symptoms: str) -> tuple[str, bool, str]:
    """
    Route patient symptoms through the 4-tier cascade.

    Returns:
        specialist            : "cardiology" | "neurology" | "cancer" | "pathology" | "unknown"
        secondary_check_needed: bool
        reasoning             : string describing which tier resolved the routing and its confidence
    """
    low = symptoms.lower()

    # Tier 1: Rule
    specialist, conf = _rule_route(low)
    if specialist is not None:
        secondary = _secondary_check(specialist, low)
        logger.info("[TRIAGE_ROUTER] Tier 1 (Rule) → %s | conf=%.2f", specialist, conf)
        return specialist, secondary, f"[Rule] conf={conf:.2f}"

    # Tier 2: BioBERT embedding
    specialist, score = await _biobert_route(symptoms)
    if score >= BIOBERT_CONFIDENCE_THRESHOLD:
        secondary = _secondary_check(specialist, low)
        logger.info("[TRIAGE_ROUTER] Tier 2 (BioBERT NLI) → %s | score=%.3f", specialist, score)
        return specialist, secondary, f"[BioBERT NLI] score={score:.3f}"
    logger.info("[TRIAGE_ROUTER] Tier 2 (BioBERT NLI) low confidence → %s | score=%.3f - escalating", specialist, score)

    # Tier 3: Fine-tuned ClinicalBERT classifier (skipped if model not trained yet)
    if await _ensure_clinicalbert_loaded():
        specialist, score = await _clinical_route(symptoms)
        if score >= CLINICAL_CONFIDENCE_THRESHOLD:
            secondary = _secondary_check(specialist, low)
            logger.info("[TRIAGE_ROUTER] Tier 3 (ClinicalBERT) → %s | prob=%.3f", specialist, score)
            return specialist, secondary, f"[ClinicalBERT] prob={score:.3f}"
        logger.info("[TRIAGE_ROUTER] Tier 3 (ClinicalBERT) low confidence → %s | prob=%.3f - escalating to LLM", specialist, score)
    else:
        logger.info("[TRIAGE_ROUTER] Tier 3 (ClinicalBERT) unavailable - skipping to LLM")

    # Tier 4: LLM fallback
    specialist, secondary, reasoning = await _llm_route(symptoms)
    logger.info("[TRIAGE_ROUTER] Tier 4 (LLM) → %s | reason: %s", specialist, reasoning)
    return specialist, secondary, f"[LLM] {reasoning}"
