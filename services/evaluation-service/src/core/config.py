import os
from dotenv import load_dotenv

load_dotenv()

# -- MongoDB -------------------------------------------------------------------
MONGO_URI                     = os.getenv("MONGO_URI",                     "mongodb://127.0.0.1:27017")
MONGO_DB                      = os.getenv("MONGO_DB",                      "agentic_ai_healthcare_db")
MONGO_EVAL_COLLECTION         = os.getenv("MONGO_EVAL_COLLECTION",         "mimic_evaluation_cases")
MONGO_TFIDF_REPORT_COLLECTION = os.getenv("MONGO_TFIDF_REPORT_COLLECTION", "tfidf_baseline_reports")
MONGO_XAI_REPORT_COLLECTION   = os.getenv("MONGO_XAI_REPORT_COLLECTION",   "xai_evaluation_reports")

# -- External services ---------------------------------------------------------
XAI_SERVICE_URL = os.getenv("XAI_SERVICE_URL", "http://localhost:8016")

# -- Report output -------------------------------------------------------------
_REPORTS_BASE = os.getenv("EVALUATION_SVC_REPORTS", "evaluation-svc-reports")

XAI_VALIDATION_AGGREGATE_REPORT_PLOTS_DIR   = os.getenv("XAI_VALIDATION_AGGREGATE_REPORT_PLOTS_DIR",   os.path.join(_REPORTS_BASE, "xai-validation-svc", "aggregate_report_plots"))
XAI_VALIDATION_STATISTICAL_REPORT_PLOTS_DIR = os.getenv("XAI_VALIDATION_STATISTICAL_REPORT_PLOTS_DIR", os.path.join(_REPORTS_BASE, "xai-validation-svc", "statistical_report_plots"))
CANCER_AGENT_AGGREGATE_REPORT_PLOTS_DIR  = os.getenv("CANCER_AGENT_AGGREGATE_REPORT_PLOTS_DIR",  os.path.join(_REPORTS_BASE, "cancer-agent",        "aggregate_report_plots"))
CANCER_AGENT_STATISTICAL_REPORT_PLOTS_DIR = os.getenv("CANCER_AGENT_STATISTICAL_REPORT_PLOTS_DIR", os.path.join(_REPORTS_BASE, "cancer-agent",       "statistical_report_plots"))
