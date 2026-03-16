import os
import re
import time
import requests
import streamlit as st
from components.banner import render_banner, render_footer
from constant.constants import MAX_SYMPTOMS_CHARS, ORCHESTRATOR_URL_DEFAULT

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", ORCHESTRATOR_URL_DEFAULT)
DIAGNOSE_ENDPOINT = f"{ORCHESTRATOR_URL}/orchestrator/diagnose"

st.set_page_config(
    page_title="Healthcare AI - Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- Page-specific CSS ---------------------------------------------------------
st.markdown(
    """
    <style>
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #1565C0;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.55rem 2rem;
            border: none;
            transition: background-color 0.2s;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #0D47A1;
        }
.badge-completed {
            display: inline-block;
            background: #E8F5E9;
            color: #2E7D32;
            border: 1px solid #A5D6A7;
            border-radius: 20px;
            padding: 0.2rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .badge-review {
            display: inline-block;
            background: #FFF3E0;
            color: #E65100;
            border: 1px solid #FFCC80;
            border-radius: 20px;
            padding: 0.2rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .badge-unknown {
            display: inline-block;
            background: #ECEFF1;
            color: #546E7A;
            border: 1px solid #CFD8DC;
            border-radius: 20px;
            padding: 0.2rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .badge-danger {
            display: inline-block;
            background: #FFEBEE;
            color: #C62828;
            border: 1px solid #EF9A9A;
            border-radius: 20px;
            padding: 0.2rem 0.9rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .sev-critical { color: #B71C1C; font-weight: 700; }
        .sev-high     { color: #E64A19; font-weight: 700; }
        .sev-moderate { color: #F57F17; font-weight: 700; }
        .sev-low      { color: #2E7D32; font-weight: 700; }
        .sev-unknown  { color: #546E7A; font-weight: 600; }
        .info-row {
            display: flex;
            gap: 2rem;
            flex-wrap: wrap;
            margin: 0.8rem 0;
        }
        .info-item label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #90A4AE;
            display: block;
        }
        .info-item span {
            font-size: 0.95rem;
            font-weight: 600;
            color: #263238;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -- Session state defaults ----------------------------------------------------
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False
if "pending_symptoms" not in st.session_state:
    st.session_state.pending_symptoms = None
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None
if "response_time" not in st.session_state:
    st.session_state.response_time = None

# -- Guard: redirect if no patient info ----------------------------------------
if not st.session_state.get("patient_id") or not st.session_state.get("patient_name"):
    st.warning("Session expired. Please check in again.")
    if st.button("← Back to Check-in"):
        st.switch_page("pages/1_patient_login.py")
    st.stop()

patient_id: str = st.session_state.patient_id
patient_name: str = st.session_state.patient_name

# -- Banner --------------------------------------------------------------------
render_banner(patient_name=patient_name, patient_id=patient_id)

# -- Symptom input form --------------------------------------------------------
_, center_col, _ = st.columns([1, 3, 1])

with center_col:
    st.markdown("#### Describe Your Symptoms")
    st.markdown(
        "Provide as much detail as possible about your symptoms, their duration, and severity.",
        help=f"Maximum {MAX_SYMPTOMS_CHARS} characters",
    )

    with st.form("symptoms_form"):
        symptoms = st.text_area(
            label="Your symptoms",
            placeholder=(
                "e.g. I have been experiencing chest pain radiating to my left arm "
                "for the past two days, along with shortness of breath and mild fever..."
            ),
            height=160,
            max_chars=MAX_SYMPTOMS_CHARS,
            label_visibility="collapsed",
        )
        char_count = len(symptoms)
        st.caption(f"{char_count} / {MAX_SYMPTOMS_CHARS} characters")
        submitted = st.form_submit_button(
            "Submit for Diagnosis →",
            use_container_width=False,
            disabled=st.session_state.is_loading,
        )

    if submitted:
        if not symptoms.strip():
            st.error("Please describe your symptoms before submitting.")
        else:
            st.session_state.pending_symptoms = symptoms.strip()
            st.session_state.is_loading = True
            st.session_state.diagnosis_result = None
            st.session_state.response_time = None
            st.session_state._start_time = time.monotonic()
            st.rerun()

# -- Call orchestrator and display results -------------------------------------
with center_col:
    if st.session_state.is_loading and st.session_state.pending_symptoms:
        with st.spinner("Analysing symptoms — consulting specialist agents, please wait..."):
            try:
                resp = requests.post(
                    DIAGNOSE_ENDPOINT,
                    json={"patient_id": patient_id, "symptoms": st.session_state.pending_symptoms},
                    timeout=120,
                )
                resp.raise_for_status()
                st.session_state.diagnosis_result = resp.json()
                st.session_state.response_time = time.monotonic() - st.session_state.get("_start_time", time.monotonic())
            except requests.exceptions.ConnectionError:
                st.session_state.is_loading = False
                st.error(
                    "Unable to reach the Healthcare AI backend. "
                    "Please ensure the orchestrator service is running."
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.session_state.is_loading = False
                st.error("The request timed out. The service may be overloaded — please try again.")
                st.stop()
            except Exception as exc:
                st.session_state.is_loading = False
                st.error(f"An unexpected error occurred: {exc}")
                st.stop()
        st.session_state.is_loading = False
        st.rerun()

    if st.session_state.diagnosis_result:
        data = st.session_state.diagnosis_result

        # -- Parse response ------------------------------------------------
        if not data.get("is_success"):
            err = data.get("error", {})
            st.error(
                f"Orchestrator error [{err.get('code', 'UNKNOWN')}]: {err.get('message', 'No details available.')}"
            )
            st.stop()

        payload = data.get("payload", {})
        diagnosis = payload.get("diagnosis") or {}
        treatment = payload.get("treatment") or {}
        xai_diag = payload.get("xai_diagnosis_validation") or {}
        xai_treat = payload.get("xai_treatment_validation") or {}
        audit_trail = payload.get("audit_trail") or []
        status = payload.get("status", "UNKNOWN")
        specialist = payload.get("specialist_agent", "N/A")
        conflict = payload.get("conflict_detected", False)
        conflict_reason = payload.get("conflict_reason", "")
        human_review_reason = payload.get("human_review_reason", "")

        # -- Result card ---------------------------------------------------
        st.markdown("---")
        st.markdown("#### Diagnosis Report")

        badge_class = (
            "badge-completed" if status in ("COMPLETED", "COMPLETED_FROM_CACHE")
            else "badge-review" if status == "HUMAN_REVIEW_REQUIRED"
            else "badge-unknown"
        )
        st.markdown(
            f"""
            <div class="info-row">
                <div class="info-item">
                    <label>Status</label>
                    <span><span class="{badge_class}">{status.replace("_", " ")}</span></span>
                </div>
                <div class="info-item">
                    <label>Specialist Agent</label>
                    <span>{specialist or "N/A"}</span>
                </div>
                <div class="info-item">
                    <label>Patient ID</label>
                    <span>{payload.get("patient_id", patient_id)}</span>
                </div>
                <div class="info-item">
                    <label>Response Time</label>
                    <span>{f"{st.session_state.response_time:.2f}s" if st.session_state.response_time is not None else "N/A"}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if status == "HUMAN_REVIEW_REQUIRED" and human_review_reason:
            st.warning(f"**Human Review Required:** {human_review_reason}")

        if conflict and conflict_reason:
            st.warning(f"**Specialist Conflict Detected:** {conflict_reason}")

        # -- Diagnosis section ---------------------------------------------
        if diagnosis:
            st.markdown("---")
            st.markdown("**Diagnosis Summary**")

            severity = diagnosis.get("severity", "UNKNOWN").upper()
            sev_class = {
                "CRITICAL": "sev-critical",
                "HIGH": "sev-high",
                "MODERATE": "sev-moderate",
                "LOW": "sev-low",
            }.get(severity, "sev-unknown")

            def _yes_no_class(val: str) -> str:
                return "sev-critical" if str(val).upper() == "YES" else "sev-low" if str(val).upper() == "NO" else "sev-unknown"

            emergency = diagnosis.get("emergency_care_needed", "N/A")
            hospitalisation = diagnosis.get("hospitalization_needed", "N/A")

            st.markdown(
                f"""
                <div class="info-row">
                    <div class="info-item">
                        <label>Severity</label>
                        <span class="{sev_class}">{severity}</span>
                    </div>
                    <div class="info-item">
                        <label>Emergency Care Needed</label>
                        <span class="{_yes_no_class(emergency)}">{emergency}</span>
                    </div>
                    <div class="info-item">
                        <label>Hospitalisation Needed</label>
                        <span class="{_yes_no_class(hospitalisation)}">{hospitalisation}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            summary = diagnosis.get("summary", "")
            if summary:
                st.info(summary)

            full_details = diagnosis.get("full_details") or {}
            if full_details:
                with st.expander("Full Diagnosis Details", expanded=False):
                    _sev_map = {
                        "CRITICAL": "sev-critical", "HIGH": "sev-high",
                        "MODERATE": "sev-moderate", "LOW": "sev-low",
                    }
                    for key, val in full_details.items():
                        label = re.sub(r"([A-Z])", r" \1", key).strip().title()
                        if isinstance(val, list):
                            if val:
                                st.markdown(f"**{label}**")
                                for item in val:
                                    st.markdown(f"- {item}")
                        elif isinstance(val, bool):
                            css = "sev-critical" if val else "sev-low"
                            st.markdown(
                                f"**{label}:** <span class='{css}'>{'Yes' if val else 'No'}</span>",
                                unsafe_allow_html=True,
                            )
                        elif isinstance(val, str) and val:
                            val_upper = val.strip().upper()
                            if val_upper in ("YES", "NO"):
                                css = _yes_no_class(val_upper)
                                st.markdown(
                                    f"**{label}:** <span class='{css}'>{val}</span>",
                                    unsafe_allow_html=True,
                                )
                            elif val_upper in _sev_map:
                                css = _sev_map[val_upper]
                                st.markdown(
                                    f"**{label}:** <span class='{css}'>{val}</span>",
                                    unsafe_allow_html=True,
                                )
                            elif len(val) > 120:
                                st.markdown(f"**{label}**")
                                st.info(val)
                            else:
                                st.markdown(f"**{label}:** {val}")
                        elif val is not None:
                            st.markdown(f"**{label}:** {val}")
        else:
            st.info("No diagnosis data available in this response.")

        # -- Treatment section ---------------------------------------------
        if treatment:
            st.markdown("---")
            with st.expander("Treatment Recommendations", expanded=True):
                t = treatment.get("treatment", {})
                st.caption(f"Agent: {treatment.get('agent', 'N/A')}  ·  ID: {treatment.get('agent_id', 'N/A')}")

                urgency = t.get("urgency", "")
                if urgency:
                    urgency_class = {
                        "IMMEDIATE": "sev-critical",
                        "SOON": "sev-high",
                        "ROUTINE": "sev-low",
                    }.get(urgency.upper(), "sev-unknown")
                    st.markdown(
                        f"**Urgency:** <span class='{urgency_class}'>{urgency}</span>",
                        unsafe_allow_html=True,
                    )

                if t.get("treatmentPlan"):
                    st.markdown("**Treatment Plan**")
                    st.info(t["treatmentPlan"])

                if t.get("medications"):
                    st.markdown("**Medications**")
                    for med in t["medications"]:
                        st.markdown(f"- {med}")

                follow_up = t.get("followUpRequired", "")
                timeframe = t.get("followUpTimeframe", "")
                if follow_up:
                    suffix = f" — {timeframe}" if timeframe and timeframe != "NONE" else ""
                    st.markdown(f"**Follow-up Required:** {follow_up}{suffix}")

                referral = t.get("referralRequired", "")
                if referral and referral != "NONE":
                    st.markdown(f"**Referral:** {referral}")

                if t.get("lifestyleRecommendations"):
                    st.markdown("**Lifestyle Recommendations**")
                    for item in t["lifestyleRecommendations"]:
                        st.markdown(f"- {item}")

                if t.get("monitoringRequired"):
                    st.markdown("**Monitoring**")
                    for item in t["monitoringRequired"]:
                        st.markdown(f"- {item}")

        # -- XAI validations -----------------------------------------------
        if xai_diag or xai_treat:
            st.markdown("---")
            for xai_label, xai_data in [
                ("XAI Diagnosis Validation", xai_diag),
                ("XAI Treatment Validation", xai_treat),
            ]:
                if not xai_data:
                    continue
                with st.expander(xai_label, expanded=False):
                    result = xai_data.get("result", {})
                    st.caption(
                        f"Agent: {xai_data.get('agent', 'N/A')}  ·  "
                        f"Type: {xai_data.get('validation_type', 'N/A')}  ·  "
                        f"ID: {xai_data.get('agent_id', 'N/A')}"
                    )

                    rec = result.get("recommendation", "")
                    rec_class = {
                        "APPROVE": "badge-completed",
                        "REJECT": "badge-danger",
                        "REVIEW": "badge-review",
                    }.get(rec.upper(), "badge-unknown")
                    is_validated = result.get("is_validated", False)
                    confidence = result.get("confidence_score", 0.0)

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(
                            f"**Recommendation:** <span class='{rec_class}'>{rec}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Validated:** {'✓ Yes' if is_validated else '✗ No'}")
                    with col2:
                        st.markdown(f"**Confidence Score:** {confidence:.0%}")
                        st.progress(float(confidence))

                    if result.get("validation_summary"):
                        st.markdown("**Summary**")
                        st.info(result["validation_summary"])

                    if result.get("key_concerns"):
                        st.markdown("**Key Concerns**")
                        for concern in result["key_concerns"]:
                            st.markdown(f"- {concern}")

        # -- Audit trail ---------------------------------------------------
        if audit_trail:
            with st.expander("Audit Trail", expanded=False):
                for i, step in enumerate(audit_trail, 1):
                    st.markdown(f"`{i:02d}` {step}")


# -- Footer --------------------------------------------------------------------
render_footer()
