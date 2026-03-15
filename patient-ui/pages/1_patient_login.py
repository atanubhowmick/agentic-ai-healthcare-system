import streamlit as st
from components.banner import render_banner, render_footer

st.set_page_config(
    page_title="Healthcare AI - Patient Check-in",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- Page-specific CSS --------------------------------------------------------
st.markdown(
    """
    <style>
        .stTextInput > label {
            font-weight: 600;
            color: #37474F;
        }
        div[data-testid="stFormSubmitButton"] > button {
            background-color: #1565C0;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            border: none;
            width: 100%;
            margin-top: 0.5rem;
            transition: background-color 0.2s;
        }
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #0D47A1;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -- Session state init --------------------------------------------------------
if "patient_id" not in st.session_state:
    st.session_state.patient_id = ""
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

# -- Banner --------------------------------------------------------------------
render_banner()

# -- Centered form card --------------------------------------------------------
_, col, _ = st.columns([1, 2, 1])

with col:
    st.markdown("#### Patient Check-in")
    st.caption(
        "Note: In an Enterprise application, this screen will be replaced by a "
        "secure patient login flow. All the patient details will be captured based "
        "on login credential."
    )

    with st.form("patient_checkin_form"):
        patient_id = st.text_input(
            "Patient ID",
            placeholder="e.g. P-12345",
            help="Your unique patient identifier",
        )
        patient_name = st.text_input(
            "Full Name",
            placeholder="e.g. John Smith",
            help="Your full name as registered",
        )
        submitted = st.form_submit_button("Continue →", use_container_width=True)

    if submitted:
        if not patient_id.strip():
            st.error("Patient ID is required.")
        elif not patient_name.strip():
            st.error("Full Name is required.")
        else:
            st.session_state.patient_id = patient_id.strip()
            st.session_state.patient_name = patient_name.strip()
            st.switch_page("pages/2_diagnosis.py")

render_footer()
