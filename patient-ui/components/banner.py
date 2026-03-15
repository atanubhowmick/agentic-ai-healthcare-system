import streamlit as st


_COMMON_CSS = """
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
        [data-testid="stDecoration"] { display: none; }
        [data-testid="stDeployButton"] { display: none; }
        #MainMenu { visibility: hidden; }
        header[data-testid="stHeader"] { display: none !important; }

        .block-container {
            padding-top: 0 !important;
        }

        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.6rem 1.2rem;
            background: #1565C0;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        .top-bar-left {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .top-bar-icon {
            font-size: 2rem;
            line-height: 1;
        }
        .top-bar-text {
            display: flex;
            flex-direction: column;
        }
        .top-bar-title {
            color: #FFFFFF;
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.3;
        }
        .top-bar-subtitle {
            color: #BBDEFB;
            font-size: 0.8rem;
            font-weight: 400;
            line-height: 1.3;
        }
        .top-bar-patient {
            color: #BBDEFB;
            font-size: 0.95rem;
            font-weight: 500;
            text-align: right;
        }
        .top-bar-patient span {
            color: #FFFFFF;
            font-weight: 700;
        }
        .footer-note {
            text-align: center;
            color: #546E7A;
            font-size: 0.82rem;
            margin-top: 2rem;
        }
    </style>
"""


def render_banner(patient_name: str | None = None, patient_id: str | None = None) -> None:
    """Render the common page CSS and the blue top banner.

    Args:
        patient_name: When provided, shown on the right side of the banner.
        patient_id:   When provided, shown alongside patient_name.
    """
    st.markdown(_COMMON_CSS, unsafe_allow_html=True)

    patient_html = ""
    if patient_name and patient_id:
        patient_html = f"""
        <div class="top-bar-patient">
            Patient: <span>{patient_name}</span>
            &nbsp;|&nbsp; ID: <span>{patient_id}</span>
        </div>
        """

    st.markdown(
        f"""
        <div class="top-bar">
            <div class="top-bar-left">
                <span class="top-bar-icon">🏥</span>
                <div class="top-bar-text">
                    <div class="top-bar-title">Healthcare Agentic AI System</div>
                    <div class="top-bar-subtitle">AI-Powered Diagnosis with Specialized Agents</div>
                </div>
            </div>
            {patient_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the common disclaimer footer."""
    st.markdown(
        '<p class="footer-note">This system is for clinical research purposes only. '
        "Not a substitute for professional medical advice.</p>",
        unsafe_allow_html=True,
    )
