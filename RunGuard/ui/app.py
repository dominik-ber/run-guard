import base64
from pathlib import Path

import streamlit as st

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

st.set_page_config(page_title="RunGuard", page_icon=str(LOGO_PATH))

# Header: logo + title in one flex row so they stay vertically centred
logo_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
        <img src="data:image/png;base64,{logo_b64}" width="120">
        <div>
            <h1 style="margin: 0; padding: 0;">RunGuard</h1>
            <p style="margin: 0; font-size: 1.2rem;">Predict your running injury risk from your training log</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Upload --> this is where "activities.csv" download from Strava is loaded
uploaded_file = st.file_uploader("Upload your training data (CSV)", type="csv")

# Predict
if st.button("Predict"):
    # TODO: process CSV and call /predict — fake response until the API is ready
    response = {"risk_score": "Low", "probability": 0.27}

    st.write(f"Current injury risk score: **{response['risk_score']}**")
    st.write(f"Probability of injury at next run: **{response['probability']:.0%}**")

# Footer
st.divider()
st.caption("Not medical advice, prediction model trained on elite runners.")
