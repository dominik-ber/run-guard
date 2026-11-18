import base64
from pathlib import Path

import os

import requests
import streamlit as st

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
API_URL = os.environ.get("API_URL", "http://localhost:8000")

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

def show_probabilities(response):
    """Show the injury probability of the 3 models side by side.

    response: the JSON answer of the API, as a dictionary.
    """
    st.write("Probability of injury at next run:")

    # An older API version only returns "probability" (the main XGBoost model).
    # The UI is redeployed automatically, the API is not, so we handle both cases.
    if "probability_mlp" not in response:
        st.metric("XGBoost", f"{response['probability']:.1%}")
        return

    # Three columns next to each other, one model per column
    column_1, column_2, column_3 = st.columns(3)
    column_1.metric(
        "XGBoost",
        f"{response['probability']:.1%}",
        help="Main model. Injury days get extra weight in training (scale_pos_weight).",
    )
    column_2.metric(
        "XGBoost, no weight",
        f"{response['probability_xgb_no_weight']:.1%}",
        help="The same XGBoost, trained with every day counting the same.",
    )
    column_3.metric(
        "Neural network (MLP)",
        f"{response['probability_mlp']:.1%}",
        help="Small Keras neural network, trained without extra weight.",
    )

    st.caption(
        "The main XGBoost counts every injury day about 75 times in training, "
        "so its numbers come out much higher. The two other models are trained without this weight: "
        "their numbers are on the real scale, where the average injury rate is about 1.5% per day."
    )


def call_api_and_show(method, path, **kwargs):
    """Call the prediction API and show the probabilities, or the error it returned."""
    try:
        r = requests.request(method, f"{API_URL}{path}", **kwargs)
    except requests.exceptions.RequestException:
        st.error(f"Could not reach the prediction API at {API_URL}.")
        return

    if r.ok:
        show_probabilities(r.json())
    else:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(f"Prediction failed: {detail}")


strava_tab, garmin_tab = st.tabs(["Strava upload", "Garmin"])

# Strava: the user uploads the "activities.csv" from their Strava data export
with strava_tab:
    uploaded_file = st.file_uploader("Upload your Strava activities.csv", type="csv")

    if st.button("Predict", key="predict_strava"):
        if uploaded_file is None:
            st.warning("Please upload your activities.csv first.")
        else:
            # Send the raw CSV; the API cleans it and builds the features
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            call_api_and_show("POST", "/predict", files=files, timeout=30)

# Garmin: the API fetches the activities live with the user's own Personal API key
with garmin_tab:
    api_key = st.text_input(
        "Garmin Personal API key",
        type="password",  # masked in the browser; only kept for this session
        help="The key is only used for this request and is never stored.",
    )
    st.caption(
        "Uses your last 7 days of Garmin activities. "
        "[How to get a Personal API key](https://developer.garmin.com/gc-developer-program/personal/personal-api/). "
        "Leave empty to try the demo account."
    )

    if st.button("Fetch from Garmin & predict", key="predict_garmin"):
        with st.spinner("Fetching activities from Garmin..."):
            call_api_and_show("POST", "/predict/garmin", json={"api_key": api_key}, timeout=60)

# Footer
st.divider()
st.caption("Not medical advice, prediction model trained on elite runners.")
