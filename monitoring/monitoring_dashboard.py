# ============================================================
# MONITORING DASHBOARD — ANN From Scratch: MNIST Digit Recognizer
# ============================================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

st.set_page_config(page_title="MNIST ANN Monitoring", layout="wide")
st.title("🔢 ANN From Scratch — MNIST Digit Recognizer Monitoring Dashboard")

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv("MNIST_API_URL", "http://localhost:8000") + "/predict"

# ── Thresholds ───────────────────────────────────────────────
PSI_MODERATE = 0.10
PSI_HIGH = 0.20
LOW_CONFIDENCE_ALERT = 0.55

# ── Path resolution — Streamlit Cloud safe ────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MONITOR_PATH = os.path.join(BASE_DIR, "ann_models", "monitor_scores.csv")
LOG_PATH = os.path.join(BASE_DIR, "logs", "prediction_logs.csv")
PSI_PATH = os.path.join(BASE_DIR, "ann_models", "feature_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "ann_models", "challenger_log.json")
REGISTRY_PATH = os.path.join(BASE_DIR, "ann_models", "latest_model.json")


# ============================================================
# SIDEBAR — LIVE DIGIT PREDICTION (draw-a-digit style upload)
# ============================================================

st.sidebar.header("🔮 Predict a Digit")
st.sidebar.caption("Upload a 28x28 grayscale digit image, or generate random pixels to sanity-check the API.")

uploaded = st.sidebar.file_uploader("Upload PNG/JPG (28x28 or will be resized)", type=["png", "jpg", "jpeg"])
use_random = st.sidebar.button("Use random noise instead")

pixels_payload = None

if uploaded is not None:
    try:
        from PIL import Image

        img = Image.open(uploaded).convert("L").resize((28, 28))
        pixels_payload = (255 - np.array(img)).flatten().astype(float).tolist()  # invert if needed
        st.sidebar.image(img, caption="Resized 28x28 input", width=100)
    except Exception as e:
        st.sidebar.error(f"Could not read image: {e}")

if use_random:
    pixels_payload = np.random.randint(0, 256, 784).astype(float).tolist()

if pixels_payload is not None:
    with st.sidebar:
        with st.spinner("Calling API..."):
            try:
                response = requests.post(API_URL, json={"pixels": pixels_payload}, timeout=15)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Predicted digit: **{result['predicted_digit']}**")
                    st.markdown(f"Confidence: `{result['confidence']}`")
                    probs = pd.Series(result["probabilities"]).astype(float)
                    st.bar_chart(probs)
                else:
                    st.error(f"API error: HTTP {response.status_code}")
                    st.code(response.text[:300])
            except requests.exceptions.Timeout:
                st.warning(
                    "Request timed out. If deployed on Render free tier, the API may be cold-starting."
                )
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(MONITOR_PATH):
    df_monitor = pd.read_csv(MONITOR_PATH)
    if "confidence" in df_monitor.columns:
        avg_conf = df_monitor["confidence"].mean()
        if avg_conf < LOW_CONFIDENCE_ALERT:
            st.error(
                f"⚠️ LOW CONFIDENCE ALERT — mean prediction confidence is {avg_conf:.3f} "
                f"(threshold {LOW_CONFIDENCE_ALERT}). Model may be facing out-of-distribution inputs."
            )
            alerts_found = True

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    high_drift = df_psi[df_psi["psi_mean_intensity"] >= PSI_HIGH]
    moderate_drift = df_psi[
        (df_psi["psi_mean_intensity"] >= PSI_MODERATE) & (df_psi["psi_mean_intensity"] < PSI_HIGH)
    ]
    if len(high_drift) > 0:
        st.error(
            f"🔴 CRITICAL DRIFT — scenario(s) {list(high_drift['scenario'])} exceed PSI {PSI_HIGH}. Retrain recommended."
        )
        alerts_found = True
    if len(moderate_drift) > 0:
        st.warning(
            f"🟠 MODERATE DRIFT — scenario(s) {list(moderate_drift['scenario'])} exceed PSI {PSI_MODERATE}. Monitor closely."
        )
        alerts_found = True

if not alerts_found:
    st.success("✅ No active alerts — model operating within normal parameters.")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER HISTORY
# ============================================================

st.markdown("---")
st.subheader("🏆 Champion vs Challenger History")

if os.path.exists(REGISTRY_PATH):
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    st.markdown(
        f"**Current Champion:** `{registry.get('model_prefix', 'unknown')}`  "
        f"(updated {registry.get('updated_at', 'n/a')})"
    )

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        history = json.load(f)
    if history:
        df_hist = pd.DataFrame(history)
        display_cols = [
            c
            for c in [
                "evaluated_at",
                "challenger_name",
                "decision",
                "challenger_accuracy",
                "challenger_macro_f1",
                "champion_name",
                "champion_accuracy",
                "reason",
            ]
            if c in df_hist.columns
        ]
        st.dataframe(
            df_hist[display_cols].sort_values("evaluated_at", ascending=False), width="stretch"
        )
    else:
        st.info("No challenger evaluations logged yet.")
else:
    st.info("No challenger log found — run scripts/train_model.py to create one.")


# ============================================================
# SECTION 3 — KPIs + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 KPIs & Prediction Charts")

if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", len(df_log))
    col2.metric("Avg Confidence", f"{df_log['confidence'].mean():.3f}" if "confidence" in df_log else "n/a")
    col3.metric("Avg Latency (ms)", f"{df_log['latency_ms'].mean():.2f}" if "latency_ms" in df_log else "n/a")
    col4.metric(
        "Unique Digits Seen", df_log["predicted_digit"].nunique() if "predicted_digit" in df_log else "n/a"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "predicted_digit" in df_log.columns:
        df_log["predicted_digit"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#4C72B0")
        axes[0].set_title("Predicted Digit Distribution")
        axes[0].set_xlabel("Digit")
    if "confidence" in df_log.columns:
        axes[1].hist(df_log["confidence"], bins=20, color="#55A868")
        axes[1].set_title("Confidence Score Distribution")
        axes[1].set_xlabel("Confidence")
    st.pyplot(fig)
else:
    st.info("No prediction logs yet — call the /predict endpoint or run the simulator.")


# ============================================================
# SECTION 4 — PSI DRIFT MONITORING
# ============================================================

st.markdown("---")
st.subheader("📈 PSI Drift Monitoring")

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    st.dataframe(df_psi, width="stretch")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = [
        "#C44E52" if v >= PSI_HIGH else "#DD8452" if v >= PSI_MODERATE else "#55A868"
        for v in df_psi["psi_mean_intensity"]
    ]
    ax2.bar(df_psi["scenario"], df_psi["psi_mean_intensity"], color=colors)
    ax2.axhline(PSI_MODERATE, linestyle="--", color="orange", label="Moderate threshold")
    ax2.axhline(PSI_HIGH, linestyle="--", color="red", label="High threshold")
    ax2.set_ylabel("PSI (mean pixel intensity)")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("No PSI drift report yet — run scripts/run_simulation.py to generate one.")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS
# ============================================================

st.markdown("---")
st.subheader("🕒 Recent Predictions")

if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)
    st.dataframe(df_log.tail(20).sort_index(ascending=False), width="stretch")
else:
    st.info("No predictions logged yet.")