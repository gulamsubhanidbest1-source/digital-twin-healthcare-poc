from pathlib import Path
import pandas as pd
import streamlit as st

from src.data_generation import generate_synthetic_data
from src.model import train_model, summarize_patient_risk


DATA_PATH = Path("data/healthcare_twin_data.csv")


def ensure_data_exists():
    DATA_PATH.parent.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        generated = generate_synthetic_data()
        generated.to_csv(DATA_PATH, index=False)


st.set_page_config(page_title="Digital Twin Health Risk Monitor", layout="wide")

ensure_data_exists()

df = pd.read_csv(DATA_PATH)
model, metrics = train_model(df)

st.title("Healthcare Digital Twin Risk Monitor")
st.caption("Early cardiovascular risk prediction using wearable signals + EHR-like data")

left_col, right_col = st.columns([1.5, 1])

with left_col:
    st.subheader("Model Performance")
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    st.metric("AUC-ROC", f"{metrics['roc_auc']:.2%}")
    st.metric("Precision", f"{metrics['precision']:.2%}")

with right_col:
    st.subheader("Clinical Interpretation")
    st.markdown(
        """
        This digital twin estimates a patient's short-term cardiovascular risk based on:
        - heart rate and HRV
        - sleep quantity and recovery
        - oxygen saturation and activity patterns
        - lifestyle and chronic disease indicators
        """
    )

st.divider()

st.subheader("Build a Patient Twin")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 90, 52)
        bmi = st.slider("BMI", 16.0, 42.0, 28.0, step=0.1)
        sleep_hours = st.slider("Sleep (hours)", 3.0, 10.0, 6.5, step=0.1)
        resting_hr = st.slider("Resting Heart Rate", 45, 120, 72)

    with col2:
        hrv = st.slider("HRV", 10, 90, 38)
        spo2 = st.slider("SpO2 (%)", 88, 100, 97, step=1)
        activity_score = st.slider("Activity Score", 0, 100, 52)
        diabetes = st.checkbox("Diabetes")

    with col3:
        hypertension = st.checkbox("Hypertension")
        smoker = st.checkbox("Smoker")
        stress = st.slider("Stress Index", 0, 100, 40)

    submitted = st.form_submit_button("Assess Risk")

if submitted:
    patient_row = {
        "age": age,
        "bmi": bmi,
        "sleep_hours": sleep_hours,
        "resting_hr": resting_hr,
        "hrv": hrv,
        "spo2": spo2,
        "activity_score": activity_score,
        "diabetes": int(diabetes),
        "hypertension": int(hypertension),
        "smoker": int(smoker),
        "stress_index": stress,
    }

    risk_score, explanation = summarize_patient_risk(model, patient_row)

    st.subheader("Patient Twin Summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Risk Probability", f"{risk_score:.0%}")
    summary_cols[1].metric("Age", age)
    summary_cols[2].metric("Sleep", f"{sleep_hours:.1f} hrs")
    summary_cols[3].metric("HR", f"{resting_hr} bpm")

    st.info(explanation)

    st.subheader("Key Risk Drivers")
    st.write(
        """
        - Higher resting heart rate and lower HRV often indicate reduced cardiovascular resilience.
        - Lower sleep duration and reduced activity are associated with reduced recovery capacity.
        - Elevated BMI, diabetes, hypertension, and smoking elevate long-term risk.
        - SpO2 and workload can capture signs of deteriorating patient status.
        """
    )

st.divider()

st.subheader("Dataset Snapshot")

st.dataframe(df.head(10), use_container_width=True)

st.markdown(
    """
    This app is a prototype for a hackathon submission; it is intentionally designed to be easy to explain,
    demo, and extend with real clinical data sources.
    """
)
