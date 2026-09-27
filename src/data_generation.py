from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_data(n_patients: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    age = rng.integers(24, 82, size=n_patients)
    bmi = np.clip(rng.normal(27.5, 5.5, size=n_patients), 16, 45)
    sleep_hours = np.clip(rng.normal(6.4, 1.8, size=n_patients), 2.5, 10.0)
    resting_hr = np.clip(rng.normal(72, 12, size=n_patients), 45, 120)
    hrv = np.clip(rng.normal(40, 12, size=n_patients), 8, 90)
    spo2 = np.clip(rng.normal(97.5, 1.8, size=n_patients), 88, 100)
    activity_score = np.clip(rng.normal(52, 22, size=n_patients), 0, 100)
    stress_index = np.clip(rng.normal(42, 20, size=n_patients), 0, 100)

    diabetes = rng.binomial(1, 0.26, size=n_patients)
    hypertension = rng.binomial(1, 0.34, size=n_patients)
    smoker = rng.binomial(1, 0.18, size=n_patients)

    risk_score = (
        0.06 * (age - 50)
        + 0.18 * (bmi - 25)
        - 0.22 * (sleep_hours - 7)
        + 0.04 * (resting_hr - 70)
        - 0.08 * (hrv - 40)
        - 0.16 * (spo2 - 98)
        - 0.03 * (activity_score - 50)
        + 0.015 * stress_index
        + 0.7 * diabetes
        + 0.8 * hypertension
        + 0.9 * smoker
    )

    risk_probability = 1 / (1 + np.exp(-risk_score))
    risk_label = (risk_probability > 0.52).astype(int)

    df = pd.DataFrame(
        {
            "patient_id": [f"P-{i:03d}" for i in range(1, n_patients + 1)],
            "age": age,
            "bmi": np.round(bmi, 2),
            "sleep_hours": np.round(sleep_hours, 2),
            "resting_hr": np.round(resting_hr, 2),
            "hrv": np.round(hrv, 2),
            "spo2": np.round(spo2, 2),
            "activity_score": np.round(activity_score, 2),
            "stress_index": np.round(stress_index, 2),
            "diabetes": diabetes,
            "hypertension": hypertension,
            "smoker": smoker,
            "risk_probability": np.round(risk_probability, 4),
            "risk_label": risk_label,
        }
    )

    return df
