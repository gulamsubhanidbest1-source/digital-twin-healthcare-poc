# Digital Twin for Early Cardiovascular Risk Monitoring

A healthcare-focused digital twin prototype designed for the Digital Twin Challenge 2026. This project builds a patient-specific risk monitoring dashboard that combines wearable sensor trends with EHR-like variables to estimate early cardiovascular risk and provide explainable alerts.

## Problem Statement
Healthcare is increasingly moving toward proactive, individualized care. A digital twin is a dynamic virtual representation of a patient that continuously integrates health signals and predicts risk before a major deterioration occurs.

This prototype demonstrates how a digital twin can combine:
- wearable signals such as heart rate, HRV, sleep, activity, and SpO2
- static medical variables such as age, BMI, smoking status, hypertension, and diabetes
- machine learning-based probability estimation for near-term cardiovascular risk

## What the project does
- Generates a synthetic patient cohort representative of healthcare monitoring data
- Trains an explainable risk model for early cardiovascular risk prediction
- Allows users to simulate a patient digital twin through an interactive dashboard
- Renders a risk score, feature-based explanation, and a patient summary
- Provides a compelling submission-ready project structure for hackathon evaluation

## Demo Highlights
- Real-time risk estimation
- Personalized patient risk explanation
- Explainability through model feature contribution
- Clinical-friendly dashboard interface

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── app.py
├── data/
│   └── healthcare_twin_data.csv
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   └── model.py
└── .gitignore
```

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- XGBoost (optional) / RandomForest classification baseline

## Local Setup

1. Clone the repository:

```bash
git clone https://github.com/gulamsubhanidbest1-source/digital-twin-healthcare-poc.git
cd digital-twin-healthcare-poc
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

## Project Workflow

1. Generate synthetic patient data from wearable + EHR-like health variables.
2. Train a classification model to estimate heart-risk likelihood.
3. Create a digital twin profile for an individual patient.
4. Visualize risk trends and explain which features most strongly influence the prediction.

## Submission-ready Value Proposition
This prototype illustrates a realistic digital twin use case in healthcare:
- dynamic monitoring
- personalized risk estimation
- proactive intervention opportunity
- easy-to-demonstrate clinical dashboard

## License
MIT
