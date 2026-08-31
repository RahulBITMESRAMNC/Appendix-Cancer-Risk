
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

from config import (
    ENCODERS_PATH, FINAL_MODEL_PATH, FEATURE_LIST_PATH, SHAP_WEIGHTS_PATH,
    TARGET_COL,
)

st.set_page_config(page_title="Appendix Cancer Risk (Demo)", layout="wide")


@st.cache_resource
def load_artifacts():
    encoders = joblib.load(ENCODERS_PATH)
    model = joblib.load(FINAL_MODEL_PATH)
    top_features = joblib.load(FEATURE_LIST_PATH)
    weights = joblib.load(SHAP_WEIGHTS_PATH)
    return encoders, model, top_features, weights


encoders, model, top_features, weights = load_artifacts()

st.title("Appendix Cancer Risk")
st.warning(
    "**Reproducibility caveat:** an AUC sanity check on the held-out test "
    "set came back at ~0.50 (chance level) for this dataset copy — see the "
    "README. This app demonstrates a full modelling + explainability "
    "pipeline, but its output is not a validated clinical risk score.",
    icon="⚠️",
)

st.sidebar.header("Patient inputs")


def cat_options(col):
    return list(encoders[col].classes_) if col in encoders else None


with st.sidebar:
    age = st.slider("Age", 18, 89, 53)
    gender = st.selectbox("Gender", cat_options("Gender"))
    bmi = st.slider("BMI", 1.1, 48.1, 26.0)
    country = st.selectbox("Country", cat_options("Country"))
    smoking = st.selectbox("Smoking Status", cat_options("Smoking_Status"))
    alcohol = st.selectbox("Alcohol Consumption", cat_options("Alcohol_Consumption"))
    activity = st.selectbox("Physical Activity Level", cat_options("Physical_Activity_Level"))
    diet = st.selectbox("Diet Type", cat_options("Diet_Type"))
    family_hist = st.selectbox("Family History of Cancer", cat_options("Family_History_Cancer"))
    genetic = st.selectbox("Genetic Mutations", cat_options("Genetic_Mutations"))
    chronic = st.selectbox("Chronic Diseases", cat_options("Chronic_Diseases"))
    radiation = st.selectbox("Radiation Exposure", cat_options("Radiation_Exposure"))
    prev_cancer = st.selectbox("Previous Cancers", cat_options("Previous_Cancers"))
    bp = st.slider("Blood Pressure", 90, 179, 120)
    cholesterol = st.slider("Cholesterol Level", 150, 299, 200)
    wbc = st.slider("White Blood Cell Count", 0.5, 13.7, 6.5)
    rbc = st.slider("Red Blood Cell Count", 2.8, 7.6, 4.8)
    platelet = st.slider("Platelet Count", 150, 399, 250)
    tumor_marker = st.selectbox("Tumor Markers", cat_options("Tumor_Markers"))
    severity = st.selectbox("Symptom Severity", cat_options("Symptom_Severity"))
    delay_days = st.slider("Diagnosis Delay (days)", 0, 729, 180)
    treatment = st.selectbox("Treatment Type", cat_options("Treatment_Type"))
    survival_years = st.slider("Survival Years After Diagnosis", 0.0, 67.8, 5.0)

    submitted = st.button("Predict risk", type="primary")


def build_input_row():
    raw = {
        "Country": country, "Age": age, "Gender": gender, "BMI": bmi,
        "Smoking_Status": smoking, "Alcohol_Consumption": alcohol,
        "Family_History_Cancer": family_hist, "Genetic_Mutations": genetic,
        "Chronic_Diseases": chronic, "Physical_Activity_Level": activity,
        "Diet_Type": diet, "Radiation_Exposure": radiation,
        "Previous_Cancers": prev_cancer, "Blood_Pressure": bp,
        "Cholesterol_Level": cholesterol, "White_Blood_Cell_Count": wbc,
        "Red_Blood_Cell_Count": rbc, "Platelet_Count": platelet,
        "Tumor_Markers": tumor_marker, "Symptom_Severity": severity,
        "Diagnosis_Delay_Days": delay_days, "Treatment_Type": treatment,
        "Survival_Years_After_Diagnosis": survival_years,
    }
    df = pd.DataFrame([raw])

    # Apply the same label encoders fit during preprocessing.
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Composite feature engineered from the SHAP interaction analysis.
    df["Chronic_Severity"] = df["Chronic_Diseases"] * df["Symptom_Severity"]

    # Keep only the final selected features, in the trained order, then
    # apply the same SHAP-derived weights used at training time.
    df_final = df[top_features].mul(pd.Series(weights), axis=1)
    return df_final


col1, col2 = st.columns([1, 1])

if submitted:
    X_input = build_input_row()
    proba = model.predict_proba(X_input)[0, 1]
    pred_label = encoders[TARGET_COL].inverse_transform(
        [int(proba >= 0.5)]
    )[0]

    with col1:
        st.subheader("Prediction")
        st.metric("Predicted class", pred_label)
        st.metric("Predicted probability of cancer", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

    with col2:
        st.subheader("Why this prediction? (SHAP waterfall)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_input)
        # Binary LightGBM sometimes returns 3D shap Explanation (n, features, classes)
        sv = shap_values[0]
        if sv.values.ndim > 1:
            sv = sv[:, 1]
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.plots.waterfall(sv, show=False, max_display=12)
        st.pyplot(fig, bbox_inches="tight")

    st.caption(
        "The waterfall plot shows how each (SHAP-weighted) feature pushed "
        "this specific prediction above or below the model's baseline "
        "output — a localized, per-patient diagnostic footprint."
    )
else:
    st.info("Set the patient's values in the sidebar, then click **Predict risk**.")
