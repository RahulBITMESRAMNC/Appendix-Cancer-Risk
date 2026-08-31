"""
config.py
---------
Central place for file paths and constants shared across the pipeline.
Keeping these in one module means every script (preprocess, train, shap,
app) stays in sync without hard-coded strings scattered everywhere.
"""

import os

# --- Directory layout ---------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- File paths -----------------------------------------------------------
RAW_CSV = os.path.join(DATA_DIR, "appendix_cancer_prediction_dataset.csv")
PROCESSED_TRAIN = os.path.join(DATA_DIR, "train_processed.csv")
PROCESSED_TEST = os.path.join(DATA_DIR, "test_processed.csv")
ENGINEERED_TRAIN = os.path.join(DATA_DIR, "train_engineered.csv")
ENGINEERED_TEST = os.path.join(DATA_DIR, "test_engineered.csv")

ENCODERS_PATH = os.path.join(MODELS_DIR, "label_encoders.pkl")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "lightgbm_baseline.pkl")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "lightgbm_final.pkl")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "final_feature_list.pkl")
SHAP_WEIGHTS_PATH = os.path.join(MODELS_DIR, "shap_weights.pkl")
RESULTS_JSON = os.path.join(OUTPUTS_DIR, "results.json")

# --- Modelling constants ---------------------------------------------------
TARGET_COL = "Appendix_Cancer_Prediction"
ID_COL = "Patient_ID"
RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K_FEATURES = 15

# Columns that are categorical (string) in the raw dataset and need encoding
CATEGORICAL_COLS = [
    "Country", "Gender", "Smoking_Status", "Alcohol_Consumption",
    "Family_History_Cancer", "Genetic_Mutations", "Chronic_Diseases",
    "Physical_Activity_Level", "Diet_Type", "Radiation_Exposure",
    "Previous_Cancers", "Tumor_Markers", "Symptom_Severity", "Treatment_Type",
]
