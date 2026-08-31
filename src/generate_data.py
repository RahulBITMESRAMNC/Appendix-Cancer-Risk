"""
generate_data.py
-----------------
This project ships with the REAL Kaggle "Appendix Cancer Prediction" dataset
already placed at data/appendix_cancer_prediction_dataset.csv, so you do not
need to run this script to reproduce the results in this repo.

This script is kept as an OPTIONAL fallback: if you ever want to demo the
pipeline without downloading the real dataset (e.g., on a machine with no
Kaggle access, or to stress-test the code on a bigger/smaller sample), it
synthesizes a dataset with the same 21 features, distributions, and ~15.1%
positive class rate described in the source paper.

Run:
    python src/generate_data.py --n_samples 260000 --out data/synthetic.csv
"""

import argparse
import numpy as np
import pandas as pd

from config import DATA_DIR
import os

RNG = np.random.default_rng(42)

COUNTRIES = ["USA", "India", "China", "Brazil", "Germany", "Nigeria",
             "Saudi Arabia", "Japan", "UK", "South Africa"]
CHRONIC = ["None", "Diabetes", "Hypertension", "Diabetes+Hypertension", "Other"]
TREATMENTS = ["Surgery", "Chemotherapy", "Radiation"]


def make_dataset(n_samples: int, positive_rate: float = 0.151) -> pd.DataFrame:
    n_pos = int(n_samples * positive_rate)
    n_neg = n_samples - n_pos
    y = np.array([1] * n_pos + [0] * n_neg)
    RNG.shuffle(y)

    df = pd.DataFrame({
        "Patient_ID": [f"PID_{i:06d}" for i in range(1, n_samples + 1)],
        "Country": RNG.choice(COUNTRIES, n_samples, p=_weighted(COUNTRIES)),
        "Age": np.clip(RNG.normal(53.4, 20.75, n_samples), 18, 89).round().astype(int),
        "Gender": RNG.choice(["Male", "Female", "Other"], n_samples, p=[0.489, 0.491, 0.02]),
        "BMI": np.clip(RNG.normal(26, 6, n_samples), 1.1, 48.1).round(1),
        "Smoking_Status": RNG.choice(["Yes", "No"], n_samples, p=[0.25, 0.75]),
        "Alcohol_Consumption": RNG.choice(["Low", "Moderate", "High"], n_samples),
        "Family_History_Cancer": RNG.choice(["Yes", "No"], n_samples, p=[0.2, 0.8]),
        "Genetic_Mutations": RNG.choice(["Yes", "No"], n_samples, p=[0.12, 0.88]),
        "Chronic_Diseases": RNG.choice(CHRONIC, n_samples),
        "Physical_Activity_Level": RNG.choice(["Low", "Moderate", "High"], n_samples),
        "Diet_Type": RNG.choice(["Vegetarian", "Non-Vegetarian", "Vegan"], n_samples),
        "Radiation_Exposure": RNG.choice(["Yes", "No"], n_samples, p=[0.1, 0.9]),
        "Previous_Cancers": RNG.choice(["Yes", "No"], n_samples, p=[0.08, 0.92]),
        "Blood_Pressure": RNG.integers(90, 180, n_samples),
        "Cholesterol_Level": RNG.integers(150, 300, n_samples),
        "White_Blood_Cell_Count": np.round(np.clip(RNG.normal(6.5, 2.2, n_samples), 0.5, 13.7), 1),
        "Red_Blood_Cell_Count": np.round(np.clip(RNG.normal(4.8, 0.9, n_samples), 2.8, 7.6), 1),
        "Platelet_Count": RNG.integers(150, 400, n_samples),
        "Tumor_Markers": RNG.choice(["Positive", "Negative"], n_samples, p=[0.2, 0.8]),
        "Symptom_Severity": RNG.choice(["Mild", "Moderate", "Severe"], n_samples),
        "Diagnosis_Delay_Days": RNG.integers(0, 730, n_samples),
        "Treatment_Type": RNG.choice(TREATMENTS, n_samples),
        "Survival_Years_After_Diagnosis": np.round(np.clip(RNG.exponential(8, n_samples), 0, 67.8), 1),
        "Appendix_Cancer_Prediction": np.where(y == 1, "Yes", "No"),
    })
    return df


def _weighted(items):
    w = np.array([0.25, 0.20, 0.18] + [0.37 / (len(items) - 3)] * (len(items) - 3))
    return w / w.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=260_000)
    parser.add_argument("--out", type=str, default=os.path.join(DATA_DIR, "synthetic.csv"))
    args = parser.parse_args()

    df = make_dataset(args.n_samples)
    df.to_csv(args.out, index=False)
    print(f"Synthetic dataset written to {args.out} ({len(df):,} rows)")
    print(df["Appendix_Cancer_Prediction"].value_counts(normalize=True))
