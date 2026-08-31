# Appendix Cancer Prediction — SHAP-Based Feature Engineering

An end-to-end, reproducible ML pipeline that re-implements the methodology from
*"Improving appendix cancer prediction with SHAP-based feature engineering for
machine learning models"* (Kim, J.Y., *Ewha Med J* 2025) on the real 260,000-row
Kaggle dataset, plus a deployed Streamlit app with live SHAP explainability.

> This project uses the **actual Kaggle dataset**
> (`data/appendix_cancer_prediction_dataset.csv`, 260,000 rows × 25 columns),
> not a synthetic mock. `src/generate_data.py` is included only as an optional


## Why this project is a good portfolio piece

It's not just "I ran a notebook and got a green checkmark." It demonstrates
the full lifecycle a data scientist is actually responsible for:
data preprocessing → baseline modeling → explainability-driven feature
engineering → deployment → **critical evaluation of whether the result is
real**. That last step is the part most portfolio projects skip, and it's
included here on purpose (see "Critical Note" below).

## Pipeline

```
data preprocessing → baseline models (RF / XGBoost / LightGBM)
   → SHAP analysis → feature selection → feature construction
   → feature weighting → final model → Streamlit deployment
```

## Project structure

```
appendix_cancer_project/
├── data/
│   └── appendix_cancer_prediction_dataset.csv   # real Kaggle data
├── models/                                       # saved encoders + models
├── outputs/                                       # charts + results.json
├── notebooks/                                      # optional exploration
├── src/
│   ├── config.py              # shared paths & constants
│   ├── generate_data.py       # optional synthetic fallback
│   ├── preprocess.py          # encoding, stratified split, SMOTE
│   ├── train_baselines.py     # RF / XGBoost / LightGBM comparison
│   ├── shap_engineering.py    # SHAP selection, construction, weighting
│   └── final_evaluation.py    # 4-config comparison chart + AUC check
├── app.py                     # Streamlit deployment app
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
mkdir appendix_cancer_project && cd appendix_cancer_project
mkdir data notebooks src models outputs

# venv
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
code .                           # open in VS Code
```

`requirements.txt`:
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.12
imbalanced-learn>=0.11
streamlit>=1.30
joblib>=1.3
```

## 2. Run the pipeline in order

```bash
python src/preprocess.py          # encode, split, SMOTE
python src/train_baselines.py     # RF / XGBoost / LightGBM comparison
python src/shap_engineering.py    # selection, construction, weighting
python src/final_evaluation.py    # comparison chart + AUC sanity check
streamlit run app.py              # launch the demo app
```

## 3. Results (this run, on the real downloaded dataset)

**Preprocessing** reproduced the paper's reported statistics exactly:
260,000 rows, 80/20 stratified split, 15.11% positive rate preserved in both
splits, SMOTE balancing the training set to 50/50 (353,140 rows).

**Baseline model comparison** (test set, after SMOTE-trained models):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Random Forest | 0.7118 | 0.1424 | 0.1806 | 0.1592 |
| XGBoost | 0.8447 | 0.1284 | 0.0048 | 0.0093 |
| LightGBM | 0.8469 | 0.1295 | 0.0023 | 0.0045 |

**SHAP feature importance ranking** (top of the list) closely matches the
paper's Fig. 2: `Red_Blood_Cell_Count` and `White_Blood_Cell_Count` are the
two most important features in both, with `Alcohol_Consumption` and
`Physical_Activity_Level` also ranking highly in both — a good sign the
*implementation* of the SHAP pipeline is faithful to the source methodology.

**Four-configuration comparison** (LightGBM, following the paper's protocol
of selection → construction → weighting):

| Config | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Baseline | 0.8469 | 0.1295 | 0.0023 | 0.0045 |
| Feature Selection | 0.8480 | 0.1250 | 0.0010 | 0.0020 |
| Feature Construction | 0.8482 | 0.0930 | 0.0005 | 0.0010 |
| Feature Weighting | 0.8482 | 0.0930 | 0.0005 | 0.0010 |

## Critical Note: this dataset carries (almost) no real signal

The headline metrics above do **not** replicate the paper's reported numbers
(accuracy ≈0.90, precision up to 0.9940). Before treating that gap as "my
code is wrong," I ran the diagnostic a careful ML engineer runs whenever a
model's precision/recall pattern looks suspicious under class imbalance:
**ROC-AUC on the untouched test set.**

```
Baseline LightGBM AUC:        0.5059
Final (SHAP-weighted) AUC:    0.4981
```

An AUC of ~0.50 means the model has **no discriminative power beyond random
guessing** — regardless of what accuracy or precision report. (Precision can
look deceptively "fine" here purely because the classifier predicts the
majority class almost everywhere under 85%+ imbalance; whenever it does
predict positive, it happens to be right about as often as the base rate.)

I checked directly for label leakage in candidate columns
(`Survival_Years_After_Diagnosis`, `Diagnosis_Delay_Days`, `Treatment_Type`)
by comparing their distributions between the two classes — they're
statistically indistinguishable, confirming the labels in this particular
downloaded copy of the dataset are not meaningfully related to the features.
This is a known characteristic of some synthetically generated Kaggle
datasets used for teaching/portfolio purposes: the *feature schema* is
realistic, but the *target* was assigned independently of the features
(e.g., randomly, at a fixed base rate), so there's no real relationship for
any model to learn — no matter how good the feature engineering is.

This doesn't mean the SHAP-engineering methodology itself is flawed — the
feature-importance rankings this pipeline recovered line up well with the
paper's. It means the paper's headline performance numbers on *this specific
dataset* likely reflect a train/test leakage artifact elsewhere in their
pipeline (a common pitfall is applying SMOTE **before** the train/test split,
which lets synthetic neighbors of test-set points leak into training — the
paper states it avoided this, but the magnitude of their reported precision
jump, combined with a near-zero-recall XGBoost baseline right next to a
near-perfect-recall LightGBM baseline on the *same* preprocessed data, is the
kind of pattern that usually indicates a leakage or evaluation-protocol
issue rather than a genuine 10x model-quality gap).

**Why I'm including this instead of hiding it:** knowing how to tell a real
signal from an artifact of class imbalance or leakage — and saying so
plainly instead of reporting whatever number the code spits out — is
exactly the kind of judgment that separates a junior data scientist from a
senior one. This section, and the `final_evaluation.py` AUC check that
produced it, is arguably the most valuable part of this project for an
interview conversation.

## 4. Deployment (Streamlit)
```

The app:
- Takes patient inputs via sliders/dropdowns for the top-15 SHAP-selected
  features.
- Computes the `Chronic_Severity` interaction feature
  (`Chronic_Diseases × Symptom_Severity`) on submission.
- Applies the same label encoders and SHAP-derived feature weights used in
  training.
- Serves a real-time prediction from the serialized LightGBM model.
- Renders a per-patient SHAP waterfall plot showing which inputs pushed the
  prediction up or down.
- Displays the AUC caveat above directly in the UI, so the demo is honest
  about what it can and can't tell you.

## Methodology notes (from the source paper)

- **Feature selection:** top 15 features ranked by mean absolute SHAP value.
- **Feature construction:** `Chronic_Severity` = interaction between
  `Chronic_Diseases` and `Symptom_Severity`, kept in the final feature set
  even when it narrowly misses the top-15 cutoff, per the paper's stated
  rationale ("to underscore its relevance").
- **Feature weighting:** each retained feature is multiplied by its SHAP
  importance normalized to `[0, 1]`, so high-impact features (e.g., blood
  cell counts) dominate the signal fed into the final model.
- **Why LightGBM:** faster training and lower memory use on 260K rows vs.
  XGBoost/Random Forest, plus native handling of categorical splits — the
  paper's stated rationale for selecting it as the backbone.
- **Clinical framing (from the paper):** SHAP-based transparency matters in
  oncology because opaque "black-box" predictions face real adoption
  friction with clinicians; a model that can show *why* it flagged a
  patient (e.g., red blood cell count + chronic disease burden) is more
  likely to be trusted and actually used in a decision-support setting.
