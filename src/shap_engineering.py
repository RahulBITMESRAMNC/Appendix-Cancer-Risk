"""
shap_engineering.py
--------------------
Implements the 3 SHAP-driven feature engineering steps from the paper and
trains four LightGBM configurations so their metrics can be compared:

  0. Baseline            - all original features (already trained in
                            train_baselines.py, reloaded here for the
                            comparison chart).
  1. Feature Selection   - keep only the top-15 features by mean |SHAP value|.
  2. Feature Construction- add the Chronic_Severity interaction feature
                            (Chronic_Diseases x Symptom_Severity), then
                            re-select the top-15 on the expanded set.
  3. Feature Weighting   - multiply each retained feature by its normalized
                            global SHAP importance before final training.

Run:
    python src/preprocess.py
    python src/train_baselines.py
    python src/shap_engineering.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    PROCESSED_TRAIN, PROCESSED_TEST, TARGET_COL, RANDOM_STATE,
    BASELINE_MODEL_PATH, FINAL_MODEL_PATH, FEATURE_LIST_PATH,
    SHAP_WEIGHTS_PATH, RESULTS_JSON, TOP_K_FEATURES,
)


def load_processed():
    train = pd.read_csv(PROCESSED_TRAIN)
    test = pd.read_csv(PROCESSED_TEST)
    return (train.drop(columns=[TARGET_COL]), train[TARGET_COL],
            test.drop(columns=[TARGET_COL]), test[TARGET_COL])


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }


def get_mean_abs_shap(model, X_sample: pd.DataFrame) -> pd.Series:
    """Global feature importance = mean(|SHAP value|) per feature."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # LGBMClassifier binary output: shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=X_sample.columns).sort_values(ascending=False)


def add_chronic_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction feature highlighted in the paper: chronic disease burden
    combined with symptom severity. Both inputs are already label-encoded
    integers at this point, so the product captures compounded risk
    (e.g., a patient with more chronic conditions AND more severe symptoms
    gets a disproportionately higher score than either alone)."""
    df = df.copy()
    df["Chronic_Severity"] = df["Chronic_Diseases"] * df["Symptom_Severity"]
    return df


def train_lgbm(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=300, max_depth=-1, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    model.fit(X_train, y_train)
    return model


def run():
    X_train, y_train, X_test, y_test = load_processed()

    # ---- Reload / retrain baseline (all original features) ----------------
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    baseline_metrics = evaluate(baseline_model, X_test, y_test)
    print(f"[0] Baseline              -> {baseline_metrics}")

    # SHAP explainer sample (subsample for speed on 200k+ rows)
    shap_sample = X_train.sample(n=min(20_000, len(X_train)), random_state=RANDOM_STATE)
    importance = get_mean_abs_shap(baseline_model, shap_sample)
    print("\nTop 10 features by mean |SHAP value|:")
    print(importance.head(10))

    # ---- Step 1: Feature Selection -----------------------------------------
    top_features = importance.head(TOP_K_FEATURES).index.tolist()
    X_train_sel, X_test_sel = X_train[top_features], X_test[top_features]
    model_sel = train_lgbm(X_train_sel, y_train)
    metrics_sel = evaluate(model_sel, X_test_sel, y_test)
    print(f"\n[1] Feature Selection      -> {metrics_sel}")

    # ---- Step 2: Feature Construction --------------------------------------
    X_train_con = add_chronic_severity(X_train)
    X_test_con = add_chronic_severity(X_test)

    con_sample = X_train_con.sample(n=min(20_000, len(X_train_con)), random_state=RANDOM_STATE)
    model_full_con = train_lgbm(X_train_con, y_train)
    importance_con = get_mean_abs_shap(model_full_con, con_sample)

    # Re-select top-15 from the expanded feature set, keeping Chronic_Severity
    # even if it narrowly misses the cut, to underscore its clinical relevance
    # (matching the paper's stated methodology).
    top_features_con = importance_con.head(TOP_K_FEATURES).index.tolist()
    if "Chronic_Severity" not in top_features_con:
        top_features_con = top_features_con[:-1] + ["Chronic_Severity"]

    X_train_con_final = X_train_con[top_features_con]
    X_test_con_final = X_test_con[top_features_con]
    model_con = train_lgbm(X_train_con_final, y_train)
    metrics_con = evaluate(model_con, X_test_con_final, y_test)
    print(f"[2] Feature Construction   -> {metrics_con}")

    # ---- Step 3: Feature Weighting -----------------------------------------
    # Normalize SHAP importances of the retained features to [0, 1] and use
    # them as multiplicative weights, so high-impact features (e.g. blood
    # cell counts) dominate the signal fed to the final model.
    weights = importance_con.reindex(top_features_con)
    weights_norm = weights / weights.max()

    X_train_weighted = X_train_con_final.mul(weights_norm, axis=1)
    X_test_weighted = X_test_con_final.mul(weights_norm, axis=1)

    final_model = train_lgbm(X_train_weighted, y_train)
    metrics_weighted = evaluate(final_model, X_test_weighted, y_test)
    print(f"[3] Feature Weighting      -> {metrics_weighted}")

    # ---- Persist artifacts for the Streamlit app ---------------------------
    joblib.dump(final_model, FINAL_MODEL_PATH)
    joblib.dump(top_features_con, FEATURE_LIST_PATH)
    joblib.dump(weights_norm.to_dict(), SHAP_WEIGHTS_PATH)
    print(f"\nSaved final model    -> {FINAL_MODEL_PATH}")
    print(f"Saved feature list   -> {FEATURE_LIST_PATH}")
    print(f"Saved SHAP weights   -> {SHAP_WEIGHTS_PATH}")

    # ---- Save all 4 configurations' metrics for the comparison chart -------
    all_results = {}
    try:
        with open(RESULTS_JSON) as f:
            all_results = json.load(f)
    except FileNotFoundError:
        pass

    all_results["shap_pipeline"] = {
        "baseline": baseline_metrics,
        "feature_selection": metrics_sel,
        "feature_construction": metrics_con,
        "feature_weighting": metrics_weighted,
    }
    all_results["top_features"] = top_features_con
    all_results["shap_weights"] = weights_norm.to_dict()
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved -> {RESULTS_JSON}")
    print("\nFinal comparison:")
    print(f"{'Config':22s} accuracy  precision  recall  f1")
    for label, m in [("Baseline", baseline_metrics), ("Selection", metrics_sel),
                      ("Construction", metrics_con), ("Weighting", metrics_weighted)]:
        print(f"{label:22s} {m['accuracy']:.4f}   {m['precision']:.4f}    "
              f"{m['recall']:.4f}  {m['f1']:.4f}")


if __name__ == "__main__":
    run()
