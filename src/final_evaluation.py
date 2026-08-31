"""
final_evaluation.py
--------------------
Loads outputs/results.json (written by train_baselines.py and
shap_engineering.py) and renders the 4-configuration comparison chart
(Baseline vs. Feature Selection vs. Feature Construction vs. Feature
Weighting) plus an AUC sanity-check bar, which is what actually flags
the near-random-signal issue described in the README.

Run (after preprocess.py, train_baselines.py, shap_engineering.py):
    python src/final_evaluation.py
"""

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import RESULTS_JSON, OUTPUTS_DIR, BASELINE_MODEL_PATH, FINAL_MODEL_PATH
from config import PROCESSED_TEST, TARGET_COL, FEATURE_LIST_PATH, SHAP_WEIGHTS_PATH
import os


def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)


def plot_metric_comparison(pipeline_results: dict, out_path: str):
    configs = ["baseline", "feature_selection", "feature_construction", "feature_weighting"]
    labels = ["Baseline", "Selection", "Construction", "Weighting"]
    metrics = ["accuracy", "precision", "recall", "f1"]

    df = pd.DataFrame(
        {m: [pipeline_results[c][m] for c in configs] for m in metrics}, index=labels
    )

    ax = df.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("LightGBM performance across SHAP-based feature engineering stages")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart -> {out_path}")


def compute_auc_check(out_path: str):
    """A quick AUC sanity check on held-out test data. This is the single
    most important diagnostic in this project: if AUC ~ 0.50, the model has
    no real discriminative power regardless of what accuracy/precision say
    under class imbalance. See README 'Critical Note' section."""
    test = pd.read_csv(PROCESSED_TEST)
    X_test, y_test = test.drop(columns=[TARGET_COL]), test[TARGET_COL]

    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])

    final_model = joblib.load(FINAL_MODEL_PATH)
    top_features = joblib.load(FEATURE_LIST_PATH)
    weights = joblib.load(SHAP_WEIGHTS_PATH)

    X_test_with_interaction = X_test.copy()
    X_test_with_interaction["Chronic_Severity"] = (
        X_test_with_interaction["Chronic_Diseases"] * X_test_with_interaction["Symptom_Severity"]
    )
    X_test_final = X_test_with_interaction[top_features].mul(pd.Series(weights), axis=1)
    final_auc = roc_auc_score(y_test, final_model.predict_proba(X_test_final)[:, 1])

    aucs = pd.Series({"Baseline": baseline_auc, "Final (weighted)": final_auc})
    ax = aucs.plot(kind="bar", figsize=(5, 4), color=["#888888", "#4C72B0"], rot=0)
    ax.axhline(0.5, color="red", linestyle="--", label="Random guessing (AUC=0.50)")
    ax.set_ylim(0, 1.0)
    ax.set_title("ROC-AUC: is there real signal?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved AUC sanity-check chart -> {out_path}")
    print(f"Baseline AUC: {baseline_auc:.4f} | Final AUC: {final_auc:.4f}")
    return {"baseline_auc": baseline_auc, "final_auc": final_auc}


def run():
    results = load_results()
    chart_path = os.path.join(OUTPUTS_DIR, "config_comparison.png")
    plot_metric_comparison(results["shap_pipeline"], chart_path)

    auc_chart_path = os.path.join(OUTPUTS_DIR, "auc_sanity_check.png")
    auc_results = compute_auc_check(auc_chart_path)

    results["auc_sanity_check"] = auc_results
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
