"""
train_baselines.py
-------------------
Trains and compares three baseline classifiers on the SMOTE-balanced
training set, evaluated against the untouched, imbalanced test set
(the realistic deployment scenario):

  - Random Forest
  - XGBoost
  - LightGBM

Prints a markdown comparison table and saves the LightGBM baseline model
(it wins on every metric in the source paper, and is used as the backbone
for the SHAP-based feature engineering stage).

Run:
    python src/preprocess.py     # must run first
    python src/train_baselines.py
"""

import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import (
    PROCESSED_TRAIN, PROCESSED_TEST, TARGET_COL, RANDOM_STATE,
    BASELINE_MODEL_PATH, RESULTS_JSON,
)


def load_processed():
    train = pd.read_csv(PROCESSED_TRAIN)
    test = pd.read_csv(PROCESSED_TEST)
    X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
    X_test, y_test = test.drop(columns=[TARGET_COL]), test[TARGET_COL]
    return X_train, y_train, X_test, y_test


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    print(f"{name:15s} | acc={metrics['accuracy']:.4f}  prec={metrics['precision']:.4f}  "
          f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}")
    return metrics


def run():
    X_train, y_train, X_test, y_test = load_processed()

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=-1, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        results[name] = evaluate(name, model, X_test, y_test)

    # Markdown table for the README / portfolio writeup
    md_lines = ["| Model | Accuracy | Precision | Recall | F1-score |",
                "|---|---|---|---|---|"]
    for name, m in results.items():
        md_lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} |"
        )
    md_table = "\n".join(md_lines)
    print("\n" + md_table)

    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest model by F1-score: {best_name}")

    joblib.dump(trained_models["LightGBM"], BASELINE_MODEL_PATH)
    print(f"Saved LightGBM baseline -> {BASELINE_MODEL_PATH}")

    all_results = {}
    try:
        with open(RESULTS_JSON) as f:
            all_results = json.load(f)
    except FileNotFoundError:
        pass
    all_results["baseline_comparison"] = results
    all_results["baseline_markdown_table"] = md_table
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    run()
