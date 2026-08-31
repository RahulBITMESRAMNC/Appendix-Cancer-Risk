"""
preprocess.py
-------------
Loads the raw Appendix Cancer dataset and reproduces the preprocessing
pipeline described in the paper:

  1. Drop uninformative columns (Patient_ID).
  2. Label-encode categorical columns.
  3. Stratified 80:20 train/test split (preserves the ~15.1% positive rate).
  4. SMOTE applied to the TRAINING SET ONLY, to avoid leaking synthetic
     minority-class information into the held-out test set.

Run:
    python src/preprocess.py
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from config import (
    RAW_CSV, PROCESSED_TRAIN, PROCESSED_TEST, ENCODERS_PATH,
    TARGET_COL, ID_COL, CATEGORICAL_COLS, RANDOM_STATE, TEST_SIZE,
)


def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def drop_uninformative(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in [ID_COL] if c in df.columns]
    return df.drop(columns=cols_to_drop)


def encode_categoricals(df: pd.DataFrame):
    """Label-encode categorical columns (including the target) and return
    the encoders so the same mapping can be reused at inference time."""
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Encode the binary target (Yes/No -> 1/0) separately and keep the mapping explicit.
    target_le = LabelEncoder()
    df[TARGET_COL] = target_le.fit_transform(df[TARGET_COL].astype(str))
    encoders[TARGET_COL] = target_le

    return df, encoders


def stratified_split(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]:,} rows | positive rate = {y_train.mean():.4f}")
    print(f"Test:  {X_test.shape[0]:,} rows | positive rate = {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """SMOTE is fit exclusively on the training split so no synthetic
    signal (or leaked information) reaches the untouched test set."""
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_res.shape[0]:,} rows | positive rate = {y_res.mean():.4f}")
    return X_res, y_res


def run():
    df = load_raw()
    df = drop_uninformative(df)
    df_encoded, encoders = encode_categoricals(df)

    X_train, X_test, y_train, y_test = stratified_split(df_encoded)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    train_out = X_train_res.copy()
    train_out[TARGET_COL] = y_train_res
    test_out = X_test.copy()
    test_out[TARGET_COL] = y_test

    train_out.to_csv(PROCESSED_TRAIN, index=False)
    test_out.to_csv(PROCESSED_TEST, index=False)
    joblib.dump(encoders, ENCODERS_PATH)

    print(f"\nSaved processed train -> {PROCESSED_TRAIN}")
    print(f"Saved processed test  -> {PROCESSED_TEST}")
    print(f"Saved encoders        -> {ENCODERS_PATH}")


if __name__ == "__main__":
    run()
