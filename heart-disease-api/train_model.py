"""
train_model.py
--------------
Trains an XGBoost classifier on the Heart Disease dataset,
evaluates it, and saves the trained model + scaler to disk.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/heart.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# ── Feature columns (matches schema.py) ───────────────────────────────────────
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"


def load_data(path: str) -> pd.DataFrame:
    """Load and do basic validation on the dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run: python data/generate_data.py  (or download the real dataset from Kaggle)"
        )
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    missing = df[FEATURES + [TARGET]].isnull().sum()
    if missing.any():
        print(f"⚠️  Missing values found:\n{missing[missing > 0]}")
        df = df.dropna(subset=FEATURES + [TARGET])
        print(f"   Dropped rows with nulls → {df.shape[0]} rows remaining")

    return df


def train(df: pd.DataFrame):
    """Full training pipeline: split → scale → train → evaluate → save."""

    X = df[FEATURES]
    y = df[TARGET]

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ── Feature scaling ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )
    print("✅ Model trained successfully")

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n📈 Accuracy : {acc:.4f}")
    print(f"📈 ROC-AUC  : {roc_auc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
    print("🔲 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n💾 Model  saved → {MODEL_PATH}")
    print(f"💾 Scaler saved → {SCALER_PATH}")

    return model, scaler


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    train(df)
    print("\n🚀 Training complete! Run the API with: uvicorn main:app --reload")
