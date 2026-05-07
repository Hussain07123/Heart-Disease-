"""
main.py
-------
Heart Disease Prediction API built with FastAPI + XGBoost.

Endpoints:
    GET  /           → Welcome message
    GET  /health     → Model health check
    POST /predict    → Single patient prediction
    POST /predict/batch → Batch predictions (list of patients)

Run locally:
    uvicorn main:app --reload

Swagger UI:
    http://127.0.0.1:8000/docs

ReDoc:
    http://127.0.0.1:8000/redoc
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from schema import PatientInput, PredictionResponse, HealthResponse

# ── Constants ──────────────────────────────────────────────────────────────────
VERSION     = "1.0.0"
MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES    = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ── Load model artifacts ───────────────────────────────────────────────────────
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            "Model artifacts not found. Please run: python train_model.py"
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_artifacts()
    MODEL_LOADED = True
except RuntimeError as e:
    print(f"⚠️  Warning: {e}")
    model, scaler = None, None
    MODEL_LOADED = False

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "A machine learning REST API that predicts the likelihood of heart disease "
        "based on patient clinical data.\n\n"
        "**Model:** XGBoost Classifier\n"
        "**Dataset:** Cleveland Heart Disease Dataset\n"
        "**Author:** Hussain07123\n"
        "**GitHub:** https://github.com/Hussain07123"
    ),
    version=VERSION,
    contact={
        "name": "Hussain",
        "url": "https://github.com/Hussain07123",
    },
    license_info={
        "name": "MIT",
    },
)

# ── CORS middleware (allows frontend apps to call this API) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ─────────────────────────────────────────────────────────────────────
def get_risk_level(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Moderate"
    else:
        return "High"


def get_message(prediction: int, probability: float) -> str:
    risk = get_risk_level(probability)
    if prediction == 0:
        return (
            f"✅ Low likelihood of heart disease detected. "
            f"Risk level: {risk} ({probability:.1%} confidence). "
            "Maintain a healthy lifestyle and schedule regular checkups."
        )
    else:
        return (
            f"⚠️ Heart disease may be present. "
            f"Risk level: {risk} ({probability:.1%} confidence). "
            "Please consult a qualified cardiologist for a full evaluation."
        )


def run_prediction(patient: PatientInput) -> PredictionResponse:
    """Core prediction logic shared by single and batch endpoints."""
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py first."
        )

    # Build feature array in correct column order
    input_df = pd.DataFrame(
        [[getattr(patient, f) for f in FEATURES]],
        columns=FEATURES
    )
    input_scaled  = scaler.transform(input_df)
    prediction    = int(model.predict(input_scaled)[0])
    probability   = float(model.predict_proba(input_scaled)[0][1])
    risk_level    = get_risk_level(probability)
    message       = get_message(prediction, probability)

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=risk_level,
        message=message,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """Welcome endpoint."""
    return {
        "message": "❤️ Heart Disease Prediction API",
        "version": VERSION,
        "docs": "/docs",
        "author": "Hussain07123",
        "github": "https://github.com/Hussain07123",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Check if the API and model are running correctly."""
    return HealthResponse(
        status="ok" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED,
        version=VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput):
    """
    Predict heart disease risk for a single patient.

    Submit patient clinical data and receive:
    - **prediction**: 1 (disease likely) or 0 (no disease)
    - **probability**: confidence score (0.0 – 1.0)
    - **risk_level**: Low / Moderate / High
    - **message**: human-readable summary
    """
    return run_prediction(patient)


@app.post(
    "/predict/batch",
    response_model=List[PredictionResponse],
    tags=["Prediction"]
)
def predict_batch(patients: List[PatientInput]):
    """
    Predict heart disease risk for multiple patients at once.

    Submit a list of patient records and receive a prediction for each.
    Maximum 100 patients per request.
    """
    if len(patients) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 patients per batch request."
        )
    return [run_prediction(p) for p in patients]
