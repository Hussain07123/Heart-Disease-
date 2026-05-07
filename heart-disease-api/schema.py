"""
schema.py
---------
Pydantic models for request validation and response formatting.
FastAPI uses these to auto-generate Swagger UI docs at /docs.
"""

from pydantic import BaseModel, Field, validator
from typing import Literal


class PatientInput(BaseModel):
    """
    Input features for heart disease prediction.
    All fields match the Cleveland Heart Disease Dataset columns.
    """

    age: int = Field(
        ..., ge=1, le=120,
        description="Age of the patient in years",
        example=52
    )
    sex: Literal[0, 1] = Field(
        ...,
        description="Sex: 1 = male, 0 = female",
        example=1
    )
    cp: Literal[0, 1, 2, 3] = Field(
        ...,
        description=(
            "Chest pain type: "
            "0 = typical angina, "
            "1 = atypical angina, "
            "2 = non-anginal pain, "
            "3 = asymptomatic"
        ),
        example=0
    )
    trestbps: int = Field(
        ..., ge=80, le=250,
        description="Resting blood pressure (mm Hg)",
        example=125
    )
    chol: int = Field(
        ..., ge=100, le=600,
        description="Serum cholesterol (mg/dl)",
        example=212
    )
    fbs: Literal[0, 1] = Field(
        ...,
        description="Fasting blood sugar > 120 mg/dl: 1 = true, 0 = false",
        example=0
    )
    restecg: Literal[0, 1, 2] = Field(
        ...,
        description=(
            "Resting ECG results: "
            "0 = normal, "
            "1 = ST-T wave abnormality, "
            "2 = left ventricular hypertrophy"
        ),
        example=1
    )
    thalach: int = Field(
        ..., ge=60, le=220,
        description="Maximum heart rate achieved (bpm)",
        example=168
    )
    exang: Literal[0, 1] = Field(
        ...,
        description="Exercise induced angina: 1 = yes, 0 = no",
        example=0
    )
    oldpeak: float = Field(
        ..., ge=0.0, le=10.0,
        description="ST depression induced by exercise relative to rest",
        example=1.0
    )
    slope: Literal[0, 1, 2] = Field(
        ...,
        description=(
            "Slope of peak exercise ST segment: "
            "0 = upsloping, 1 = flat, 2 = downsloping"
        ),
        example=2
    )
    ca: Literal[0, 1, 2, 3] = Field(
        ...,
        description="Number of major vessels (0–3) colored by fluoroscopy",
        example=0
    )
    thal: Literal[0, 1, 2] = Field(
        ...,
        description=(
            "Thalassemia: "
            "0 = normal, "
            "1 = fixed defect, "
            "2 = reversible defect"
        ),
        example=2
    )

    class Config:
        json_schema_extra = {
            "example": {
                "age": 52,
                "sex": 1,
                "cp": 0,
                "trestbps": 125,
                "chol": 212,
                "fbs": 0,
                "restecg": 1,
                "thalach": 168,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 0,
                "thal": 2,
            }
        }


class PredictionResponse(BaseModel):
    """API response with prediction result and confidence."""

    prediction: int = Field(
        ...,
        description="Prediction: 1 = heart disease likely, 0 = no heart disease"
    )
    probability: float = Field(
        ...,
        description="Confidence score (0.0 – 1.0) for heart disease"
    )
    risk_level: str = Field(
        ...,
        description="Risk category: Low / Moderate / High"
    )
    message: str = Field(
        ...,
        description="Human-readable summary of the result"
    )


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""
    status: str
    model_loaded: bool
    version: str
