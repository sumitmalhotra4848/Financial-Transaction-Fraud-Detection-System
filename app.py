from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class Transaction(BaseModel):
    Time: float = Field(..., description="Transaction time")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


class PredictionResponse(BaseModel):
    isolation_forest_score: float
    autoencoder_score: float
    hybrid_risk_score: float
    risk_level: str
    prediction: str
    explanation: List[str]


class Models:
    iso = None
    ae = None
    scaler = None
    config = None


models = Models()

app = FastAPI(
    title="Financial Transaction Fraud Detection API",
    description=(
        "FastAPI backend for the Isolation Forest + Autoencoder "
        "hybrid fraud detection model."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models():
    required = [
        MODEL_DIR / "isolation_forest.pkl",
        MODEL_DIR / "autoencoder.keras",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "risk_configuration.pkl",
    ]

    missing = [p.name for p in required if not p.exists()]
    if missing:
        # Keep the API bootable so /health can explain the deployment problem.
        print("WARNING: Missing model artifacts:", ", ".join(missing))
        return

    models.iso = joblib.load(MODEL_DIR / "isolation_forest.pkl")
    models.ae = keras.models.load_model(MODEL_DIR / "autoencoder.keras")
    models.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    models.config = joblib.load(MODEL_DIR / "risk_configuration.pkl")


def require_models():
    if any(
        x is None
        for x in [models.iso, models.ae, models.scaler, models.config]
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts are not loaded. Add isolation_forest.pkl, "
                "autoencoder.keras, scaler.pkl and risk_configuration.pkl "
                "inside saved_models/."
            ),
        )


def normalize_score(raw, low, high):
    denom = (high - low) if (high - low) > 1e-12 else 1e-12
    return np.clip((raw - low) / denom, 0, 1)


def reconstruction_error(model, X_scaled):
    reconstructed = model.predict(X_scaled.values, verbose=0)
    bce = -(
        X_scaled.values * np.log(reconstructed + 1e-8)
        + (1 - X_scaled.values) * np.log(1 - reconstructed + 1e-8)
    )
    return np.mean(bce, axis=1)


def iso_anomaly_score(model, X_scaled):
    # Isolation Forest score_samples: higher = more normal.
    # Negate to make higher = more anomalous.
    return -model.score_samples(X_scaled)


def risk_category(score):
    if score <= 30:
        return "Low Risk"
    if score <= 70:
        return "Medium Risk"
    return "High Risk"


def predict_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    require_models()

    data = df[FEATURES].copy()

    # Matches the notebook: scale only Time and Amount.
    data[["Time", "Amount"]] = models.scaler.transform(
        data[["Time", "Amount"]]
    )

    iso_raw = iso_anomaly_score(models.iso, data)
    iso_norm = normalize_score(
        iso_raw,
        models.config["iso_low"],
        models.config["iso_high"],
    )

    ae_error = reconstruction_error(models.ae, data)
    ae_norm = normalize_score(
        ae_error,
        models.config["ae_low"],
        models.config["ae_high"],
    )

    hybrid = (
        models.config["ISO_WEIGHT"] * iso_norm
        + models.config["AE_WEIGHT"] * ae_norm
    ) * 100

    output = []

    for i in range(len(df)):
        iso_score = float(iso_norm[i])
        ae_score = float(ae_norm[i])
        risk_score = float(hybrid[i])

        category = risk_category(risk_score)
        decision = (
            "FRAUD"
            if risk_score >= models.config["ADAPTIVE_THRESHOLD"]
            else "LEGITIMATE"
        )

        signals = []

        if iso_score > 0.7:
            signals.append("High Isolation Forest anomaly score")

        if ae_score > 0.7:
            signals.append("High Autoencoder reconstruction error")

        if risk_score >= models.config["ADAPTIVE_THRESHOLD"]:
            signals.append(
                "Overall hybrid risk score above adaptive threshold"
            )

        if not signals:
            signals.append("No strong anomaly signals detected")

        output.append(
            {
                "isolation_forest_score": round(iso_score, 4),
                "autoencoder_score": round(ae_score, 4),
                "hybrid_risk_score": round(risk_score, 2),
                "risk_level": category,
                "prediction": decision,
                "explanation": signals,
            }
        )

    return output


@app.get("/")
def root():
    return {
        "message": "Financial Transaction Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    loaded = all(
        x is not None
        for x in [models.iso, models.ae, models.scaler, models.config]
    )

    return {
        "status": "healthy" if loaded else "degraded",
        "models_loaded": loaded,
        "required_features": FEATURES,
    }


@app.get("/model-info")
def model_info():
    require_models()

    return {
        "model": "Isolation Forest + Autoencoder",
        "iso_weight": models.config["ISO_WEIGHT"],
        "autoencoder_weight": models.config["AE_WEIGHT"],
        "adaptive_threshold": models.config["ADAPTIVE_THRESHOLD"],
        "risk_percentile": models.config["RISK_PERCENTILE"],
        "risk_categories": {
            "low": "0-30",
            "medium": "31-70",
            "high": ">70",
        },
        "features": FEATURES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    df = pd.DataFrame([transaction.model_dump()])
    return predict_df(df)[0]


@app.post("/predict/batch")
def predict_batch(transactions: List[Transaction]):
    if not transactions:
        raise HTTPException(
            status_code=400,
            detail="At least one transaction is required.",
        )

    df = pd.DataFrame([t.model_dump() for t in transactions])

    return {
        "count": len(df),
        "predictions": predict_df(df),
    }


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        content = await file.read()
        from io import BytesIO

        df = pd.read_csv(BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {exc}",
        )

    missing = [column for column in FEATURES if column not in df.columns]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "CSV is missing required columns.",
                "missing_columns": missing,
                "required_columns": FEATURES,
            },
        )

    predictions = predict_df(df)

    return {
        "filename": file.filename,
        "count": len(df),
        "predictions": predictions,
    }
