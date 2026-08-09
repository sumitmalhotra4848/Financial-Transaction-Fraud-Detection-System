"""
scoring.py — the hybrid fraud risk scoring core.

This mirrors the framework built in financial_transaction_fraud_detection.ipynb
(Isolation Forest + Autoencoder -> weighted hybrid score -> adaptive threshold),
with no web-framework dependency, so it can be unit tested or reused outside FastAPI.
"""

import glob
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

TF_AVAILABLE = True
try:
    from tensorflow import keras
except Exception:
    TF_AVAILABLE = False


class ModelLoadError(Exception):
    """Raised when the saved model artifacts cannot be found or loaded."""


def risk_category(score: float) -> str:
    if score <= 30:
        return "Low Risk"
    elif score <= 70:
        return "Medium Risk"
    return "High Risk"


def isoforest_raw_anomaly_score(model, X_scaled):
    return -model.score_samples(X_scaled)


def normalize_score(raw_scores, low, high):
    denom = (high - low) if (high - low) > 1e-12 else 1e-12
    return np.clip((np.asarray(raw_scores) - low) / denom, 0, 1)


def reconstruction_error(model, X_scaled_values):
    reconstructed = model.predict(X_scaled_values, verbose=0)
    return np.mean(np.square(X_scaled_values - reconstructed), axis=1)


def load_models(models_dir: str) -> dict:
    """Loads the four artifacts saved by the notebook's Section 19. Raises
    ModelLoadError with a clear message if anything is missing or fails to load."""
    iso_path = os.path.join(models_dir, "isolation_forest.pkl")
    ae_path = os.path.join(models_dir, "autoencoder.keras")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    config_path = os.path.join(models_dir, "risk_configuration.pkl")

    missing = [p for p in [iso_path, ae_path, scaler_path, config_path] if not os.path.exists(p)]
    if missing:
        raise ModelLoadError(
            f"Missing artifact(s) in '{models_dir}': "
            f"{', '.join(os.path.basename(m) for m in missing)}. "
            f"Run the notebook's Section 19 first to produce them."
        )

    if not TF_AVAILABLE:
        raise ModelLoadError("TensorFlow is not installed. Run: pip install tensorflow-cpu")

    try:
        iso_forest = joblib.load(iso_path)
        autoencoder = keras.models.load_model(ae_path)
        scaler = joblib.load(scaler_path)
        config = joblib.load(config_path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load artifacts from '{models_dir}': {e}") from e

    return {
        "iso_forest": iso_forest,
        "autoencoder": autoencoder,
        "scaler": scaler,
        "config": config,
    }


def score_dataframe(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full hybrid framework on a dataframe of raw transactions
    (Time, V1-V28, Amount) and returns a dataframe of per-transaction scores."""
    cfg = bundle["config"]
    feature_cols = cfg["FEATURE_COLUMNS"]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {missing_cols}")

    X = df[feature_cols].copy()
    scaled = bundle["scaler"].transform(X)
    scaled_df = pd.DataFrame(scaled, columns=feature_cols)

    iso_raw = isoforest_raw_anomaly_score(bundle["iso_forest"], scaled_df)
    iso_norm = normalize_score(iso_raw, cfg["iso_low"], cfg["iso_high"])

    ae_err = reconstruction_error(bundle["autoencoder"], scaled_df.values)
    ae_norm = normalize_score(ae_err, cfg["ae_low"], cfg["ae_high"])

    hybrid = (cfg["ISO_WEIGHT"] * iso_norm + cfg["AE_WEIGHT"] * ae_norm) * 100
    categories = np.array([risk_category(s) for s in hybrid])
    decisions = np.where(hybrid >= cfg["ADAPTIVE_THRESHOLD"], "FRAUD", "LEGITIMATE")

    out = df.copy().reset_index(drop=True)
    out["isolation_forest_score"] = np.round(iso_norm, 4)
    out["autoencoder_score"] = np.round(ae_norm, 4)
    out["hybrid_risk_score"] = np.round(hybrid, 2)
    out["risk_level"] = categories
    out["decision"] = decisions
    return out


def explain_row(row: pd.Series, amount_high_cutoff: float, threshold: float) -> list:
    signals = []
    if row["isolation_forest_score"] > 0.7:
        signals.append(f"High Isolation Forest anomaly score ({row['isolation_forest_score']:.2f})")
    if row["autoencoder_score"] > 0.7:
        signals.append(f"High Autoencoder reconstruction error ({row['autoencoder_score']:.2f})")
    if "Amount" in row and amount_high_cutoff is not None and row["Amount"] > amount_high_cutoff:
        signals.append(f"Unusually high transaction amount (${row['Amount']:.2f})")
    if row["hybrid_risk_score"] >= threshold:
        signals.append(
            f"Hybrid risk score ({row['hybrid_risk_score']:.1f}) above adaptive threshold ({threshold:.1f})"
        )
    if not signals:
        signals.append("No strong individual anomaly signals detected; risk score within normal range")
    return signals


def find_local_creditcard_csv() -> Optional[str]:
    candidates = glob.glob(os.path.expanduser("~/.cache/kagglehub/**/creditcard.csv"), recursive=True)
    candidates += glob.glob("**/creditcard.csv", recursive=True)
    return candidates[0] if candidates else None
