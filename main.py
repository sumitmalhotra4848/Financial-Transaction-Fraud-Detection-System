"""
main.py — Fraud Risk API

A FastAPI service around the hybrid Isolation Forest + Autoencoder fraud risk
scoring framework built in financial_transaction_fraud_detection.ipynb. Loads the
real trained artifacts from saved_models/ (produced by the notebook's Section 19).

Run:
    uvicorn main:app --reload

Interactive API docs:
    http://127.0.0.1:8000/docs

Configuration (all optional, via environment variables):
    MODELS_DIR             default: "saved_models"
    CREDITCARD_CSV         default: auto-detected (kagglehub cache or cwd)
    MODEL_COMPARISON_CSV   default: "model_comparison.csv"
    HIGH_RISK_CSV          default: "high_risk_transactions.csv"
"""

import io
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import scoring

MODELS_DIR = os.environ.get("MODELS_DIR", "saved_models")
CREDITCARD_CSV = os.environ.get("CREDITCARD_CSV") or scoring.find_local_creditcard_csv()
MODEL_COMPARISON_CSV = os.environ.get("MODEL_COMPARISON_CSV", "model_comparison.csv")
HIGH_RISK_CSV = os.environ.get("HIGH_RISK_CSV", "high_risk_transactions.csv")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

state: dict = {
    "bundle": None,
    "load_error": None,
    "reference_df": None,
    "amount_high_cutoff": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["bundle"] = scoring.load_models(MODELS_DIR)
        state["load_error"] = None
    except scoring.ModelLoadError as e:
        state["bundle"] = None
        state["load_error"] = str(e)

    if CREDITCARD_CSV and os.path.exists(CREDITCARD_CSV):
        try:
            ref = pd.read_csv(CREDITCARD_CSV)
            state["reference_df"] = ref
            state["amount_high_cutoff"] = float(ref["Amount"].quantile(0.99)) if "Amount" in ref.columns else None
        except Exception:
            state["reference_df"] = None

    yield
    state.clear()


app = FastAPI(
    title="Fraud Risk API",
    description="Hybrid Isolation Forest + Autoencoder fraud risk scoring, served as a REST API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0)


class ScoreResult(BaseModel):
    isolation_forest_score: float
    autoencoder_score: float
    hybrid_risk_score: float
    risk_level: str
    decision: str
    explanation: List[str]


class BatchSummary(BaseModel):
    total: int
    flagged_fraud: int
    high_risk: int
    medium_risk: int
    low_risk: int
    average_risk_score: float
    results: List[dict]


class ConfigOut(BaseModel):
    iso_weight: float
    ae_weight: float
    adaptive_threshold: float
    risk_percentile: int
    feature_columns: List[str]
    models_loaded: bool
    reference_dataset_loaded: bool
    reference_dataset_rows: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_bundle():
    if state["bundle"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Models are not loaded: {state['load_error']}",
        )
    return state["bundle"]


def build_result(bundle: dict, row: pd.Series) -> ScoreResult:
    cfg = bundle["config"]
    return ScoreResult(
        isolation_forest_score=float(row["isolation_forest_score"]),
        autoencoder_score=float(row["autoencoder_score"]),
        hybrid_risk_score=float(row["hybrid_risk_score"]),
        risk_level=row["risk_level"],
        decision=row["decision"],
        explanation=scoring.explain_row(row, state["amount_high_cutoff"], cfg["ADAPTIVE_THRESHOLD"]),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Fraud Risk API is running. See /docs for the API, or add static/index.html for a UI."}


@app.get("/health")
def health():
    return {
        "status": "ok" if state["bundle"] is not None else "degraded",
        "models_loaded": state["bundle"] is not None,
        "load_error": state["load_error"],
        "reference_dataset_loaded": state["reference_df"] is not None,
    }


@app.get("/api/config", response_model=ConfigOut)
def get_config():
    bundle = require_bundle()
    cfg = bundle["config"]
    return ConfigOut(
        iso_weight=cfg["ISO_WEIGHT"],
        ae_weight=cfg["AE_WEIGHT"],
        adaptive_threshold=cfg["ADAPTIVE_THRESHOLD"],
        risk_percentile=cfg["RISK_PERCENTILE"],
        feature_columns=cfg["FEATURE_COLUMNS"],
        models_loaded=True,
        reference_dataset_loaded=state["reference_df"] is not None,
        reference_dataset_rows=(len(state["reference_df"]) if state["reference_df"] is not None else None),
    )


@app.post("/api/score", response_model=ScoreResult)
def score_transaction(transaction: Transaction):
    bundle = require_bundle()
    df = pd.DataFrame([transaction.model_dump()])
    try:
        scored = scoring.score_dataframe(bundle, df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return build_result(bundle, scored.iloc[0])


@app.post("/api/score/batch")
async def score_batch(
    file: UploadFile = File(..., description="CSV with columns Time, V1-V28, Amount"),
    format: str = Query("json", enum=["json", "csv"]),
):
    bundle = require_bundle()
    raw_bytes = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    try:
        scored_df = scoring.score_dataframe(bundle, raw_df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if format == "csv":
        buf = io.StringIO()
        scored_df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=scored_transactions.csv"},
        )

    counts = scored_df["risk_level"].value_counts()
    summary = BatchSummary(
        total=len(scored_df),
        flagged_fraud=int((scored_df["decision"] == "FRAUD").sum()),
        high_risk=int(counts.get("High Risk", 0)),
        medium_risk=int(counts.get("Medium Risk", 0)),
        low_risk=int(counts.get("Low Risk", 0)),
        average_risk_score=float(scored_df["hybrid_risk_score"].mean()),
        results=scored_df.to_dict(orient="records"),
    )
    return summary


@app.get("/api/sample")
def get_sample(fraud: Optional[bool] = Query(None, description="true = known fraud, false = known legitimate, omit = any")):
    if state["reference_df"] is None:
        raise HTTPException(
            status_code=404,
            detail="No reference dataset loaded. Set the CREDITCARD_CSV environment variable to a creditcard.csv path and restart the server.",
        )
    ref = state["reference_df"]
    pool = ref
    if fraud is not None and "Class" in ref.columns:
        pool = ref[ref["Class"] == (1 if fraud else 0)]
    if len(pool) == 0:
        raise HTTPException(status_code=404, detail="No matching rows in the reference dataset.")
    sampled = pool.sample(1).iloc[0]
    bundle = require_bundle()
    feature_cols = bundle["config"]["FEATURE_COLUMNS"]
    result = {c: float(sampled[c]) for c in feature_cols}
    if "Class" in ref.columns:
        result["_actual_class"] = "Fraud" if sampled["Class"] == 1 else "Legitimate"
    return result


@app.get("/api/model-comparison")
def get_model_comparison():
    if not os.path.exists(MODEL_COMPARISON_CSV):
        raise HTTPException(
            status_code=404,
            detail=f"'{MODEL_COMPARISON_CSV}' not found. Re-run notebook Section 13 to produce it.",
        )
    df = pd.read_csv(MODEL_COMPARISON_CSV)
    return df.to_dict(orient="records")


@app.get("/api/high-risk")
def get_high_risk(limit: int = Query(50, ge=1, le=1000)):
    if not os.path.exists(HIGH_RISK_CSV):
        raise HTTPException(
            status_code=404,
            detail=f"'{HIGH_RISK_CSV}' not found. Re-run notebook Section 17 to produce it.",
        )
    df = pd.read_csv(HIGH_RISK_CSV)
    if "Hybrid Risk Score" in df.columns:
        df = df.sort_values("Hybrid Risk Score", ascending=False)
    return df.head(limit).to_dict(orient="records")
