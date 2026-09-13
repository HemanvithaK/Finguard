# src/api/app.py
import os
import sys
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))
sys.path.append(str(PROJECT_ROOT / "src" / "agents"))

import lightgbm as lgb
import pandas as pd
import numpy as np
from entities import generate_users
from features import engineer_features, haversine_distance
from train_baseline import FEATURE_COLS

app = FastAPI(
    title="FinGuard API",
    description="Fraud detection and investigation API",
    version="1.0.0",
)

# ---- Load model once at startup, not per request ----
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "lgbm_baseline.txt"
booster = lgb.Booster(model_file=str(MODEL_PATH))

# Load user profiles once
users = generate_users(n_users=5000)
users_df = pd.DataFrame(users)


# ---- Request/response schemas ----
class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    timestamp: str
    lat: float
    lon: float
    device_id: str


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_flagged: bool
    latency_ms: float


class InvestigationRequest(BaseModel):
    transaction_id: str
    fraud_probability: float


class InvestigationResponse(BaseModel):
    transaction_id: str
    evidence_summary: str
    dispute_reasoning: str
    decision: str
    confidence: str
    latency_ms: float


# ---- Helper: compute features for a single transaction ----
def compute_single_txn_features(txn: TransactionRequest) -> dict:
    """
    Computes the same features LightGBM was trained on, for a single
    incoming transaction. In production, some of these (like time_since_prev_txn)
    would come from a streaming feature store; here we compute defaults.
    """
    user = users_df[users_df["user_id"] == txn.user_id]
    if user.empty:
        raise HTTPException(status_code=404, detail=f"User {txn.user_id} not found")

    user = user.iloc[0]

    dist = haversine_distance(
        txn.lat, txn.lon,
        float(user["home_lat"]), float(user["home_lon"])
    )

    is_known = 1 if txn.device_id == f"D_{txn.user_id}_primary" else 0
    amount_ratio = txn.amount / user["avg_txn_amount"]
    hour = pd.to_datetime(txn.timestamp).hour

    features = {
        "amount": txn.amount,
        "time_since_prev_txn": 86400.0,  # default: assume ~1 day since last txn
        "txns_last_10min": 1.0,          # default: just this transaction
        "dist_from_home_km": float(dist),
        "amount_vs_avg_ratio": amount_ratio,
        "is_known_device": is_known,
        "distinct_merchants_1hr": 1.0,   # default: just this merchant
        "hour_of_day": hour,
    }
    return features


# ---- Endpoints ----
@app.get("/health")
def health_check():
    """Simple health check — used by load balancers and monitoring."""
    return {"status": "healthy", "model_loaded": booster is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(txn: TransactionRequest):
    """
    Real-time fraud scoring: takes a transaction, returns fraud probability.
    Designed to run on every transaction — must be fast (no LLM calls).
    """
    start = time.time()

    features = compute_single_txn_features(txn)
    feature_df = pd.DataFrame([features])[FEATURE_COLS]
    prob = float(booster.predict(feature_df)[0])

    latency = (time.time() - start) * 1000

    return PredictionResponse(
        transaction_id=txn.transaction_id,
        fraud_probability=round(prob, 6),
        is_flagged=prob > 0.5,
        latency_ms=round(latency, 2),
    )


@app.post("/investigate", response_model=InvestigationResponse)
def investigate_fraud(req: InvestigationRequest):
    """
    Deep investigation: runs the LangGraph agent on a flagged transaction.
    Only called for flagged transactions — slower, uses LLM calls.
    """
    start = time.time()

    from investigation_agent import investigate_transaction
    result = investigate_transaction(req.transaction_id, req.fraud_probability)

    latency = (time.time() - start) * 1000

    return InvestigationResponse(
        transaction_id=req.transaction_id,
        evidence_summary=result["evidence_summary"],
        dispute_reasoning=result["dispute_reasoning"],
        decision=result["decision"],
        confidence=result["confidence"],
        latency_ms=round(latency, 2),
    )