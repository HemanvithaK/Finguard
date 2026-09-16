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
from features import haversine_distance
from train_baseline import FEATURE_COLS
from src.api.feature_store import FeatureStore

app = FastAPI(
    title="FinGuard API",
    description="Fraud detection and investigation API with real-time feature computation",
    version="2.0.0",
)

# ---- Load model and user profiles once at startup ----
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "lgbm_baseline.txt"
booster = lgb.Booster(model_file=str(MODEL_PATH))

users = generate_users(n_users=5000)
users_df = pd.DataFrame(users)

# ---- Initialize feature store ----
feature_store = FeatureStore()


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
    feature_source: str  # "live" or "default" — transparency about feature quality


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


# ---- Prediction endpoint ----
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": booster is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(txn: TransactionRequest):
    """
    Real-time fraud scoring with live velocity features from the feature store.
    After scoring, records the transaction so future predictions for this user
    have accurate velocity context.
    """
    start = time.time()

    user = users_df[users_df["user_id"] == txn.user_id]
    if user.empty:
        raise HTTPException(status_code=404, detail=f"User {txn.user_id} not found")
    user = user.iloc[0]

    # Compute static features (same as before)
    dist = haversine_distance(
        txn.lat, txn.lon,
        float(user["home_lat"]), float(user["home_lon"])
    )
    is_known = 1 if txn.device_id == f"D_{txn.user_id}_primary" else 0
    amount_ratio = txn.amount / user["avg_txn_amount"]
    hour = pd.to_datetime(txn.timestamp).hour

    # Compute velocity features from feature store (NOT hardcoded anymore)
    velocity = feature_store.get_velocity_features(txn.user_id, txn.timestamp)
    feature_source = "live" if velocity["time_since_prev_txn"] != 999999.0 else "default"

    features = {
        "amount": txn.amount,
        "time_since_prev_txn": velocity["time_since_prev_txn"],
        "txns_last_10min": velocity["txns_last_10min"],
        "dist_from_home_km": float(dist),
        "amount_vs_avg_ratio": amount_ratio,
        "is_known_device": is_known,
        "distinct_merchants_1hr": velocity["distinct_merchants_1hr"],
        "hour_of_day": hour,
    }

    feature_df = pd.DataFrame([features])[FEATURE_COLS]
    prob = float(booster.predict(feature_df)[0])

    # Record this transaction AFTER scoring so it's available for
    # future velocity computations (but doesn't influence its own score)
    feature_store.record_transaction({
        "transaction_id": txn.transaction_id,
        "user_id": txn.user_id,
        "merchant_id": txn.merchant_id,
        "amount": txn.amount,
        "timestamp": txn.timestamp,
        "device_id": txn.device_id,
    })

    latency = (time.time() - start) * 1000

    return PredictionResponse(
        transaction_id=txn.transaction_id,
        fraud_probability=round(prob, 6),
        is_flagged=prob > 0.5,
        latency_ms=round(latency, 2),
        feature_source=feature_source,
    )


@app.post("/investigate", response_model=InvestigationResponse)
def investigate_fraud(req: InvestigationRequest):
    """Deep investigation via LangGraph agent."""
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