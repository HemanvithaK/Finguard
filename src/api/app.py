# src/api/app.py
import os
import sys
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))
sys.path.append(str(PROJECT_ROOT / "src" / "agents"))

import lightgbm as lgb
import torch
import pandas as pd
import numpy as np
from entities import generate_users
from features import haversine_distance
from train_baseline import FEATURE_COLS
from src.api.feature_store import FeatureStore
from src.api.drift_detector import DriftDetector
from src.api.ab_router import ABRouter
from sequence_data import TransactionSequenceDataset, SEQ_FEATURES, MAX_SEQ_LEN
from transformer_model import FraudSequenceTransformer

app = FastAPI(
    title="FinGuard API",
    description="Fraud detection API with feature store, drift monitoring, and A/B testing",
    version="4.0.0",
)

# ---- Load models at startup ----
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "lgbm_baseline.txt"
booster = lgb.Booster(model_file=str(MODEL_PATH))

TRANSFORMER_PATH = PROJECT_ROOT / "data" / "processed" / "transformer_v2.pt"
transformer = FraudSequenceTransformer(n_features=len(SEQ_FEATURES))
if TRANSFORMER_PATH.exists():
    transformer.load_state_dict(torch.load(str(TRANSFORMER_PATH), map_location="cpu", weights_only=True))
    transformer.eval()
    transformer_loaded = True
    print("Transformer model loaded for A/B testing")
else:
    transformer_loaded = False
    print("Transformer model not found — A/B testing will use LightGBM only")

# ---- Load user profiles ----
users = generate_users(n_users=5000)
users_df = pd.DataFrame(users)

# ---- Initialize feature store, drift detector, A/B router ----
feature_store = FeatureStore()

BASELINE_PATH = PROJECT_ROOT / "data" / "processed" / "drift_baseline.json"
drift_detector = DriftDetector(
    feature_names=FEATURE_COLS,
    baseline_path=str(BASELINE_PATH) if BASELINE_PATH.exists() else None,
)

ab_router = ABRouter(model_a_name="LightGBM", model_b_name="Transformer", b_fraction=0.2)


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
    feature_source: str
    model_used: str  # which model handled this request


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


# ---- Helper: predict with transformer ----
def predict_transformer(features: dict) -> float:
    """
    Run the transformer on a single transaction.
    Creates a minimal sequence (just this transaction) since we don't
    have the full user history loaded in the API. In production, the
    feature store would provide the recent sequence.
    """
    feature_values = [features.get(f, 0.0) for f in SEQ_FEATURES]
    # Pad to MAX_SEQ_LEN with zeros (same as training)
    sequence = np.zeros((MAX_SEQ_LEN, len(SEQ_FEATURES)), dtype=np.float32)
    sequence[-1] = feature_values  # current transaction at last position

    tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logit = transformer(tensor)
        prob = torch.sigmoid(logit).item()
    return prob


# ---- Endpoints ----
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "lgbm_loaded": booster is not None,
        "transformer_loaded": transformer_loaded,
        "ab_split": f"{int((1-ab_router.b_fraction)*100)}/{int(ab_router.b_fraction*100)}",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(txn: TransactionRequest):
    """Real-time fraud scoring with A/B routing, live features, and drift monitoring."""
    start = time.time()

    user = users_df[users_df["user_id"] == txn.user_id]
    if user.empty:
        raise HTTPException(status_code=404, detail=f"User {txn.user_id} not found")
    user = user.iloc[0]

    # Static features
    dist = haversine_distance(
        txn.lat, txn.lon,
        float(user["home_lat"]), float(user["home_lon"])
    )
    is_known = 1 if txn.device_id == f"D_{txn.user_id}_primary" else 0
    amount_ratio = txn.amount / user["avg_txn_amount"]
    hour = pd.to_datetime(txn.timestamp).hour

    # Live velocity features
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

    # A/B routing: decide which model scores this transaction
    model_used = ab_router.route()

    if model_used == "Transformer" and transformer_loaded:
        prob = predict_transformer(features)
    else:
        model_used = "LightGBM"  # fallback if transformer not loaded
        feature_df = pd.DataFrame([features])[FEATURE_COLS]
        prob = float(booster.predict(feature_df)[0])

    is_flagged = prob > 0.5

    # Record in feature store
    feature_store.record_transaction({
        "transaction_id": txn.transaction_id,
        "user_id": txn.user_id,
        "merchant_id": txn.merchant_id,
        "amount": txn.amount,
        "timestamp": txn.timestamp,
        "device_id": txn.device_id,
    })

    # Record for drift monitoring
    drift_detector.record(features, prob)

    latency = (time.time() - start) * 1000

    # Record for A/B comparison
    ab_router.record(model_used, prob, latency, is_flagged)

    return PredictionResponse(
        transaction_id=txn.transaction_id,
        fraud_probability=round(prob, 6),
        is_flagged=is_flagged,
        latency_ms=round(latency, 2),
        feature_source=feature_source,
        model_used=model_used,
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


@app.get("/drift")
def check_drift():
    """Per-feature PSI scores and prediction distribution monitoring."""
    return drift_detector.check_drift()


@app.get("/ab-results")
def ab_results():
    """
    Live A/B test comparison — flag rates, latency stats, and prediction
    distributions for both models. Call this to decide whether the
    challenger model (Transformer) is ready to replace the champion (LightGBM).
    """
    return ab_router.get_results()