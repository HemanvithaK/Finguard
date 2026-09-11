# src/agents/tools.py
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))

from entities import generate_users


def _load_data():
    """Load transactions and user profiles once."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "transactions.csv", parse_dates=["timestamp"])
    users = generate_users(n_users=5000)
    users_df = pd.DataFrame(users)
    return df, users_df


# Cache so we don't reload on every tool call
_DATA = None

def _get_data():
    global _DATA
    if _DATA is None:
        _DATA = _load_data()
    return _DATA


def get_user_history(user_id: str) -> dict:
    """
    Retrieves a user's recent transaction history and behavioral profile.
    A real system would query a database; we query our synthetic dataset.
    """
    df, users_df = _get_data()
    user_txns = df[df["user_id"] == user_id].sort_values("timestamp", ascending=False)

    if user_txns.empty:
        return {"error": f"No transactions found for user {user_id}"}

    user_profile = users_df[users_df["user_id"] == user_id].iloc[0]

    recent_txns = user_txns.head(10)[["amount", "merchant_id", "timestamp", "device_id", "is_fraud"]].to_dict("records")

    return {
        "user_id": user_id,
        "avg_txn_amount": float(user_profile["avg_txn_amount"]),
        "home_lat": float(user_profile["home_lat"]),
        "home_lon": float(user_profile["home_lon"]),
        "preferred_categories": user_profile["preferred_categories"],
        "total_transactions": len(user_txns),
        "fraud_history": int(user_txns["is_fraud"].sum()),
        "recent_transactions": recent_txns,
    }


def get_merchant_info(merchant_id: str) -> dict:
    """
    Retrieves merchant profile information.
    """
    df, _ = _get_data()
    merchant_txns = df[df["merchant_id"] == merchant_id]

    if merchant_txns.empty:
        return {"error": f"No transactions found for merchant {merchant_id}"}

    return {
        "merchant_id": merchant_id,
        "total_transactions": len(merchant_txns),
        "fraud_rate": float(merchant_txns["is_fraud"].mean()),
        "avg_transaction_amount": float(merchant_txns["amount"].mean()),
        "unique_customers": int(merchant_txns["user_id"].nunique()),
    }


def get_transaction_details(transaction_id: str) -> dict:
    """
    Retrieves full details of a specific transaction.
    """
    df, users_df = _get_data()
    txn = df[df["transaction_id"] == transaction_id]

    if txn.empty:
        return {"error": f"Transaction {transaction_id} not found"}

    txn = txn.iloc[0]
    user_profile = users_df[users_df["user_id"] == txn["user_id"]].iloc[0]

    # Compute some risk signals inline
    amount_ratio = txn["amount"] / user_profile["avg_txn_amount"]

    from features import haversine_distance
    sys.path.append(str(PROJECT_ROOT / "src" / "models"))
    dist_from_home = haversine_distance(
        txn["lat"], txn["lon"],
        float(user_profile["home_lat"]), float(user_profile["home_lon"])
    )

    return {
        "transaction_id": transaction_id,
        "user_id": txn["user_id"],
        "merchant_id": txn["merchant_id"],
        "amount": float(txn["amount"]),
        "timestamp": str(txn["timestamp"]),
        "lat": float(txn["lat"]),
        "lon": float(txn["lon"]),
        "device_id": txn["device_id"],
        "is_known_device": txn["device_id"] == f"D_{txn['user_id']}_primary",
        "amount_vs_avg_ratio": round(amount_ratio, 2),
        "distance_from_home_km": round(float(dist_from_home), 2),
        "model_fraud_probability": None,  # filled in by the caller
    }