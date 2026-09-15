# src/models/graph_data.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData


def _standardize(t):
    """Zero-mean/unit-variance per feature."""
    mean = t.mean(dim=0, keepdim=True)
    std = t.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (t - mean) / std


def _get_pre_fraud_mask(df):
    """
    For each user, returns a boolean mask that is True for transactions
    that occurred BEFORE that user's first fraud event. For users with
    no fraud, all their transactions are included.
    
    This prevents the data leak where fraud transactions inflate the
    node features of victimized users.
    """
    # Find the timestamp of each user's first fraud transaction
    fraud_txns = df[df["is_fraud"] == 1]
    first_fraud_time = fraud_txns.groupby("user_id")["timestamp"].min()

    # For each transaction, check if it's before the user's first fraud
    mask = pd.Series(True, index=df.index)
    for user_id, fraud_time in first_fraud_time.items():
        user_mask = df["user_id"] == user_id
        mask[user_mask & (df["timestamp"] >= fraud_time)] = False

    return mask


def construct_graph(transactions_df):
    """
    Heterogeneous graph:
      nodes: card (user), merchant, device
      edges: (card->merchant) transaction, (card->device) used_by, + reverse edges
    
    IMPORTANT: Node features are computed ONLY from pre-fraud transactions
    to avoid data leakage. Labels use all transactions (a user is labeled
    as fraud if they have ANY fraudulent transaction).
    """
    df = transactions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Integer codes per entity
    df["card_code"] = df["user_id"].astype("category").cat.codes
    df["merchant_code"] = df["merchant_id"].astype("category").cat.codes
    df["device_code"] = df["device_id"].astype("category").cat.codes

    data = HeteroData()

    # ---- Compute pre-fraud mask ----
    pre_fraud_mask = _get_pre_fraud_mask(df)
    df_clean = df[pre_fraud_mask]  # only pre-fraud transactions for features

    print(f"Total transactions: {len(df)}")
    print(f"Pre-fraud transactions (used for features): {len(df_clean)}")
    print(f"Excluded (fraud + post-fraud): {len(df) - len(df_clean)}")

    # ---- Node features (from pre-fraud transactions ONLY) ----
    # Some users might have ALL transactions excluded (if fraud was their
    # first transaction). Handle with fillna defaults.
    all_card_codes = sorted(df["card_code"].unique())
    all_merchant_codes = sorted(df["merchant_code"].unique())
    all_device_codes = sorted(df["device_code"].unique())

    card_stats = df_clean.groupby("card_code").agg(
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        txn_count=("amount", "count"),
        n_merchants=("merchant_code", "nunique"),
        n_devices=("device_code", "nunique"),
    ).reindex(all_card_codes).fillna(0)
    data["card"].x = _standardize(torch.tensor(card_stats.values, dtype=torch.float))

    merchant_stats = df_clean.groupby("merchant_code").agg(
        avg_amount=("amount", "mean"),
        txn_count=("amount", "count"),
        n_cards=("card_code", "nunique"),
    ).reindex(all_merchant_codes).fillna(0)
    data["merchant"].x = _standardize(torch.tensor(merchant_stats.values, dtype=torch.float))

    device_stats = df_clean.groupby("device_code").agg(
        txn_count=("amount", "count"),
        n_cards=("card_code", "nunique"),
    ).reindex(all_device_codes).fillna(0)
    data["device"].x = _standardize(torch.tensor(device_stats.values, dtype=torch.float))

    # ---- Labels: card-level fraud (uses ALL transactions, not just pre-fraud) ----
    card_labels = df.groupby("card_code")["is_fraud"].max().reindex(all_card_codes).fillna(0)
    data["card"].y = torch.tensor(card_labels.values, dtype=torch.float)

    # ---- Edges (from ALL transactions — the graph structure itself is fine,
    # only the node FEATURES needed the leak fix) ----
    data["card", "transaction", "merchant"].edge_index = torch.tensor(
        np.vstack([df["card_code"].values, df["merchant_code"].values]), dtype=torch.long
    )
    data["card", "used_by", "device"].edge_index = torch.tensor(
        np.vstack([df["card_code"].values, df["device_code"].values]), dtype=torch.long
    )

    # Reverse edges for bidirectional message passing
    data["merchant", "rev_transaction", "card"].edge_index = \
        data["card", "transaction", "merchant"].edge_index.flip(0)
    data["device", "rev_used_by", "card"].edge_index = \
        data["card", "used_by", "device"].edge_index.flip(0)

    n_fraud = int(card_labels.sum())
    print(f"Graph: {len(all_card_codes)} cards, {len(all_merchant_codes)} merchants, "
          f"{len(all_device_codes)} devices, {len(df)} edges, fraud cards: {n_fraud}")

    return data