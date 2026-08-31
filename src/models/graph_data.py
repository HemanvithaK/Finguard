# src/models/graph_data.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData


def _standardize(t):
    """Zero-mean/unit-variance per feature — raw stats (e.g. txn counts vs
    dollar amounts) are on wildly different scales, which destabilizes
    training. Verified: without this, early loss spikes 10x."""
    mean = t.mean(dim=0, keepdim=True)
    std = t.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (t - mean) / std


def construct_graph(transactions_df):
    """
    Heterogeneous graph:
      nodes: card (user), merchant, device
      edges: (card->merchant) transaction, (card->device) used_by, + reverse edges
    Node features are aggregated per entity. Label: card-level fraud
    (1 if that user has any fraudulent transaction).
    """
    df = transactions_df.copy()

    # Integer codes per entity — these become node indices in the graph
    df["card_code"] = df["user_id"].astype("category").cat.codes
    df["merchant_code"] = df["merchant_id"].astype("category").cat.codes
    df["device_code"] = df["device_id"].astype("category").cat.codes

    data = HeteroData()

    # ---- Node features (aggregated per entity) ----
    card_stats = df.groupby("card_code").agg(
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        txn_count=("amount", "count"),
        n_merchants=("merchant_code", "nunique"),
        n_devices=("device_code", "nunique"),
    ).sort_index()
    data["card"].x = _standardize(torch.tensor(card_stats.values, dtype=torch.float))

    merchant_stats = df.groupby("merchant_code").agg(
        avg_amount=("amount", "mean"),
        txn_count=("amount", "count"),
        n_cards=("card_code", "nunique"),
    ).sort_index()
    data["merchant"].x = _standardize(torch.tensor(merchant_stats.values, dtype=torch.float))

    device_stats = df.groupby("device_code").agg(
        txn_count=("amount", "count"),
        n_cards=("card_code", "nunique"),
    ).sort_index()
    data["device"].x = _standardize(torch.tensor(device_stats.values, dtype=torch.float))

    # ---- Labels: was this card/user ever hit by fraud? ----
    card_labels = df.groupby("card_code")["is_fraud"].max().sort_index()
    data["card"].y = torch.tensor(card_labels.values, dtype=torch.float)

    # ---- Edges ----
    data["card", "transaction", "merchant"].edge_index = torch.tensor(
        np.vstack([df["card_code"].values, df["merchant_code"].values]), dtype=torch.long
    )
    data["card", "used_by", "device"].edge_index = torch.tensor(
        np.vstack([df["card_code"].values, df["device_code"].values]), dtype=torch.long
    )

    # Reverse edges so messages flow BOTH ways — without these, card nodes
    # (the ones we classify) never receive information from their neighbors
    data["merchant", "rev_transaction", "card"].edge_index = \
        data["card", "transaction", "merchant"].edge_index.flip(0)
    data["device", "rev_used_by", "card"].edge_index = \
        data["card", "used_by", "device"].edge_index.flip(0)

    print(f"Graph: {df['card_code'].nunique()} cards, "
          f"{df['merchant_code'].nunique()} merchants, "
          f"{df['device_code'].nunique()} devices, "
          f"{len(df)} transaction edges, fraud cards: {int(card_labels.sum())}")
    return data