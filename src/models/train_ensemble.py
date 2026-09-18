# src/models/train_ensemble.py
import sys
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, classification_report,
    precision_recall_curve
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))

import lightgbm as lgb
from entities import generate_users
from features import engineer_features
from train_baseline import time_based_split, FEATURE_COLS
from sequence_data import TransactionSequenceDataset, SEQ_FEATURES
from transformer_model import FraudSequenceTransformer
from graph_data import construct_graph


def get_lgbm_predictions(df):
    booster = lgb.Booster(model_file=str(PROJECT_ROOT / "data" / "processed" / "lgbm_baseline.txt"))
    return booster.predict(df[FEATURE_COLS])


def get_transformer_predictions(df):
    dataset = TransactionSequenceDataset(df)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = FraudSequenceTransformer(n_features=len(SEQ_FEATURES))
    model.load_state_dict(torch.load(
        PROJECT_ROOT / "data" / "processed" / "transformer_v2.pt",
        map_location="cpu", weights_only=True
    ))
    model.eval()

    probs = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            probs.extend(torch.sigmoid(logits).numpy())
    return np.array(probs)


def get_gnn_card_scores(df):
    """
    Get per-user fraud scores from the GNN. Since the GNN predicts at
    the card/user level (not per-transaction), we run it once and map
    each transaction's user_id to that user's GNN score.
    """
    from gnn_model import FraudGNN

    graph = construct_graph(df)

    model = FraudGNN(hidden_channels=64)
    model.load_state_dict(torch.load(
        PROJECT_ROOT / "data" / "processed" / "gnn_baseline.pt",
        map_location="cpu", weights_only=True
    ))
    model.eval()

    with torch.no_grad():
        logits = model(graph.x_dict, graph.edge_index_dict)
        card_probs = torch.sigmoid(logits).numpy()

    # Map card_code back to user_id, then to each transaction
    df_temp = df.copy()
    df_temp["card_code"] = df_temp["user_id"].astype("category").cat.codes
    user_scores = pd.Series(card_probs, index=range(len(card_probs)))
    gnn_scores = df_temp["card_code"].map(user_scores).values

    return gnn_scores.astype(float)


def build_meta_features(df):
    """
    Build the feature matrix for the meta-learner using all three
    base models plus engineered interaction features.
    """
    print("  Getting LightGBM predictions...")
    lgbm_probs = get_lgbm_predictions(df)

    print("  Getting Transformer predictions...")
    transformer_probs = get_transformer_predictions(df)

    print("  Getting GNN card-level scores...")
    gnn_probs = get_gnn_card_scores(df)

    meta_features = pd.DataFrame({
        # Base model scores
        "lgbm_prob": lgbm_probs,
        "transformer_prob": transformer_probs,
        "gnn_prob": gnn_probs,

        # Pairwise disagreements — when models disagree, that's informative
        "lgbm_transformer_diff": lgbm_probs - transformer_probs,
        "lgbm_gnn_diff": lgbm_probs - gnn_probs,
        "transformer_gnn_diff": transformer_probs - gnn_probs,

        # Aggregate signals
        "prob_max": np.maximum(np.maximum(lgbm_probs, transformer_probs), gnn_probs),
        "prob_mean": (lgbm_probs + transformer_probs + gnn_probs) / 3,
        "prob_min": np.minimum(np.minimum(lgbm_probs, transformer_probs), gnn_probs),

        # How many models agree this is fraud (>0.5)?
        "n_models_flagging": (
            (lgbm_probs > 0.5).astype(int) +
            (transformer_probs > 0.3).astype(int) +  # lower threshold for transformer (calibration)
            (gnn_probs > 0.5).astype(int)
        ),

        # Key raw features for context
        "amount": df["amount"].values,
        "time_since_prev_txn": df["time_since_prev_txn"].values,
        "dist_from_home_km": df["dist_from_home_km"].values,
        "is_known_device": df["is_known_device"].values,
        "txns_last_10min": df["txns_last_10min"].values,
        "distinct_merchants_1hr": df["distinct_merchants_1hr"].values,
        "amount_vs_avg_ratio": df["amount_vs_avg_ratio"].values,
        "hour_of_day": df["hour_of_day"].values,
    })

    return meta_features


def per_typology_recall(test_df, probs, model_name, threshold=0.5):
    preds = (probs > threshold).astype(int)
    fraud_mask = test_df["is_fraud"] == 1
    fraud_df = test_df[fraud_mask].copy()
    fraud_df["predicted"] = preds[fraud_mask.values]

    breakdown = fraud_df.groupby("fraud_type").apply(
        lambda g: pd.Series({
            "n_cases": len(g),
            "caught": int(g["predicted"].sum()),
            "recall": round(g["predicted"].mean(), 3),
        }),
        include_groups=False
    )
    print(f"\n{model_name} — Per-typology recall (threshold={threshold}):")
    print(breakdown)
    return breakdown


def main():
    print("Loading data...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "transactions.csv", parse_dates=["timestamp"])
    users = generate_users(n_users=5000)
    users_df = pd.DataFrame(users)
    df = engineer_features(df, users_df)

    train_df, test_df = time_based_split(df, split_date="2026-03-15")

    # Last 20% of training data for meta-learner training
    meta_train_cutoff = int(len(train_df) * 0.8)
    meta_train_df = train_df.iloc[meta_train_cutoff:].copy().reset_index(drop=True)

    print(f"\nMeta-learner training set: {len(meta_train_df)} transactions")
    print(f"Test set: {len(test_df)} transactions")

    print("\nBuilding meta-features for training...")
    meta_train_X = build_meta_features(meta_train_df)
    meta_train_y = meta_train_df["is_fraud"].values

    print("\nBuilding meta-features for testing...")
    meta_test_X = build_meta_features(test_df)
    meta_test_y = test_df["is_fraud"].values

    # Scale features to fix the convergence issue
    scaler = StandardScaler()
    meta_train_X_scaled = pd.DataFrame(
        scaler.fit_transform(meta_train_X),
        columns=meta_train_X.columns
    )
    meta_test_X_scaled = pd.DataFrame(
        scaler.transform(meta_test_X),
        columns=meta_test_X.columns
    )

    # ---- Train TWO meta-learners and pick the best ----

    # Meta-learner A: Logistic Regression (simple, interpretable)
    from sklearn.linear_model import LogisticRegression
    print("\nTraining Meta-learner A: Logistic Regression...")
    meta_lr = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=42)
    meta_lr.fit(meta_train_X_scaled, meta_train_y)
    lr_probs = meta_lr.predict_proba(meta_test_X_scaled)[:, 1]

    # Meta-learner B: LightGBM (can learn non-linear interactions)
    print("Training Meta-learner B: LightGBM...")
    n_neg = (meta_train_y == 0).sum()
    n_pos = (meta_train_y == 1).sum()
    meta_lgbm = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,        # shallow trees — we don't want the meta-learner to overfit
        learning_rate=0.05,
        scale_pos_weight=n_neg / n_pos,
        random_state=42,
        verbose=-1,
    )
    meta_lgbm.fit(meta_train_X, meta_train_y)  # LightGBM doesn't need scaling
    lgbm_meta_probs = meta_lgbm.predict_proba(meta_test_X)[:, 1]

    # Individual base model scores for comparison
    lgbm_probs = meta_test_X["lgbm_prob"].values
    transformer_probs = meta_test_X["transformer_prob"].values
    gnn_probs = meta_test_X["gnn_prob"].values

    # ---- Full comparison ----
    print(f"\n{'='*60}")
    print(f"  FULL MODEL COMPARISON")
    print(f"{'='*60}")

    all_models = [
        ("LightGBM (base)", lgbm_probs, 0.5),
        ("Transformer (base)", transformer_probs, 0.5),
        ("GNN (base)", gnn_probs, 0.5),
        ("Ensemble — LR meta", lr_probs, 0.5),
        ("Ensemble — LightGBM meta", lgbm_meta_probs, 0.5),
    ]

    results_summary = []
    for name, probs, threshold in all_models:
        pr_auc = average_precision_score(meta_test_y, probs)
        roc_auc = roc_auc_score(meta_test_y, probs)

        # Find optimal F1 threshold
        precisions, recalls, thresholds = precision_recall_curve(meta_test_y, probs)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        print(f"\n--- {name} ---")
        print(f"PR-AUC:  {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"At default threshold (0.5): {classification_report(meta_test_y, probs > 0.5, zero_division=0)}")
        print(f"Optimal F1 threshold: {best_threshold:.4f}")
        print(f"At optimal: precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f}")
        per_typology_recall(test_df.reset_index(drop=True), probs, name, threshold=best_threshold)

        results_summary.append({
            "model": name,
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "best_f1": round(f1_scores[best_idx], 3),
            "precision_at_best": round(precisions[best_idx], 3),
            "recall_at_best": round(recalls[best_idx], 3),
        })

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    # ---- Feature importances from LightGBM meta-learner ----
    print(f"\n--- LightGBM Meta-learner Feature Importances ---")
    importances = pd.Series(
        meta_lgbm.feature_importances_,
        index=meta_train_X.columns
    ).sort_values(ascending=False)
    print(importances.to_string())

    # Save the best ensemble
    best_meta = meta_lgbm  # pick whichever performed better
    joblib.dump({
        "meta_model": best_meta,
        "scaler": scaler,
        "feature_names": meta_train_X.columns.tolist(),
    }, str(PROJECT_ROOT / "data" / "processed" / "ensemble_model.pkl"))
    print(f"\nEnsemble saved to data/processed/ensemble_model.pkl")


if __name__ == "__main__":
    main()