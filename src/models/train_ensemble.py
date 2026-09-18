# src/models/train_ensemble.py
import sys
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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


def build_meta_features(df):
    """
    Meta-feature matrix from two base models (LightGBM + Transformer)
    plus engineered interaction features and raw context.
    """
    print("  Getting LightGBM predictions...")
    lgbm_probs = get_lgbm_predictions(df)

    print("  Getting Transformer predictions...")
    transformer_probs = get_transformer_predictions(df)

    meta_features = pd.DataFrame({
        # Base model scores
        "lgbm_prob": lgbm_probs,
        "transformer_prob": transformer_probs,

        # Model disagreement — the key signal from our error analysis
        "prob_diff": lgbm_probs - transformer_probs,
        "prob_max": np.maximum(lgbm_probs, transformer_probs),
        "prob_mean": (lgbm_probs + transformer_probs) / 2,
        "prob_min": np.minimum(lgbm_probs, transformer_probs),

        # Both models agree it's fraud?
        "both_flag": ((lgbm_probs > 0.5) & (transformer_probs > 0.3)).astype(int),

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
    print(f"\n{model_name} — Per-typology recall (threshold={threshold:.4f}):")
    print(breakdown)
    return breakdown


def main():
    print("Loading data...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "transactions.csv", parse_dates=["timestamp"])
    users = generate_users(n_users=5000)
    users_df = pd.DataFrame(users)
    df = engineer_features(df, users_df)

    train_df, test_df = time_based_split(df, split_date="2026-03-15")

    # Last 20% of training data for meta-learner
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

    # Scale features
    scaler = StandardScaler()
    meta_train_X_scaled = pd.DataFrame(
        scaler.fit_transform(meta_train_X),
        columns=meta_train_X.columns
    )
    meta_test_X_scaled = pd.DataFrame(
        scaler.transform(meta_test_X),
        columns=meta_test_X.columns
    )

    # Train meta-learner
    print("\nTraining stacking ensemble (Logistic Regression)...")
    meta_model = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=42)
    meta_model.fit(meta_train_X_scaled, meta_train_y)

    ensemble_probs = meta_model.predict_proba(meta_test_X_scaled)[:, 1]
    lgbm_probs = meta_test_X["lgbm_prob"].values
    transformer_probs = meta_test_X["transformer_prob"].values

    # ---- Coefficients: what the meta-learner learned ----
    print("\nMeta-learner coefficients:")
    coefs = pd.Series(meta_model.coef_[0], index=meta_train_X.columns).sort_values(ascending=False)
    print(coefs.to_string())

    # ---- Full comparison ----
    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*60}")

    results_summary = []
    for name, probs in [("LightGBM (base)", lgbm_probs),
                         ("Transformer (base)", transformer_probs),
                         ("Stacking Ensemble", ensemble_probs)]:
        pr_auc = average_precision_score(meta_test_y, probs)
        roc_auc = roc_auc_score(meta_test_y, probs)

        # Optimal F1 threshold
        precisions, recalls, thresholds = precision_recall_curve(meta_test_y, probs)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        # Default threshold metrics
        preds_default = (probs > 0.5).astype(int)
        fp_default = ((preds_default == 1) & (meta_test_y == 0)).sum()

        print(f"\n--- {name} ---")
        print(f"PR-AUC:  {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"At threshold=0.5: precision={classification_report(meta_test_y, preds_default, output_dict=True, zero_division=0)['1']['precision']:.3f}, "
              f"recall={classification_report(meta_test_y, preds_default, output_dict=True, zero_division=0)['1']['recall']:.3f}, "
              f"false_positives={fp_default}")
        print(f"Optimal F1: {f1_scores[best_idx]:.3f} (threshold={best_threshold:.4f}, "
              f"precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f})")
        per_typology_recall(test_df.reset_index(drop=True), probs, name, threshold=best_threshold)

        results_summary.append({
            "model": name,
            "pr_auc": round(pr_auc, 4),
            "best_f1": round(f1_scores[best_idx], 3),
            "precision": round(precisions[best_idx], 3),
            "recall": round(recalls[best_idx], 3),
            "fp_at_0.5": fp_default,
        })

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    # Save
    joblib.dump({
        "meta_model": meta_model,
        "scaler": scaler,
        "feature_names": meta_train_X.columns.tolist(),
    }, str(PROJECT_ROOT / "data" / "processed" / "ensemble_model.pkl"))
    print(f"\nEnsemble saved to data/processed/ensemble_model.pkl")


if __name__ == "__main__":
    main()