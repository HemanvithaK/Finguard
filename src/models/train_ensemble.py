# src/models/train_ensemble.py
import sys
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from torch.utils.data import DataLoader
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
    """Get LightGBM probability scores."""
    booster = lgb.Booster(model_file=str(PROJECT_ROOT / "data" / "processed" / "lgbm_baseline.txt"))
    return booster.predict(df[FEATURE_COLS])


def get_transformer_predictions(df):
    """Get transformer probability scores."""
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
    Build the feature matrix for the meta-learner:
    - LightGBM probability
    - Transformer probability
    - The difference between them (captures disagreement)
    - The max of the two (captures when either model is confident)
    - Key raw features the meta-learner can use for context
    """
    print("  Getting LightGBM predictions...")
    lgbm_probs = get_lgbm_predictions(df)

    print("  Getting Transformer predictions...")
    transformer_probs = get_transformer_predictions(df)

    meta_features = pd.DataFrame({
        "lgbm_prob": lgbm_probs,
        "transformer_prob": transformer_probs,
        "prob_diff": lgbm_probs - transformer_probs,
        "prob_max": np.maximum(lgbm_probs, transformer_probs),
        "prob_mean": (lgbm_probs + transformer_probs) / 2,
        # Include key raw features so the meta-learner has context
        # about WHY the models disagree
        "amount": df["amount"].values,
        "time_since_prev_txn": df["time_since_prev_txn"].values,
        "dist_from_home_km": df["dist_from_home_km"].values,
        "is_known_device": df["is_known_device"].values,
        "txns_last_10min": df["txns_last_10min"].values,
        "distinct_merchants_1hr": df["distinct_merchants_1hr"].values,
    })

    return meta_features


def per_typology_recall(test_df, probs, model_name, threshold=0.5):
    """Per-typology recall breakdown."""
    preds = (probs > threshold).astype(int)
    fraud_mask = test_df["is_fraud"] == 1
    fraud_df = test_df[fraud_mask].copy()
    fraud_df["predicted"] = preds[fraud_mask.values]

    breakdown = fraud_df.groupby("fraud_type").apply(
        lambda g: pd.Series({
            "n_cases": len(g),
            "caught": int(g["predicted"].sum()),
            "recall": round(g["predicted"].mean(), 3),
        })
    )
    print(f"\n{model_name} — Per-typology recall (threshold={threshold}):")
    print(breakdown)
    return breakdown


def main():
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "transactions.csv", parse_dates=["timestamp"])
    users = generate_users(n_users=5000)
    users_df = pd.DataFrame(users)
    df = engineer_features(df, users_df)

    train_df, test_df = time_based_split(df, split_date="2026-03-15")

    # Split training data: first 80% for base models (already trained),
    # last 20% for training the meta-learner. This prevents the meta-learner
    # from seeing the same data the base models trained on, which would
    # inflate its performance unrealistically.
    meta_train_cutoff = int(len(train_df) * 0.8)
    meta_train_df = train_df.iloc[meta_train_cutoff:].copy().reset_index(drop=True)

    print(f"\nMeta-learner training set: {len(meta_train_df)} transactions")
    print(f"Test set: {len(test_df)} transactions")

    # Build meta-features
    print("\nBuilding meta-features for training...")
    meta_train_X = build_meta_features(meta_train_df)
    meta_train_y = meta_train_df["is_fraud"].values

    print("\nBuilding meta-features for testing...")
    meta_test_X = build_meta_features(test_df)
    meta_test_y = test_df["is_fraud"].values

    # Train the meta-learner (logistic regression — intentionally simple,
    # since the complexity is already in the base models)
    print("\nTraining stacking ensemble (Logistic Regression meta-learner)...")

    # Handle class imbalance with class_weight
    meta_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    meta_model.fit(meta_train_X, meta_train_y)

    # Print which base model the meta-learner trusts more
    feature_names = meta_train_X.columns.tolist()
    coefs = pd.Series(meta_model.coef_[0], index=feature_names).sort_values(ascending=False)
    print("\nMeta-learner coefficients (what it learned to trust):")
    print(coefs.to_string())

    # Predict
    ensemble_probs = meta_model.predict_proba(meta_test_X)[:, 1]

    # Also get individual model predictions for comparison
    lgbm_probs = meta_test_X["lgbm_prob"].values
    transformer_probs = meta_test_X["transformer_prob"].values

    # ---- Results comparison ----
    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*60}")

    for name, probs in [("LightGBM", lgbm_probs),
                         ("Transformer", transformer_probs),
                         ("Stacking Ensemble", ensemble_probs)]:
        pr_auc = average_precision_score(meta_test_y, probs)
        roc_auc = roc_auc_score(meta_test_y, probs)
        print(f"\n--- {name} ---")
        print(f"PR-AUC:  {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(classification_report(meta_test_y, probs > 0.5, zero_division=0))
        per_typology_recall(test_df.reset_index(drop=True), probs, name)

    # Find optimal threshold for ensemble
    precisions, recalls, thresholds = precision_recall_curve(meta_test_y, ensemble_probs)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"\n--- Ensemble at optimal F1 threshold ({best_threshold:.4f}) ---")
    print(f"Precision: {precisions[best_idx]:.3f}")
    print(f"Recall: {recalls[best_idx]:.3f}")
    print(f"F1: {f1_scores[best_idx]:.3f}")
    per_typology_recall(test_df.reset_index(drop=True), ensemble_probs, "Ensemble (optimal)", threshold=best_threshold)

    # Save the ensemble
    joblib.dump({
        "meta_model": meta_model,
        "feature_names": feature_names,
        "best_threshold": best_threshold,
    }, str(PROJECT_ROOT / "data" / "processed" / "ensemble_model.pkl"))
    print(f"\nEnsemble saved to data/processed/ensemble_model.pkl")

    return meta_model, ensemble_probs


if __name__ == "__main__":
    main()