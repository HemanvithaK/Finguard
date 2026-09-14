# src/models/transformer_error_analysis.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))

import torch
from torch.utils.data import DataLoader
from entities import generate_users
from features import engineer_features
from train_baseline import time_based_split, FEATURE_COLS
from sequence_data import TransactionSequenceDataset, SEQ_FEATURES
from transformer_model import FraudSequenceTransformer
import lightgbm as lgb


def load_and_predict():
    """Load data, run both models, return predictions side by side."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "transactions.csv", parse_dates=["timestamp"])
    users = generate_users(n_users=5000)
    users_df = pd.DataFrame(users)
    df = engineer_features(df, users_df)

    train_df, test_df = time_based_split(df, split_date="2026-03-15")

    # --- LightGBM predictions ---
    booster = lgb.Booster(model_file=str(PROJECT_ROOT / "data" / "processed" / "lgbm_baseline.txt"))
    lgbm_probs = booster.predict(test_df[FEATURE_COLS])

    # --- Transformer predictions ---
    test_dataset = TransactionSequenceDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = FraudSequenceTransformer(n_features=len(SEQ_FEATURES))
    model.load_state_dict(torch.load(
        PROJECT_ROOT / "data" / "processed" / "transformer_baseline.pt",
        map_location="cpu", weights_only=True
    ))
    model.eval()

    transformer_probs = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            logits = model(x_batch)
            probs = torch.sigmoid(logits).numpy()
            transformer_probs.extend(probs)
    transformer_probs = np.array(transformer_probs)

    # --- Combine into one DataFrame for analysis ---
    results = test_df.copy().reset_index(drop=True)
    results["lgbm_prob"] = lgbm_probs
    results["transformer_prob"] = transformer_probs
    results["lgbm_correct"] = ((lgbm_probs > 0.5) == results["is_fraud"]).astype(int)
    results["transformer_correct"] = ((transformer_probs > 0.5) == results["is_fraud"]).astype(int)

    return results


def analyze_errors(results):
    """Find WHERE the transformer fails that LightGBM succeeds."""

    fraud_only = results[results["is_fraud"] == 1].copy()
    normal_only = results[results["is_fraud"] == 0].copy()

    print(f"{'='*60}")
    print(f"  TRANSFORMER ERROR ANALYSIS")
    print(f"{'='*60}")
    print(f"\nTotal test transactions: {len(results)}")
    print(f"Fraud transactions: {len(fraud_only)}")
    print(f"Normal transactions: {len(normal_only)}")

    # 1. Overall accuracy comparison
    print(f"\n--- Overall Detection (threshold=0.5) ---")
    for name, col in [("LightGBM", "lgbm_correct"), ("Transformer", "transformer_correct")]:
        fraud_recall = fraud_only[col].mean()
        normal_precision = normal_only[col].mean()
        print(f"{name}: fraud recall={fraud_recall:.3f}, normal accuracy={normal_precision:.3f}")

    # 2. Where transformer misses but LightGBM catches
    transformer_misses = fraud_only[
        (fraud_only["lgbm_correct"] == 1) & (fraud_only["transformer_correct"] == 0)
    ]
    print(f"\n--- Fraud cases: LightGBM catches, Transformer misses: {len(transformer_misses)} ---")

    if len(transformer_misses) > 0:
        print(f"\nBy fraud type:")
        print(transformer_misses["fraud_type"].value_counts().to_string())

        print(f"\nAverage feature values of missed cases:")
        missed_features = ["amount", "time_since_prev_txn", "dist_from_home_km",
                           "amount_vs_avg_ratio", "is_known_device", "hour_of_day",
                           "txns_last_10min", "distinct_merchants_1hr"]
        for f in missed_features:
            missed_avg = transformer_misses[f].mean()
            caught_avg = fraud_only[fraud_only["transformer_correct"] == 1][f].mean()
            print(f"  {f}: missed={missed_avg:.2f} vs caught={caught_avg:.2f}")

    # 3. Where transformer catches but LightGBM misses (transformer's strengths)
    transformer_wins = fraud_only[
        (fraud_only["lgbm_correct"] == 0) & (fraud_only["transformer_correct"] == 1)
    ]
    print(f"\n--- Fraud cases: Transformer catches, LightGBM misses: {len(transformer_wins)} ---")
    if len(transformer_wins) > 0:
        print(f"\nBy fraud type:")
        print(transformer_wins["fraud_type"].value_counts().to_string())

    # 4. False positive comparison
    lgbm_fp = normal_only[normal_only["lgbm_correct"] == 0]
    transformer_fp = normal_only[normal_only["transformer_correct"] == 0]
    print(f"\n--- False Positives ---")
    print(f"LightGBM: {len(lgbm_fp)} false positives ({len(lgbm_fp)/len(normal_only)*100:.2f}%)")
    print(f"Transformer: {len(transformer_fp)} false positives ({len(transformer_fp)/len(normal_only)*100:.2f}%)")

    # 5. Probability distribution comparison on fraud cases
    print(f"\n--- Probability Distribution on Fraud Cases ---")
    for fraud_type in fraud_only["fraud_type"].unique():
        subset = fraud_only[fraud_only["fraud_type"] == fraud_type]
        print(f"\n  {fraud_type} (n={len(subset)}):")
        print(f"    LightGBM prob:    mean={subset['lgbm_prob'].mean():.3f}, "
              f"median={subset['lgbm_prob'].median():.3f}, "
              f"min={subset['lgbm_prob'].min():.3f}")
        print(f"    Transformer prob: mean={subset['transformer_prob'].mean():.3f}, "
              f"median={subset['transformer_prob'].median():.3f}, "
              f"min={subset['transformer_prob'].min():.3f}")

    # 6. The key diagnostic: what features distinguish missed from caught?
    print(f"\n--- Key Diagnostic: What Makes Missed Cases Different? ---")
    if len(transformer_misses) > 0:
        missed_types = transformer_misses["fraud_type"].value_counts()
        dominant_type = missed_types.index[0]
        print(f"\nMost missed type: {dominant_type} ({missed_types.iloc[0]} cases)")

        # Compare missed vs caught within that typology
        same_type_caught = fraud_only[
            (fraud_only["fraud_type"] == dominant_type) & (fraud_only["transformer_correct"] == 1)
        ]
        same_type_missed = transformer_misses[transformer_misses["fraud_type"] == dominant_type]

        print(f"\nWithin {dominant_type} — missed vs caught:")
        for f in missed_features:
            m = same_type_missed[f].mean()
            c = same_type_caught[f].mean() if len(same_type_caught) > 0 else float("nan")
            diff = "↑" if m > c else "↓"
            print(f"  {f}: missed={m:.2f}, caught={c:.2f} {diff}")

    return transformer_misses


if __name__ == "__main__":
    print("Loading data and running both models...")
    results = load_and_predict()
    missed = analyze_errors(results)

    # Save for further exploration in a notebook if needed
    results.to_csv(PROJECT_ROOT / "data" / "processed" / "error_analysis.csv", index=False)
    print(f"\nFull results saved to data/processed/error_analysis.csv")