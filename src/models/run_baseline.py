# src/models/run_baseline.py
import sys
from xml.parsers.expat import model
import pandas as pd
sys.path.append("../data_gen")

from entities import generate_users
from features import engineer_features
from train_baseline import time_based_split, train_lightgbm, per_typology_recall, FEATURE_COLS


def main():
    # 1. Load the raw transactions from Phase 1
    df = pd.read_csv("../../data/raw/transactions.csv", parse_dates=["timestamp"])

    # 2. Regenerate users_df with the SAME seed/params used in build_dataset.py
    users = generate_users(n_users=5000)
    users_df = pd.DataFrame(users)

    # 3. Feature engineering
    df = engineer_features(df, users_df)

    print(df[FEATURE_COLS + ["is_fraud"]].head(10))
    print("\nAny NaNs in features?\n", df[FEATURE_COLS].isna().sum())

    # 4. Time-based split
    train_df, test_df = time_based_split(df, split_date="2026-03-15")
    print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"Train fraud count: {train_df['is_fraud'].sum()}, Test fraud count: {test_df['is_fraud'].sum()}")

    # 5. Train
    model, probs = train_lightgbm(train_df, test_df)

    # 6. Per-typology recall breakdown
    per_typology_recall(test_df, probs)

    model.booster_.save_model("../../data/processed/lgbm_baseline.txt")
    print("\nModel saved to data/processed/lgbm_baseline.txt")

    return model, test_df, probs


if __name__ == "__main__":
    main()