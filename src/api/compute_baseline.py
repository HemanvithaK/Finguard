# src/api/compute_baseline.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))

import pandas as pd
from entities import generate_users
from features import engineer_features
from train_baseline import time_based_split, FEATURE_COLS
from drift_detector import DriftDetector


def main():
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "transactions.csv", parse_dates=["timestamp"])
    users = generate_users(n_users=5000)
    users_df = pd.DataFrame(users)
    df = engineer_features(df, users_df)
    train_df, _ = time_based_split(df, split_date="2026-03-15")

    detector = DriftDetector(feature_names=FEATURE_COLS)
    detector.compute_baseline(train_df, FEATURE_COLS)

    baseline_path = PROJECT_ROOT / "data" / "processed" / "drift_baseline.json"
    detector.save_baseline(str(baseline_path))
    print(f"Baseline saved to {baseline_path}")
    print(f"Features tracked: {FEATURE_COLS}")
    print(f"Training samples used: {len(train_df)}")


if __name__ == "__main__":
    main()