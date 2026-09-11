# src/evals/create_golden_set.py
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project root = two levels up from this file (src/evals/ -> src/ -> finguard/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))

from build_dataset import build_dataset


def create_golden_set(transactions_df, test_size=0.2, random_state=42):
    """Split the full dataset into golden train and test sets."""
    golden_dir = PROJECT_ROOT / "data" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)  # create the folder if it doesn't exist

    train_df, test_df = train_test_split(
        transactions_df,
        test_size=test_size,
        stratify=transactions_df["is_fraud"],
        random_state=random_state,
    )

    train_df.to_csv(golden_dir / "train.csv", index=False)
    test_df.to_csv(golden_dir / "test.csv", index=False)

    print(f"Golden train set: {len(train_df)} transactions")
    print(f"Golden test set: {len(test_df)} transactions")
    print(f"Test set fraud rate: {test_df['is_fraud'].mean():.4f}")


if __name__ == "__main__":
    transactions_df = build_dataset(n_users=5000, n_merchants=800, days=90, fraud_rate=0.05)
    create_golden_set(transactions_df)