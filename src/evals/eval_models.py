# src/evals/eval_models.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))

from entities import generate_users
from features import engineer_features
from train_baseline import FEATURE_COLS


def load_golden_test():
    """Load the golden test set and engineer features."""
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "golden" / "test.csv", parse_dates=["timestamp"])
    users = generate_users(n_users=5000)
    users_df = pd.DataFrame(users)
    test_df = engineer_features(test_df, users_df)
    return test_df


def eval_lightgbm(test_df):
    """Evaluate saved LightGBM model on golden test set."""
    import lightgbm as lgb
    model_path = PROJECT_ROOT / "data" / "processed" / "lgbm_baseline.txt"
    booster = lgb.Booster(model_file=str(model_path))
    probs = booster.predict(test_df[FEATURE_COLS])
    return probs


def eval_transformer(test_df):
    """Evaluate saved transformer on golden test set."""
    import torch
    from sequence_data import TransactionSequenceDataset, SEQ_FEATURES
    from transformer_model import FraudSequenceTransformer
    from torch.utils.data import DataLoader

    dataset = TransactionSequenceDataset(test_df)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    model = FraudSequenceTransformer(n_features=len(SEQ_FEATURES))
    model.load_state_dict(torch.load(
        PROJECT_ROOT / "data" / "processed" / "transformer_baseline.pt",
        map_location="cpu", weights_only=True
    ))
    model.eval()

    all_probs = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            probs = torch.sigmoid(logits).numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


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


def evaluate_model(test_df, probs, model_name):
    """Full evaluation: PR-AUC, ROC-AUC, classification report, per-typology."""
    labels = test_df["is_fraud"].values

    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(classification_report(labels, probs > 0.5, zero_division=0))

    per_typology_recall(test_df, probs, model_name)

    return {"model": model_name, "pr_auc": pr_auc, "roc_auc": roc_auc}


def main():
    print("Loading golden test set...")
    test_df = load_golden_test()
    print(f"Test set: {len(test_df)} transactions, {test_df['is_fraud'].sum()} fraud")

    results = []

    # LightGBM
    print("\nEvaluating LightGBM...")
    lgbm_probs = eval_lightgbm(test_df)
    results.append(evaluate_model(test_df, lgbm_probs, "LightGBM"))

    # Transformer
    print("\nEvaluating Transformer...")
    transformer_probs = eval_transformer(test_df)
    results.append(evaluate_model(test_df, transformer_probs, "Transformer"))

    # Summary comparison table
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()