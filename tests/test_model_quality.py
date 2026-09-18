# tests/test_model_quality.py
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
sys.path.append(str(PROJECT_ROOT / "src" / "evals"))

from eval_models import load_golden_test, eval_lightgbm
from sklearn.metrics import average_precision_score

# Check if torch is available (not installed in CI)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture(scope="session")
def test_df():
    return load_golden_test()


@pytest.fixture(scope="session")
def lgbm_probs(test_df):
    return eval_lightgbm(test_df)


@pytest.fixture(scope="session")
def transformer_probs(test_df):
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    from eval_models import eval_transformer
    return eval_transformer(test_df)


class TestLightGBM:
    def test_pr_auc_above_minimum(self, test_df, lgbm_probs):
        pr_auc = average_precision_score(test_df["is_fraud"], lgbm_probs)
        assert pr_auc > 0.80, f"LightGBM PR-AUC dropped to {pr_auc:.4f}"

    def test_recall_above_minimum(self, test_df, lgbm_probs):
        preds = (lgbm_probs > 0.5).astype(int)
        fraud_mask = test_df["is_fraud"] == 1
        recall = preds[fraud_mask.values].mean()
        assert recall >= 0.70, f"LightGBM recall dropped to {recall:.3f}"

    def test_no_nan_predictions(self, lgbm_probs):
        assert not np.any(np.isnan(lgbm_probs))

    def test_predictions_in_range(self, lgbm_probs):
        assert np.all(lgbm_probs >= 0) and np.all(lgbm_probs <= 1)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed — skipped in CI")
class TestTransformer:
    def test_pr_auc_above_minimum(self, test_df, transformer_probs):
        pr_auc = average_precision_score(test_df["is_fraud"], transformer_probs)
        assert pr_auc > 0.20, f"Transformer PR-AUC dropped to {pr_auc:.4f}"

    def test_no_nan_predictions(self, transformer_probs):
        assert not np.any(np.isnan(transformer_probs))

    def test_predictions_in_range(self, transformer_probs):
        assert np.all(transformer_probs >= 0) and np.all(transformer_probs <= 1)


class TestDataQuality:
    def test_fraud_rate_realistic(self, test_df):
        rate = test_df["is_fraud"].mean()
        assert 0.001 < rate < 0.05, f"Fraud rate {rate:.4f} outside realistic range"

    def test_all_typologies_present(self, test_df):
        fraud_types = test_df[test_df["is_fraud"] == 1]["fraud_type"].unique()
        for t in ["card_testing", "account_takeover", "velocity_abuse"]:
            assert t in fraud_types, f"Missing fraud typology: {t}"

    def test_no_missing_features(self, test_df):
        for col in ["user_id", "merchant_id", "amount", "timestamp", "is_fraud"]:
            assert test_df[col].isna().sum() == 0, f"Column {col} has missing values"