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

from eval_models import load_golden_test, eval_lightgbm, eval_transformer
from sklearn.metrics import average_precision_score


@pytest.fixture(scope="session")
def test_df():
    """Load golden test set once for all tests."""
    return load_golden_test()


@pytest.fixture(scope="session")
def lgbm_probs(test_df):
    """Run LightGBM predictions once for all tests."""
    return eval_lightgbm(test_df)


@pytest.fixture(scope="session")
def transformer_probs(test_df):
    """Run transformer predictions once for all tests."""
    return eval_transformer(test_df)


class TestLightGBM:
    """Regression tests for LightGBM baseline."""

    def test_pr_auc_above_minimum(self, test_df, lgbm_probs):
        """PR-AUC should never drop below 0.80 — our baseline hits 0.94."""
        pr_auc = average_precision_score(test_df["is_fraud"], lgbm_probs)
        assert pr_auc > 0.80, f"LightGBM PR-AUC dropped to {pr_auc:.4f} (minimum: 0.80)"

    def test_recall_above_minimum(self, test_df, lgbm_probs):
        """At default threshold, recall should stay above 0.70."""
        preds = (lgbm_probs > 0.5).astype(int)
        fraud_mask = test_df["is_fraud"] == 1
        recall = preds[fraud_mask.values].mean()
        assert recall >= 0.70, f"LightGBM recall dropped to {recall:.3f} (minimum: 0.70)"

    def test_no_nan_predictions(self, lgbm_probs):
        """Model should never produce NaN predictions."""
        assert not np.any(np.isnan(lgbm_probs)), "LightGBM produced NaN predictions"

    def test_predictions_in_range(self, lgbm_probs):
        """Probabilities should be between 0 and 1."""
        assert np.all(lgbm_probs >= 0) and np.all(lgbm_probs <= 1), \
            "LightGBM predictions outside [0, 1] range"


class TestTransformer:
    """Regression tests for the sequence transformer."""

    def test_pr_auc_above_minimum(self, test_df, transformer_probs):
        """PR-AUC should never drop below 0.20 — transformer is weaker
        than LightGBM and sensitive to data distribution shifts between
        training data and golden test set."""
        pr_auc = average_precision_score(test_df["is_fraud"], transformer_probs)
        assert pr_auc > 0.20, f"Transformer PR-AUC dropped to {pr_auc:.4f} (minimum: 0.20)"

    def test_no_nan_predictions(self, transformer_probs):
        """Model should never produce NaN predictions."""
        assert not np.any(np.isnan(transformer_probs)), "Transformer produced NaN predictions"

    def test_predictions_in_range(self, transformer_probs):
        """Probabilities should be between 0 and 1."""
        assert np.all(transformer_probs >= 0) and np.all(transformer_probs <= 1), \
            "Transformer predictions outside [0, 1] range"


class TestDataQuality:
    """Tests that the golden test set itself is valid."""

    def test_fraud_rate_realistic(self, test_df):
        """Fraud rate should be between 0.1% and 5%."""
        rate = test_df["is_fraud"].mean()
        assert 0.001 < rate < 0.05, f"Fraud rate {rate:.4f} outside realistic range"

    def test_all_typologies_present(self, test_df):
        """All three fraud typologies should be represented."""
        fraud_types = test_df[test_df["is_fraud"] == 1]["fraud_type"].unique()
        for t in ["card_testing", "account_takeover", "velocity_abuse"]:
            assert t in fraud_types, f"Missing fraud typology: {t}"

    def test_no_missing_features(self, test_df):
        """Critical columns should have no nulls."""
        for col in ["user_id", "merchant_id", "amount", "timestamp", "is_fraud"]:
            assert test_df[col].isna().sum() == 0, f"Column {col} has missing values"