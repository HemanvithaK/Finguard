# src/models/train_baseline.py
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    classification_report, roc_auc_score
)

FEATURE_COLS = [
    "amount", "time_since_prev_txn", "txns_last_10min",
    "dist_from_home_km", "amount_vs_avg_ratio", "is_known_device",
    "distinct_merchants_1hr", "hour_of_day"
]


def time_based_split(df, split_date="2026-03-15"):
    """Train on everything before split_date, test on everything after —
    mimics real deployment: predict future transactions using only past data."""
    train = df[df["timestamp"] < split_date]
    test = df[df["timestamp"] >= split_date]
    return train, test


def train_lightgbm(train_df, test_df):
    X_train, y_train = train_df[FEATURE_COLS], train_df["is_fraud"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["is_fraud"]

    # scale_pos_weight tells LightGBM "fraud is rare, weight mistakes on it more heavily"
    # — the standard, principled way to handle imbalance in gradient boosting,
    # instead of naive oversampling which can create unrealistic duplicate patterns
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    print("PR-AUC:", average_precision_score(y_test, probs))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, probs > 0.5))

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances:\n", importances)

    return model, probs


def per_typology_recall(test_df, probs, threshold=0.5):
    """Check recall broken down by fraud typology — aggregate recall
    can hide a typology the model is bad at, especially rare ones
    like account_takeover."""
    results = test_df.copy()
    results["predicted_fraud"] = (probs > threshold).astype(int)

    fraud_only = results[results["is_fraud"] == 1]
    breakdown = fraud_only.groupby("fraud_type").apply(
        lambda g: pd.Series({
            "n_cases": len(g),
            "caught": g["predicted_fraud"].sum(),
            "recall": g["predicted_fraud"].mean()
        })
    )
    print("\nPer-typology recall:\n", breakdown)