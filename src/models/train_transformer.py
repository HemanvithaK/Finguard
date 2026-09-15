# src/models/train_transformer.py
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

from sequence_data import TransactionSequenceDataset, MAX_SEQ_LEN, SEQ_FEATURES
from transformer_model import FraudSequenceTransformer


def train_transformer(train_df, test_df, epochs=10, batch_size=256, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_dataset = TransactionSequenceDataset(train_df)
    test_dataset = TransactionSequenceDataset(test_df)

    train_labels_arr = np.array(train_dataset.labels)
    class_counts = np.bincount(train_labels_arr.astype(int))
    sample_weights = 1.0 / class_counts[train_labels_arr.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = FraudSequenceTransformer(n_features=len(SEQ_FEATURES)).to(device)

    pos_weight = torch.tensor([2.0]).to(device)
    print(f"Using pos_weight: {pos_weight.item():.2f}")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - avg loss: {avg_loss:.4f}")

    # ---- Evaluation (raw, uncalibrated) ----
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print("\n--- Before Calibration ---")
    print("PR-AUC:", average_precision_score(all_labels, all_probs))
    print("ROC-AUC:", roc_auc_score(all_labels, all_probs))
    print(classification_report(all_labels, all_probs > 0.5, zero_division=0))

    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    idx = np.argmin(np.abs(recalls - 0.95))
    print(f"At recall={recalls[idx]:.2f}: precision={precisions[idx]:.2f}, threshold={thresholds[idx]:.3f}")

    # ---- Platt Scaling: recalibrate probabilities ----
    # The WeightedRandomSampler made training batches ~50/50 fraud/normal,
    # but real data is 99.6% normal. Raw probabilities are therefore inflated —
    # the model thinks fraud is much more common than it actually is.
    # Platt scaling fits a logistic regression on (raw_probs -> true_labels)
    # using held-out training data to produce properly calibrated probabilities.
    print("\n--- Applying Platt Scaling ---")

    cal_dataset = TransactionSequenceDataset(train_df.tail(50000))
    cal_loader = DataLoader(cal_dataset, batch_size=256, shuffle=False)

    cal_probs, cal_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in cal_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            cal_probs.extend(probs)
            cal_labels.extend(y_batch.numpy())

    cal_probs = np.array(cal_probs).reshape(-1, 1)
    cal_labels = np.array(cal_labels)

    calibrator = LogisticRegression()
    calibrator.fit(cal_probs, cal_labels)

    # Recalibrate test predictions
    all_probs_calibrated = calibrator.predict_proba(all_probs.reshape(-1, 1))[:, 1]

    print("\n--- After Platt Scaling ---")
    print("PR-AUC:", average_precision_score(all_labels, all_probs_calibrated))
    print("ROC-AUC:", roc_auc_score(all_labels, all_probs_calibrated))

    # Find optimal threshold: best F1 score on calibrated probabilities
    precisions_cal, recalls_cal, thresholds_cal = precision_recall_curve(all_labels, all_probs_calibrated)
    f1_scores = 2 * (precisions_cal[:-1] * recalls_cal[:-1]) / (precisions_cal[:-1] + recalls_cal[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_cal[best_idx]

    print(f"\nOptimal threshold (best F1): {best_threshold:.4f}")
    print(f"At optimal threshold: precision={precisions_cal[best_idx]:.3f}, recall={recalls_cal[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f}")
    print(classification_report(all_labels, all_probs_calibrated > best_threshold, zero_division=0))

    # Compare: how many false positives now vs before?
    fp_before = (all_probs > 0.5).sum() - ((all_probs > 0.5) & (all_labels == 1)).sum()
    fp_after = (all_probs_calibrated > best_threshold).sum() - ((all_probs_calibrated > best_threshold) & (all_labels == 1)).sum()
    print(f"False positives before calibration (threshold=0.5): {fp_before}")
    print(f"False positives after calibration (optimal threshold): {fp_after}")

    # Per-typology recall at optimal threshold
    def per_typology_recall(labels, probs, threshold):
        preds = (probs > threshold).astype(int)
        results = pd.DataFrame({"true_label": labels, "pred_label": preds, "prob": probs})
        results["fraud_type"] = test_dataset.fraud_types

        fraud_only = results[results["true_label"] == 1]
        breakdown = fraud_only.groupby("fraud_type").apply(
            lambda g: pd.Series({
                "n_cases": len(g),
                "caught": g["pred_label"].sum(),
                "recall": g["pred_label"].mean()
            })
        )
        print(f"\nPer-typology recall at threshold = {threshold:.4f}")
        print(breakdown)

    per_typology_recall(all_labels, all_probs_calibrated, threshold=best_threshold)

    return model, all_probs_calibrated