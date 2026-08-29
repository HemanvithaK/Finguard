import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, precision_recall_curve
import numpy as np
import pandas as pd

from sequence_data import TransactionSequenceDataset, MAX_SEQ_LEN, SEQ_FEATURES
from transformer_model import FraudSequenceTransformer


def train_transformer(train_df, test_df, epochs=10, batch_size=256, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_dataset = TransactionSequenceDataset(train_df)  # Pass full DataFrame
    test_dataset = TransactionSequenceDataset(test_df)  # Pass full DataFrame

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

    print("\nPR-AUC:", average_precision_score(all_labels, all_probs))
    print("ROC-AUC:", roc_auc_score(all_labels, all_probs))
    print(classification_report(all_labels, all_probs > 0.5))

    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    idx = np.argmin(np.abs(recalls - 0.95))
    print(f"\nAt recall={recalls[idx]:.2f}: precision={precisions[idx]:.2f}, threshold={thresholds[idx]:.3f}")

    def per_typology_recall(labels, probs, threshold=0.5):
        preds = (probs > threshold).astype(int)
        results = pd.DataFrame({"true_label": labels, "pred_label": preds, "prob": probs})
        results["fraud_type"] = test_dataset.fraud_types  # Get fraud types from test dataset

        fraud_only = results[results["true_label"] == 1]
        breakdown = fraud_only.groupby("fraud_type").apply(
            lambda g: pd.Series({
                "n_cases": len(g),
                "caught": g["pred_label"].sum(),
                "recall": g["pred_label"].mean()
            })
        )
        print("\nPer-typology recall at threshold =", threshold)
        print(breakdown)

    per_typology_recall(all_labels, all_probs, threshold=0.5)

    return model, all_probs