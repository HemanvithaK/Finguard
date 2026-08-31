# src/models/train_gnn.py
import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report

from gnn_model import FraudGNN


def train_gnn(data, epochs=100, hidden_channels=64, lr=1e-2, test_frac=0.25, seed=42):
    """
    Full-batch training on the whole graph — it's small enough that we
    don't need neighbor sampling (which would also require extra native
    dependencies that are painful on Windows).

    We split CARD NODES into train/test: loss is computed only on training
    cards; evaluation happens on held-out cards the model never saw labels for.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    data = data.to(device)

    # ---- Train/test masks over card nodes ----
    n_cards = data["card"].num_nodes
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_cards, generator=g)
    n_test = int(n_cards * test_frac)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    y = data["card"].y

    # ---- Class imbalance: dampened pos_weight (sqrt), same as transformer ----
    n_pos = y[train_idx].sum().item()
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([max(1.0, (n_neg / max(n_pos, 1)) ** 0.5)]).to(device)
    print(f"Train cards: {len(train_idx)} ({int(n_pos)} fraud) | "
          f"Test cards: {len(test_idx)} | pos_weight: {pos_weight.item():.2f}")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = FraudGNN(hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = criterion(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - loss: {loss.item():.4f}")

    # ---- Evaluation on held-out cards ----
    model.eval()
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        probs = torch.sigmoid(logits[test_idx]).cpu().numpy()
        labels = y[test_idx].cpu().numpy()

    print("\nPR-AUC:", average_precision_score(labels, probs))
    print("ROC-AUC:", roc_auc_score(labels, probs))
    print(classification_report(labels, probs > 0.5, zero_division=0))

    return model, probs, labels