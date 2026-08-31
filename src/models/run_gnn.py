# src/models/run_gnn.py
import pandas as pd
import torch

from graph_data import construct_graph
from train_gnn import train_gnn


def main():
    transactions_df = pd.read_csv("../../data/raw/transactions.csv", parse_dates=["timestamp"])
    graph = construct_graph(transactions_df)
    model, probs, labels = train_gnn(graph, epochs=100)

    torch.save(model.state_dict(), "../../data/processed/gnn_baseline.pt")
    print("\nModel saved to data/processed/gnn_baseline.pt")


if __name__ == "__main__":
    main()