# src/models/gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class FraudGNN(torch.nn.Module):
    """
    Two-layer heterogeneous GraphSAGE. Each layer passes messages along
    every edge type (including reverse edges), summing results per node
    type. Final linear head scores 'card' nodes for fraud.
    """

    def __init__(self, hidden_channels=64):
        super().__init__()

        self.conv1 = HeteroConv({
            ("card", "transaction", "merchant"): SAGEConv((-1, -1), hidden_channels),
            ("card", "used_by", "device"): SAGEConv((-1, -1), hidden_channels),
            ("merchant", "rev_transaction", "card"): SAGEConv((-1, -1), hidden_channels),
            ("device", "rev_used_by", "card"): SAGEConv((-1, -1), hidden_channels),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("card", "transaction", "merchant"): SAGEConv((-1, -1), hidden_channels),
            ("card", "used_by", "device"): SAGEConv((-1, -1), hidden_channels),
            ("merchant", "rev_transaction", "card"): SAGEConv((-1, -1), hidden_channels),
            ("device", "rev_used_by", "card"): SAGEConv((-1, -1), hidden_channels),
        }, aggr="sum")

        self.lin = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return self.lin(x_dict["card"]).squeeze(-1)  # logits per card