# src/models/transformer_model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Transformers have no inherent sense of sequence order (unlike RNNs) —
    positional encoding injects that information by adding a unique,
    deterministic sinusoidal pattern to each position in the sequence.
    """
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # not a trainable parameter

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FraudSequenceTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        # Project raw features (6-dim) up into d_model-dim space —
        # transformers need a consistent internal dimension to work in
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Binary classification head: fraud probability for the LAST
        # position in the sequence (the transaction we're predicting on)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)          # (batch, seq_len, d_model)
        last_position = x[:, -1, :]      # take only the final (most recent) transaction
        logits = self.classifier(last_position).squeeze(-1)
        return logits  # raw logits, not probabilities — loss function handles sigmoid