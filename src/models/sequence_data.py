import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_FEATURES = [
    "amount", "time_since_prev_txn", "dist_from_home_km",
    "amount_vs_avg_ratio", "is_known_device", "hour_of_day",
    "txns_last_10min", "distinct_merchants_1hr"  # added back based on error analysis
]
MAX_SEQ_LEN = 25

class TransactionSequenceDataset(Dataset):
    def __init__(self, df):
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        self.sequences = []
        self.labels = []
        self.fraud_types = []  # New: store fraud_type for per-typology recall

        for user_id, group in df.groupby("user_id"):
            features = group[SEQ_FEATURES].values.astype(np.float32)
            fraud_labels = group["is_fraud"].values
            fraud_types = group["fraud_type"].values  # New: get fraud_type

            for i in range(len(group)):
                start = max(0, i - MAX_SEQ_LEN + 1)
                window = features[start:i+1]

                pad_len = MAX_SEQ_LEN - len(window)
                if pad_len > 0:
                    padding = np.zeros((pad_len, len(SEQ_FEATURES)), dtype=np.float32)
                    window = np.vstack([padding, window])

                self.sequences.append(window)
                self.labels.append(fraud_labels[i])
                self.fraud_types.append(fraud_types[i])  # New: store fraud_type

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )