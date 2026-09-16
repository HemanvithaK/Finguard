# src/api/drift_detector.py
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from collections import deque


class DriftDetector:
    """
    Monitors incoming transaction features for distribution shift
    relative to the training data baseline.
    
    Uses PSI (Population Stability Index) — the industry-standard
    metric for feature drift in financial ML.
    PSI < 0.1: no drift
    0.1-0.2: moderate drift, monitor
    > 0.2: significant drift, investigate
    """

    def __init__(self, feature_names, window_size=1000, baseline_path=None):
        self.feature_names = feature_names
        self.window_size = window_size
        self.recent_features = {f: deque(maxlen=window_size) for f in feature_names}
        self.recent_predictions = deque(maxlen=window_size)
        self.baseline_stats = {}
        self.baseline_bins = {}
        self.alerts = deque(maxlen=100)

        if baseline_path and Path(baseline_path).exists():
            self.load_baseline(baseline_path)

    def compute_baseline(self, train_df, feature_names, n_bins=10):
        for feature in feature_names:
            values = train_df[feature].dropna().values
            self.baseline_stats[feature] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(values, percentiles)
            bin_edges = np.unique(bin_edges)
            counts, _ = np.histogram(values, bins=bin_edges)
            frequencies = counts / counts.sum()
            frequencies = np.clip(frequencies, 0.001, None)
            self.baseline_bins[feature] = {
                "edges": bin_edges.tolist(),
                "frequencies": frequencies.tolist(),
            }

    def save_baseline(self, path):
        data = {
            "stats": self.baseline_stats,
            "bins": self.baseline_bins,
            "computed_at": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_baseline(self, path):
        with open(path) as f:
            data = json.load(f)
        self.baseline_stats = data["stats"]
        self.baseline_bins = data["bins"]

    def record(self, features: dict, prediction: float):
        for f in self.feature_names:
            if f in features:
                self.recent_features[f].append(features[f])
        self.recent_predictions.append(prediction)

    def _compute_psi(self, feature_name) -> float:
        if feature_name not in self.baseline_bins:
            return 0.0
        recent_values = np.array(self.recent_features[feature_name])
        if len(recent_values) < 100:
            return 0.0

        baseline = self.baseline_bins[feature_name]
        edges = np.array(baseline["edges"])
        expected_freq = np.array(baseline["frequencies"])

        counts, _ = np.histogram(recent_values, bins=edges)
        actual_freq = counts / counts.sum()
        actual_freq = np.clip(actual_freq, 0.001, None)

        psi = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
        return float(psi)

    def check_drift(self) -> dict:
        report = {
            "timestamp": datetime.now().isoformat(),
            "samples_in_window": len(self.recent_predictions),
            "feature_drift": {},
            "prediction_drift": {},
            "alerts": [],
        }

        for feature in self.feature_names:
            psi = self._compute_psi(feature)
            status = "ok" if psi < 0.1 else "warning" if psi < 0.2 else "critical"
            report["feature_drift"][feature] = {"psi": round(psi, 4), "status": status}
            if status != "ok":
                alert = f"Feature '{feature}' drift: PSI={psi:.4f} ({status})"
                report["alerts"].append(alert)

        if len(self.recent_predictions) >= 100:
            recent_preds = np.array(self.recent_predictions)
            flag_rate = (recent_preds > 0.5).mean()
            mean_prob = recent_preds.mean()
            report["prediction_drift"] = {
                "mean_fraud_probability": round(float(mean_prob), 4),
                "flag_rate": round(float(flag_rate), 4),
                "status": "ok" if flag_rate < 0.05 else "warning" if flag_rate < 0.10 else "critical",
            }
            if report["prediction_drift"]["status"] != "ok":
                alert = f"Prediction drift: flag_rate={flag_rate:.4f} (expected <0.05)"
                report["alerts"].append(alert)

        return report