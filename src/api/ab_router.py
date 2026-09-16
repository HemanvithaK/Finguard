# src/api/ab_router.py
import random
import time
import numpy as np
from collections import deque
from datetime import datetime


class ABRouter:
    """
    Routes incoming prediction requests between two models (A and B),
    tracks per-model metrics, and reports live comparison results.
    
    In production, this would integrate with a feature flagging system
    (LaunchDarkly, Statsig) and log to a data warehouse. Here we keep
    metrics in memory for simplicity — the pattern is what matters.
    
    Traffic split is configurable: e.g., 80% model A / 20% model B
    lets you test the new model on a small slice of traffic before
    fully committing to it.
    """

    def __init__(self, model_a_name="LightGBM", model_b_name="Transformer", b_fraction=0.2):
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.b_fraction = b_fraction  # fraction of traffic sent to model B

        # Per-model metrics tracking
        self.metrics = {
            model_a_name: {
                "predictions": deque(maxlen=5000),
                "latencies": deque(maxlen=5000),
                "flag_count": 0,
                "total_count": 0,
            },
            model_b_name: {
                "predictions": deque(maxlen=5000),
                "latencies": deque(maxlen=5000),
                "flag_count": 0,
                "total_count": 0,
            },
        }
        self.started_at = datetime.now().isoformat()

    def route(self) -> str:
        """
        Decide which model handles this request.
        Returns the model name — caller uses this to pick the model.
        """
        if random.random() < self.b_fraction:
            return self.model_b_name
        return self.model_a_name

    def record(self, model_name: str, probability: float, latency_ms: float, is_flagged: bool):
        """Record one prediction's outcome for the assigned model."""
        m = self.metrics[model_name]
        m["predictions"].append(probability)
        m["latencies"].append(latency_ms)
        m["total_count"] += 1
        if is_flagged:
            m["flag_count"] += 1

    def get_results(self) -> dict:
        """
        Return live A/B comparison — flag rates, latency stats,
        and prediction distribution for both models.
        """
        results = {
            "started_at": self.started_at,
            "traffic_split": {
                self.model_a_name: f"{(1 - self.b_fraction) * 100:.0f}%",
                self.model_b_name: f"{self.b_fraction * 100:.0f}%",
            },
            "models": {},
        }

        for name, m in self.metrics.items():
            total = m["total_count"]
            if total == 0:
                results["models"][name] = {"total_predictions": 0, "status": "no data yet"}
                continue

            preds = np.array(m["predictions"])
            lats = np.array(m["latencies"])

            results["models"][name] = {
                "total_predictions": total,
                "flag_rate": round(m["flag_count"] / total, 4),
                "mean_probability": round(float(preds.mean()), 4),
                "median_probability": round(float(np.median(preds)), 4),
                "p95_probability": round(float(np.percentile(preds, 95)), 4),
                "mean_latency_ms": round(float(lats.mean()), 2),
                "p95_latency_ms": round(float(np.percentile(lats, 95)), 2),
                "max_latency_ms": round(float(lats.max()), 2),
            }

        return results