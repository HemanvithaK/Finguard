# src/api/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# ---- Prometheus Metrics ----
# These are automatically scraped by Prometheus at /metrics

PREDICTIONS_TOTAL = Counter(
    "finguard_predictions_total",
    "Total number of fraud predictions",
    ["model", "is_flagged"]
)

PREDICTION_LATENCY = Histogram(
    "finguard_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

FRAUD_PROBABILITY = Histogram(
    "finguard_fraud_probability",
    "Distribution of fraud probability scores",
    ["model"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

INVESTIGATION_LATENCY = Histogram(
    "finguard_investigation_latency_seconds",
    "Investigation agent latency in seconds",
    buckets=[1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

INVESTIGATION_DECISIONS = Counter(
    "finguard_investigation_decisions_total",
    "Investigation decisions by type",
    ["decision", "confidence"]
)

DRIFT_PSI = Gauge(
    "finguard_drift_psi",
    "Current PSI drift score per feature",
    ["feature"]
)

ACTIVE_MODEL = Gauge(
    "finguard_active_model_requests",
    "Current request count per model in A/B test",
    ["model"]
)


def metrics_endpoint():
    """Returns Prometheus-formatted metrics for scraping."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )