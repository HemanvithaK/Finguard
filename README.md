Here's a clean, short README that covers everything without being overwhelming:

```markdown
# FinGuard — Fraud Detection & Dispute Resolution

End-to-end fraud detection system that scores transactions in ~12ms, investigates flagged cases with an LLM agent, and catches 88.5% of fraud with 55% fewer false alarms than a single-model baseline.

## Architecture

```
Transaction → Feature Store (live velocity) → LightGBM + Transformer → Stacking Ensemble → fraud score
                                                                                              ↓ flagged?
User graph → GNN → card-level risk                              LangGraph Agent → evidence → reasoning → decision
```

## Detection Models

| Model | Task | Key Result |
|-------|------|------------|
| LightGBM | Per-transaction scoring | PR-AUC 0.94 |
| Transformer | Sequence patterns | PR-AUC 0.74 (error analysis → targeted fix) |
| GNN (GraphSAGE) | Card-level fraud | PR-AUC 0.89 (data leak identified + fixed) |
| **Stacking Ensemble** | **Combined scoring** | **F1 0.904, 55% fewer false positives** |

## Investigation Agent

Four-node LangGraph pipeline: gather evidence → analyze risk → draft dispute reasoning → auto_block / auto_approve / escalate. Scored 4.1/5 by LLM-as-judge evaluation.

## Production Infrastructure

- **Feature Store** — SQLite-based real-time velocity features (closed train-serve gap: same txn scored 0% → 28%)
- **Drift Detection** — PSI-based per-feature monitoring with configurable alerts
- **A/B Testing** — 80/20 traffic split with live metric comparison (caught transformer calibration bug)
- **API** — FastAPI: `/predict` (~12ms), `/investigate`, `/drift`, `/ab-results`, `/metrics`
- **Deployment** — Docker → ECR → ECS Fargate → ALB, provisioned via Terraform (18 resources)
- **CI/CD** — GitHub Actions running pytest regression tests on every push

## Quick Start

```bash
# Local
pip install -r requirements.txt
cd src/data_gen && python build_dataset.py
cd ../models && python run_baseline.py
cd ../.. && python -m uvicorn src.api.app:app --port 8000

# Docker
docker build -t finguard .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key finguard

# AWS
cd terraform && terraform init && terraform apply
```

## Tech Stack

Python · PyTorch · LightGBM · PyTorch Geometric · LangGraph · Claude API · FastAPI · Docker · AWS (ECS/ECR/ALB) · Terraform · Prometheus · Pytest · GitHub Actions
