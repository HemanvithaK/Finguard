# FinGuard — Fraud Detection & Dispute Resolution

End-to-end fraud detection system that scores transactions in ~12ms, investigates flagged cases with an LLM agent, and catches 88.5% of fraud with 55% fewer false alarms than a single-model baseline.

## Architecture

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Docker](https://img.shields.io/badge/Docker-containerized-blue)
![AWS](https://img.shields.io/badge/AWS-ECS%20Fargate-orange)
![Terraform](https://img.shields.io/badge/IaC-Terraform-purple)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen)

**Per-transaction scoring (real-time, ~12ms):**
Transaction → Feature Store → LightGBM + Transformer → Stacking Ensemble → fraud score

**Card-level monitoring (background):**
User-Merchant-Device graph → GNN → card-level risk score

**Investigation (flagged transactions only):**
Flagged txn → LangGraph Agent → evidence gathering → risk analysis → dispute reasoning → decision

## Detection Models

| Model | Task | Key Result |
|-------|------|------------|
| LightGBM | Per-transaction scoring | PR-AUC 0.94 |
| Transformer | Sequence patterns | PR-AUC 0.74 (error analysis → targeted fix) |
| GNN (GraphSAGE) | Card-level fraud | PR-AUC 0.89 (data leak identified + fixed) |
| **Stacking Ensemble** | **Combined scoring** | **F1 0.904 · 55% fewer false positives** |

## Investigation Agent

Four-node LangGraph pipeline that investigates flagged transactions:
1. **Gather Evidence** — user history, merchant profile, risk signals (no LLM, ~5ms)
2. **Analyze Risk** — structured assessment across 5 dimensions
3. **Draft Reasoning** — formal report with supporting + contradicting evidence
4. **Decide** — auto_block / auto_approve / escalate

Evaluated via LLM-as-judge: **4.1/5 overall** · card_testing 4.5/5 · velocity_abuse 2.3/5

## Production Infrastructure

| Component | What It Does |
|-----------|-------------|
| Feature Store | Real-time velocity features via SQLite (closed train-serve gap) |
| Drift Detection | PSI-based per-feature monitoring with alerts |
| A/B Testing | 80/20 traffic split with live metric comparison |
| API | FastAPI: `/predict`, `/investigate`, `/drift`, `/ab-results`, `/metrics` |
| Deployment | Docker → ECR → ECS Fargate → ALB via Terraform |
| CI/CD | GitHub Actions · pytest regression tests on every push |
| Monitoring | Prometheus metrics endpoint + Grafana dashboard config |

## Key Technical Decisions

**Caught data leakage twice.** First run hit 100% metrics — feature importance revealed zero class overlap in synthetic data. GNN node features included post-fraud transactions — fixing it improved PR-AUC from 0.85 to 0.89.

**Error-analysis-driven improvement.** Transformer missed early-burst velocity transactions (25,976s gap for missed vs 624s for caught). Extended sequence window 10→25 and added velocity features. PR-AUC 0.71→0.74.

**Platt scaling for deployment.** WeightedRandomSampler calibrated transformer to 50/50 world. Platt scaling reduced false positives 97% (3,270→88).

**Train-serve consistency.** Same transaction scored 0% with hardcoded defaults vs 28% with live feature store — velocity signal was invisible without it.
```

## Tech Stack

Python · PyTorch · LightGBM · PyTorch Geometric · LangGraph · Claude API · FastAPI · Docker · AWS (ECS/ECR/ALB) · Terraform · Prometheus · Pytest · GitHub Actions