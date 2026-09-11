# src/agents/run_agent.py
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))

from investigation_agent import investigate_transaction


def main():
    # Pick a known fraud transaction from the dataset to test with
    import pandas as pd
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "transactions.csv")
    fraud_txns = df[df["is_fraud"] == 1].head(5)

    print("Testing agent on flagged transactions:\n")

    for _, txn in fraud_txns.iterrows():
        txn_id = txn["transaction_id"]
        print(f"\n{'='*70}")
        print(f"Investigating: {txn_id} (actual: FRAUD - {txn['fraud_type']})")
        print(f"{'='*70}")

        result = investigate_transaction(txn_id, fraud_probability=0.92)

        print(f"\nEvidence Summary:\n{result['evidence_summary']}")
        print(f"\nDispute Reasoning:\n{result['dispute_reasoning']}")
        print(f"\nDecision: {result['decision']} (confidence: {result['confidence']})")

        # Only run one for now to check it works, then uncomment the loop
        break


if __name__ == "__main__":
    main()