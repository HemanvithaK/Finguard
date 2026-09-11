# src/evals/llm_judge.py
import os
import sys
import json
import pandas as pd
from pathlib import Path
from anthropic import Anthropic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src" / "agents"))
sys.path.append(str(PROJECT_ROOT / "src" / "models"))
sys.path.append(str(PROJECT_ROOT / "src" / "data_gen"))

from investigation_agent import investigate_transaction

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def judge_investigation(result: dict, actual_fraud_type: str) -> dict:
    """
    Uses an LLM to score the agent's investigation output on four criteria.
    This is NOT the same LLM that generated the output — in production
    you'd ideally use a different model to avoid self-bias, but for a
    portfolio project, same model with a strict rubric is acceptable.
    """
    prompt = f"""You are a senior fraud analyst evaluating the quality of an 
automated investigation report. Score the report on these four criteria, 
each from 1-5:

1. EVIDENCE_COMPLETENESS: Did the report consider all available evidence 
   (amount, device, location, merchant, velocity)? 
   5 = all factors addressed, 1 = major factors ignored

2. REASONING_COHERENCE: Is the reasoning logical and well-structured? 
   Does the conclusion follow from the evidence?
   5 = clear logical chain, 1 = contradictory or confused

3. TYPOLOGY_IDENTIFICATION: The actual fraud type was "{actual_fraud_type}". 
   Did the report correctly identify or hint at this pattern?
   5 = correctly identified, 3 = partially identified, 1 = completely wrong

4. DECISION_APPROPRIATENESS: Was the final decision (auto_block/auto_approve/escalate) 
   appropriate given the evidence?
   5 = exactly right, 3 = reasonable but not optimal, 1 = dangerous decision

Here is the investigation output to evaluate:

EVIDENCE SUMMARY:
{result['evidence_summary']}

DISPUTE REASONING:
{result['dispute_reasoning']}

DECISION: {result['decision']} (confidence: {result['confidence']})
ACTUAL FRAUD TYPE: {actual_fraud_type}

Respond in EXACTLY this JSON format and nothing else:
{{
    "evidence_completeness": <1-5>,
    "reasoning_coherence": <1-5>,
    "typology_identification": <1-5>,
    "decision_appropriateness": <1-5>,
    "overall_notes": "<one sentence summary of strengths and weaknesses>"
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Parse JSON from the response — handle possible markdown fencing
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        scores = {
            "evidence_completeness": 0,
            "reasoning_coherence": 0,
            "typology_identification": 0,
            "decision_appropriateness": 0,
            "overall_notes": f"Failed to parse LLM judge response: {text[:200]}",
        }

    return scores


def run_evaluation(n_samples=5):
    """
    Run the agent on N flagged fraud transactions, judge each one,
    and print a summary scorecard.
    """
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "transactions.csv")
    fraud_txns = df[df["is_fraud"] == 1].sample(n=n_samples, random_state=42)

    all_scores = []

    for _, txn in fraud_txns.iterrows():
        txn_id = txn["transaction_id"]
        fraud_type = txn["fraud_type"]

        print(f"\n{'='*60}")
        print(f"Investigating: {txn_id} (actual: {fraud_type})")
        print(f"{'='*60}")

        # Run the agent
        result = investigate_transaction(txn_id, fraud_probability=0.92)
        print(f"Decision: {result['decision']} ({result['confidence']})")

        # Judge the output
        scores = judge_investigation(result, fraud_type)
        scores["transaction_id"] = txn_id
        scores["fraud_type"] = fraud_type
        scores["agent_decision"] = result["decision"]
        all_scores.append(scores)

        print(f"Scores: EC={scores['evidence_completeness']} "
              f"RC={scores['reasoning_coherence']} "
              f"TI={scores['typology_identification']} "
              f"DA={scores['decision_appropriateness']}")
        print(f"Notes: {scores['overall_notes']}")

    # Summary scorecard
    scores_df = pd.DataFrame(all_scores)
    print(f"\n{'='*60}")
    print("  LLM-AS-JUDGE SCORECARD")
    print(f"{'='*60}")
    print(f"Samples evaluated: {n_samples}")
    print(f"\nAverage scores (out of 5):")
    for col in ["evidence_completeness", "reasoning_coherence",
                "typology_identification", "decision_appropriateness"]:
        avg = scores_df[col].mean()
        print(f"  {col}: {avg:.1f}")
    print(f"\nOverall average: {scores_df[['evidence_completeness', 'reasoning_coherence', 'typology_identification', 'decision_appropriateness']].mean().mean():.1f}")

    print(f"\nDecision distribution:")
    print(scores_df["agent_decision"].value_counts().to_string())

    print(f"\nPer-typology scores:")
    typology_avg = scores_df.groupby("fraud_type")[
        ["evidence_completeness", "reasoning_coherence",
         "typology_identification", "decision_appropriateness"]
    ].mean()
    print(typology_avg.to_string())

    # Save results
    scores_df.to_csv(PROJECT_ROOT / "data" / "processed" / "llm_judge_results.csv", index=False)
    print(f"\nDetailed results saved to data/processed/llm_judge_results.csv")

    return scores_df


if __name__ == "__main__":
    run_evaluation(n_samples=5)