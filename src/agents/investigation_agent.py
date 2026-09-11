# src/agents/investigation_agent.py
import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

from tools import get_user_history, get_merchant_info, get_transaction_details


# ---- State: what the agent accumulates as it investigates ----
class InvestigationState(TypedDict):
    transaction_id: str
    fraud_probability: float
    transaction_details: dict
    user_history: dict
    merchant_info: dict
    evidence_summary: str
    dispute_reasoning: str
    decision: str  # "auto_block", "auto_approve", or "escalate"
    confidence: str  # "high", "medium", "low"


# ---- LLM setup ----
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_tokens=1024,
)


# ---- Node 1: Gather evidence ----
def gather_evidence(state: InvestigationState) -> InvestigationState:
    """
    First step: pull transaction details, user history, and merchant info.
    No LLM needed here — just data retrieval.
    """
    txn_details = get_transaction_details(state["transaction_id"])
    txn_details["model_fraud_probability"] = state["fraud_probability"]

    user_history = get_user_history(txn_details["user_id"])
    merchant_info = get_merchant_info(txn_details["merchant_id"])

    return {
        **state,
        "transaction_details": txn_details,
        "user_history": user_history,
        "merchant_info": merchant_info,
    }


# ---- Node 2: Analyze and summarize evidence ----
def analyze_evidence(state: InvestigationState) -> InvestigationState:
    """
    Second step: LLM reads the raw evidence and produces a structured
    summary of the risk signals — what's normal, what's anomalous.
    """
    prompt = f"""You are a fraud analyst. Analyze the following evidence about a 
flagged transaction and summarize the key risk signals.

Transaction Details:
{state['transaction_details']}

User History:
{state['user_history']}

Merchant Info:
{state['merchant_info']}

Provide a concise evidence summary covering:
1. How this transaction compares to the user's normal behavior
2. Whether the device is recognized
3. Whether the merchant has elevated fraud rates
4. Any velocity or location anomalies
5. Overall risk assessment (high/medium/low)

Be specific — cite actual numbers from the evidence, not vague statements."""

    response = llm.invoke(prompt)
    return {**state, "evidence_summary": response.content}


# ---- Node 3: Draft dispute reasoning ----
def draft_reasoning(state: InvestigationState) -> InvestigationState:
    """
    Third step: LLM writes the formal dispute reasoning that would go
    to a human reviewer or into a case management system.
    """
    prompt = f"""You are a fraud analyst writing a formal investigation report.
Based on the following evidence summary, draft a clear, professional dispute 
reasoning document.

Evidence Summary:
{state['evidence_summary']}

Transaction Details:
{state['transaction_details']}

Model Fraud Probability: {state['fraud_probability']:.4f}

Your reasoning should:
1. State the conclusion (fraud or legitimate) with confidence level
2. List the supporting evidence points
3. Note any evidence that contradicts the conclusion
4. Recommend a specific action (block card, approve transaction, or escalate)

Keep it under 200 words. Be direct and specific."""

    response = llm.invoke(prompt)
    return {**state, "dispute_reasoning": response.content}


# ---- Node 4: Make decision ----
def make_decision(state: InvestigationState) -> InvestigationState:
    """
    Final step: LLM makes a structured decision based on all prior analysis.
    Returns a discrete action, not free text — this is what a downstream
    system would actually consume.
    """
    prompt = f"""You are a fraud decision engine. Based on the investigation below,
output EXACTLY one of these decisions:

- auto_block: clear fraud, block the card immediately
- auto_approve: clearly legitimate, approve the transaction
- escalate: ambiguous, needs human review

Also state your confidence: high, medium, or low.

Evidence Summary:
{state['evidence_summary']}

Dispute Reasoning:
{state['dispute_reasoning']}

Model Fraud Probability: {state['fraud_probability']:.4f}

Respond in EXACTLY this format (two lines, nothing else):
DECISION: <auto_block|auto_approve|escalate>
CONFIDENCE: <high|medium|low>"""

    response = llm.invoke(prompt)
    lines = response.content.strip().split("\n")

    decision = "escalate"  # safe default
    confidence = "low"

    for line in lines:
        line = line.strip()
        if line.startswith("DECISION:"):
            d = line.split(":")[1].strip().lower()
            if d in ("auto_block", "auto_approve", "escalate"):
                decision = d
        elif line.startswith("CONFIDENCE:"):
            c = line.split(":")[1].strip().lower()
            if c in ("high", "medium", "low"):
                confidence = c

    return {**state, "decision": decision, "confidence": confidence}


# ---- Build the graph ----
def build_investigation_graph():
    """
    Constructs the LangGraph workflow:
    gather_evidence -> analyze_evidence -> draft_reasoning -> make_decision
    """
    workflow = StateGraph(InvestigationState)

    workflow.add_node("gather_evidence", gather_evidence)
    workflow.add_node("analyze_evidence", analyze_evidence)
    workflow.add_node("draft_reasoning", draft_reasoning)
    workflow.add_node("make_decision", make_decision)

    workflow.set_entry_point("gather_evidence")
    workflow.add_edge("gather_evidence", "analyze_evidence")
    workflow.add_edge("analyze_evidence", "draft_reasoning")
    workflow.add_edge("draft_reasoning", "make_decision")
    workflow.add_edge("make_decision", END)

    return workflow.compile()


# ---- Run a single investigation ----
def investigate_transaction(transaction_id: str, fraud_probability: float) -> dict:
    """
    Entry point: takes a flagged transaction ID and its model-assigned
    fraud probability, runs the full investigation pipeline, returns
    the complete investigation state.
    """
    graph = build_investigation_graph()

    initial_state = InvestigationState(
        transaction_id=transaction_id,
        fraud_probability=fraud_probability,
        transaction_details={},
        user_history={},
        merchant_info={},
        evidence_summary="",
        dispute_reasoning="",
        decision="",
        confidence="",
    )

    result = graph.invoke(initial_state)
    return result