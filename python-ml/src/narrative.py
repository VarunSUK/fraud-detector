"""
Deterministic, template-based case-narrative generator.

Produces an analyst-memo-style summary from a decision plus its SHAP feature
contributions -- the kind of narrative synthesis "AI fluency" in a credit
risk workflow is meant to speed up (turning a score + reason codes into
something a reviewer can read in five seconds instead of re-deriving).

This is intentionally NOT a hosted LLM call: a fraud-scoring API shouldn't
take on a network dependency on an LLM provider for language this templated
and structured. If a case ever needs open-ended synthesis beyond what a
template can express (e.g. summarizing an analyst's free-text notes across a
case history), replace the body of generate() with a real model call --
e.g. Anthropic's Messages API -- keeping the same signature so callers don't
change. That's the extension point.
"""

from typing import Dict, List


def generate(
    transaction_id: str,
    action: str,
    risk_tier: str,
    reason_codes: List[str],
    feature_contributions: List[Dict],
    fraud_score: float,
    top_n: int = 3,
) -> str:
    top = sorted(feature_contributions, key=lambda c: abs(c.get("contribution", 0)), reverse=True)[:top_n]
    driver_text = "; ".join(
        f"{c['feature']} ({'+' if c['contribution'] >= 0 else ''}{c['contribution']:.2f})" for c in top
    ) or "no dominant feature"

    reason_text = ", ".join(reason_codes) if reason_codes else "none"
    action_text = {
        "approve": "approved",
        "step_up_review": "routed to manual review",
        "decline": "declined",
    }.get(action, action)

    label = transaction_id or "unlabeled transaction"

    return (
        f"{label} scored {fraud_score:.2f} and was {action_text} ({risk_tier} risk). "
        f"Policy triggers: {reason_text}. Top model signals: {driver_text}."
    )
