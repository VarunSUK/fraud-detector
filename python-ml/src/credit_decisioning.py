"""
Credit risk decisioning policy.

Turns a fraud/anomaly score plus account context into an actual action,
credit-limit signal, and reason codes -- the policy layer a credit risk
strategy team owns end to end (as opposed to a bare fraud probability, which
nobody can act on directly).

The score cutoffs below are reasonable starting points for a card-risk
policy, not tuned constants -- see analytics/sql/threshold_tradeoff.sql for
how a real policy change would be sized and pressure-tested against
historical outcomes before being pushed.
"""

from dataclasses import dataclass
from typing import List

DECLINE_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.45
LARGE_TRANSACTION_AMOUNT = 3000.0
HIGH_UTILIZATION = 0.85
ANOMALY_FLAG_THRESHOLD = 0.7

DELINQUENCY_THRESHOLD_TIGHTEN = 0.15
LIMIT_TIGHTEN_FACTOR = 0.7
LIMIT_GROWTH_FACTOR = 1.15


@dataclass
class AccountContext:
    credit_limit: float = 0.0
    current_balance: float = 0.0
    account_age_days: int = 0
    delinquent_payments_count: int = 0
    avg_monthly_spend: float = 0.0

    @property
    def utilization(self) -> float:
        if self.credit_limit <= 0:
            return 0.0
        return self.current_balance / self.credit_limit


@dataclass
class Decision:
    action: str  # "approve" | "step_up_review" | "decline"
    risk_tier: str  # "low" | "medium" | "high"
    reason_codes: List[str]
    credit_limit_current: float
    credit_limit_recommended: float
    credit_limit_adjustment_pct: float


def _recommend_credit_limit(fraud_score: float, account: AccountContext) -> float:
    """Simple, explainable limit-adjustment policy: tighten meaningfully on
    elevated risk or delinquency, hold steady pending review, grow only for
    accounts that combine low risk with high (but responsible) utilization."""
    limit = account.credit_limit
    if limit <= 0:
        return limit

    if account.delinquent_payments_count > 0 or fraud_score >= DECLINE_THRESHOLD:
        return round(limit * LIMIT_TIGHTEN_FACTOR, 2)

    if fraud_score >= REVIEW_THRESHOLD:
        return limit

    if account.utilization >= HIGH_UTILIZATION:
        return round(limit * LIMIT_GROWTH_FACTOR, 2)

    return limit


def decide(
    fraud_score: float,
    amount: float,
    account: AccountContext,
    anomaly_score: float = 0.0,
) -> Decision:
    """Applies the card-risk policy to a scored transaction."""
    reason_codes: List[str] = []

    decline_threshold = DECLINE_THRESHOLD
    review_threshold = REVIEW_THRESHOLD

    # Large transactions get a lower bar for review, consistent with
    # step-up authentication practices in real card risk policy.
    if amount >= LARGE_TRANSACTION_AMOUNT:
        review_threshold -= 0.15
        reason_codes.append("LARGE_TRANSACTION_AMOUNT")

    if account.delinquent_payments_count > 0:
        decline_threshold -= DELINQUENCY_THRESHOLD_TIGHTEN
        review_threshold -= DELINQUENCY_THRESHOLD_TIGHTEN
        reason_codes.append("RECENT_DELINQUENCY")

    if anomaly_score >= ANOMALY_FLAG_THRESHOLD:
        reason_codes.append("ANOMALY_DETECTED")

    if fraud_score >= decline_threshold:
        action = "decline"
        risk_tier = "high"
        reason_codes.append("HIGH_FRAUD_SCORE")
    elif fraud_score >= review_threshold:
        action = "step_up_review"
        risk_tier = "medium"
        reason_codes.append("ELEVATED_FRAUD_SCORE")
    else:
        action = "approve"
        risk_tier = "low"

    if account.utilization >= HIGH_UTILIZATION:
        reason_codes.append("HIGH_UTILIZATION")

    credit_limit_recommended = _recommend_credit_limit(fraud_score, account)
    adjustment_pct = 0.0
    if account.credit_limit > 0:
        adjustment_pct = (
            (credit_limit_recommended - account.credit_limit) / account.credit_limit * 100
        )

    return Decision(
        action=action,
        risk_tier=risk_tier,
        reason_codes=reason_codes or ["WITHIN_POLICY"],
        credit_limit_current=account.credit_limit,
        credit_limit_recommended=credit_limit_recommended,
        credit_limit_adjustment_pct=round(adjustment_pct, 2),
    )
