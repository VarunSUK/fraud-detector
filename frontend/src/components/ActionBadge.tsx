import type { DecisionAction, RiskTier } from "../types";

const ACTION_LABELS: Record<DecisionAction, string> = {
  approve: "Approved",
  step_up_review: "Manual Review",
  decline: "Declined",
};

export function ActionBadge({ action }: { action: DecisionAction }) {
  return <span className={`badge badge-${action}`}>{ACTION_LABELS[action] ?? action}</span>;
}

export function RiskBadge({ tier }: { tier: RiskTier }) {
  return <span className={`badge badge-risk-${tier}`}>{tier} risk</span>;
}
