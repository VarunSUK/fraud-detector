import type { DecisionAction, DriftInterpretation, RiskTier } from "../types";

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

const DRIFT_LABELS: Record<DriftInterpretation, string> = {
  stable: "Stable",
  moderate_shift: "Moderate shift",
  significant_shift: "Significant shift",
  insufficient_data: "Insufficient data",
};

export function DriftBadge({ interpretation }: { interpretation: DriftInterpretation }) {
  return <span className={`badge badge-drift-${interpretation}`}>{DRIFT_LABELS[interpretation] ?? interpretation}</span>;
}
