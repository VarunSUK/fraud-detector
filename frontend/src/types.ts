export interface Transaction {
  time: number;
  amount: number;
  transaction_id?: string;
  [key: `v${number}`]: number | string | undefined;
}

export interface AccountContext {
  credit_limit: number;
  current_balance: number;
  account_age_days: number;
  delinquent_payments_count: number;
  avg_monthly_spend: number;
}

export interface FeatureContribution {
  feature: string;
  value: number;
  importance: number;
  contribution: number;
}

export interface ScoreResponse {
  transaction_id: string;
  score: number;
  prediction: number;
  probability: number;
  model: string;
  timestamp: string;
  processing_ms: number;
}

export interface CreditLimitRecommendation {
  current: number;
  recommended: number;
  adjustment_pct: number;
}

export type DecisionAction = "approve" | "step_up_review" | "decline";
export type RiskTier = "low" | "medium" | "high";

export interface DecisionResponse {
  transaction_id: string;
  fraud_score: number;
  action: DecisionAction;
  risk_tier: RiskTier;
  reason_codes: string[];
  credit_limit_recommendation: CreditLimitRecommendation;
  narrative: string;
  feature_contributions: FeatureContribution[];
  model_scores: Record<string, number>;
  timestamp: string;
  processing_ms: number;
}

export interface Case {
  id: number;
  transaction_id: string;
  created_at: number;
  amount: number;
  fraud_score: number;
  action: string;
  risk_tier: string;
  reason_codes: string[];
  model_scores: Record<string, number>;
  credit_limit_current: number;
  credit_limit_recommended: number;
  analyst_verdict: string | null;
  is_actual_fraud: boolean | null;
}

export interface CasesResponse {
  cases: Case[];
}

export interface FunnelRow {
  action: string;
  transaction_count: number;
  pct_of_volume: number;
  total_amount: number;
  avg_fraud_score: number;
}

export interface ScoreDecileRow {
  score_decile: number;
  transaction_count: number;
  confirmed_fraud_count: number;
  fraud_rate_pct: number;
}

export interface AnalyticsSummary {
  funnel: FunnelRow[];
  score_deciles: ScoreDecileRow[];
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  models: string[];
}

export interface ApiError {
  error: string;
  message?: string;
  timestamp: string;
}
