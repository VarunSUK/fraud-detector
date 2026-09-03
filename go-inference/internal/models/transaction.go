package models

import (
	"encoding/json"
	"time"
)

// Transaction represents a credit card transaction
type Transaction struct {
	Time          float64   `json:"time" binding:"required"`
	V1            float64   `json:"v1"`
	V2            float64   `json:"v2"`
	V3            float64   `json:"v3"`
	V4            float64   `json:"v4"`
	V5            float64   `json:"v5"`
	V6            float64   `json:"v6"`
	V7            float64   `json:"v7"`
	V8            float64   `json:"v8"`
	V9            float64   `json:"v9"`
	V10           float64   `json:"v10"`
	V11           float64   `json:"v11"`
	V12           float64   `json:"v12"`
	V13           float64   `json:"v13"`
	V14           float64   `json:"v14"`
	V15           float64   `json:"v15"`
	V16           float64   `json:"v16"`
	V17           float64   `json:"v17"`
	V18           float64   `json:"v18"`
	V19           float64   `json:"v19"`
	V20           float64   `json:"v20"`
	V21           float64   `json:"v21"`
	V22           float64   `json:"v22"`
	V23           float64   `json:"v23"`
	V24           float64   `json:"v24"`
	V25           float64   `json:"v25"`
	V26           float64   `json:"v26"`
	V27           float64   `json:"v27"`
	V28           float64   `json:"v28"`
	Amount        float64   `json:"amount" binding:"required"`
	Class         int       `json:"class,omitempty"` // 0 = normal, 1 = fraud
	TransactionID string    `json:"transaction_id,omitempty"`
	Timestamp     time.Time `json:"timestamp,omitempty"`
}

// SyntheticTransaction represents a synthetic transaction (for compatibility)
type SyntheticTransaction struct {
	Amount               float64   `json:"amount" binding:"required"`
	Merchant             string    `json:"merchant" binding:"required"`
	CardType             string    `json:"card_type" binding:"required"`
	Hour                 int       `json:"hour" binding:"required"`
	DayOfWeek            int       `json:"day_of_week" binding:"required"`
	IsWeekend            bool      `json:"is_weekend"`
	PreviousTransactions int       `json:"previous_transactions"`
	AvgAmount            float64   `json:"avg_amount"`
	MaxAmount            float64   `json:"max_amount"`
	LocationCountry      string    `json:"location_country,omitempty"`
	DeviceType           string    `json:"device_type,omitempty"`
	IsFraud              bool      `json:"is_fraud,omitempty"`
	FraudType            string    `json:"fraud_type,omitempty"`
	TransactionID        string    `json:"transaction_id,omitempty"`
	Timestamp            time.Time `json:"timestamp,omitempty"`
}

// ScoreRequest represents a request to score a transaction
type ScoreRequest struct {
	Transaction *Transaction `json:"transaction" binding:"required"`
}

// ScoreResponse represents the response from scoring
type ScoreResponse struct {
	TransactionID string    `json:"transaction_id"`
	Score         float64   `json:"score"`
	Prediction    int       `json:"prediction"` // 0 = normal, 1 = fraud
	Probability   float64   `json:"probability"`
	Model         string    `json:"model"`
	Timestamp     time.Time `json:"timestamp"`
	ProcessingMs  int64     `json:"processing_ms"`
}

// ExplainRequest represents a request to explain a prediction
type ExplainRequest struct {
	Transaction *Transaction `json:"transaction" binding:"required"`
}

// FeatureContribution represents the contribution of a feature to the prediction
type FeatureContribution struct {
	Feature      string  `json:"feature"`
	Value        float64 `json:"value"`
	Importance   float64 `json:"importance"`
	Contribution float64 `json:"contribution"`
}

// ExplainResponse represents the explanation response
type ExplainResponse struct {
	TransactionID        string                `json:"transaction_id"`
	Score                float64               `json:"score"`
	Prediction           int                   `json:"prediction"`
	FeatureContributions []FeatureContribution `json:"feature_contributions"`
	// ModelScores holds the individual component scores (e.g. "lightgbm",
	// "xgboost", "isolation_forest") that were blended into Score, when the
	// predictor providing the explanation is backed by the ml-serving sidecar.
	ModelScores  map[string]float64 `json:"model_scores,omitempty"`
	Model        string             `json:"model"`
	Timestamp    time.Time          `json:"timestamp"`
	ProcessingMs int64              `json:"processing_ms"`
}

// AccountContext carries the cardholder/account state a credit decisioning
// policy needs on top of the transaction itself (credit limit, utilization,
// delinquency history).
type AccountContext struct {
	CreditLimit             float64 `json:"credit_limit"`
	CurrentBalance          float64 `json:"current_balance"`
	AccountAgeDays          int     `json:"account_age_days"`
	DelinquentPaymentsCount int     `json:"delinquent_payments_count"`
	AvgMonthlySpend         float64 `json:"avg_monthly_spend"`
}

// DecisionRequest represents a request for a credit risk decision (score + policy action).
type DecisionRequest struct {
	Transaction *Transaction    `json:"transaction" binding:"required"`
	Account     *AccountContext `json:"account"`
}

// CreditLimitRecommendation is the credit-limit adjustment signal from the decisioning policy.
type CreditLimitRecommendation struct {
	Current       float64 `json:"current"`
	Recommended   float64 `json:"recommended"`
	AdjustmentPct float64 `json:"adjustment_pct"`
}

// DecisionResponse represents the outcome of applying the credit risk policy to a scored transaction.
type DecisionResponse struct {
	TransactionID             string                    `json:"transaction_id"`
	FraudScore                float64                   `json:"fraud_score"`
	Action                    string                    `json:"action"` // "approve" | "step_up_review" | "decline"
	RiskTier                  string                    `json:"risk_tier"`
	ReasonCodes               []string                  `json:"reason_codes"`
	CreditLimitRecommendation CreditLimitRecommendation `json:"credit_limit_recommendation"`
	Narrative                 string                    `json:"narrative"`
	FeatureContributions      []FeatureContribution     `json:"feature_contributions"`
	ModelScores               map[string]float64        `json:"model_scores"`
	Timestamp                 time.Time                 `json:"timestamp"`
	ProcessingMs              int64                     `json:"processing_ms"`
}

// Case represents a pending human-review item in the analyst queue.
type Case struct {
	ID                     int64              `json:"id"`
	TransactionID          string             `json:"transaction_id"`
	CreatedAt              float64            `json:"created_at"`
	Amount                 float64            `json:"amount"`
	FraudScore             float64            `json:"fraud_score"`
	Action                 string             `json:"action"`
	RiskTier               string             `json:"risk_tier"`
	ReasonCodes            []string           `json:"reason_codes"`
	ModelScores            map[string]float64 `json:"model_scores"`
	CreditLimitCurrent     float64            `json:"credit_limit_current"`
	CreditLimitRecommended float64            `json:"credit_limit_recommended"`
	AnalystVerdict         *string            `json:"analyst_verdict"`
	IsActualFraud          *bool              `json:"is_actual_fraud"`
}

// CasesResponse wraps the pending review queue.
type CasesResponse struct {
	Cases []Case `json:"cases"`
}

// ResolveCaseRequest represents an analyst's verdict on a pending case.
type ResolveCaseRequest struct {
	Verdict       string `json:"verdict" binding:"required"` // "approve" | "decline"
	IsActualFraud *bool  `json:"is_actual_fraud,omitempty"`
}

// FunnelRow is one row of the approval funnel (approve / step_up_review / decline).
type FunnelRow struct {
	Action           string  `json:"action"`
	TransactionCount int64   `json:"transaction_count"`
	PctOfVolume      float64 `json:"pct_of_volume"`
	TotalAmount      float64 `json:"total_amount"`
	AvgFraudScore    float64 `json:"avg_fraud_score"`
}

// ScoreDecileRow is one row of the fraud-rate-by-score-decile breakdown.
type ScoreDecileRow struct {
	ScoreDecile         int     `json:"score_decile"`
	TransactionCount    int64   `json:"transaction_count"`
	ConfirmedFraudCount int64   `json:"confirmed_fraud_count"`
	FraudRatePct        float64 `json:"fraud_rate_pct"`
}

// AnalyticsSummary is the live model-evaluation/decisioning summary for the dashboard.
type AnalyticsSummary struct {
	Funnel       []FunnelRow      `json:"funnel"`
	ScoreDeciles []ScoreDecileRow `json:"score_deciles"`
}

// HealthResponse represents the health check response
type HealthResponse struct {
	Status    string    `json:"status"`
	Timestamp time.Time `json:"timestamp"`
	Version   string    `json:"version"`
	Models    []string  `json:"models"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error     string    `json:"error"`
	Message   string    `json:"message,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// ModelInfo represents information about a loaded model
type ModelInfo struct {
	Name     string             `json:"name"`
	Type     string             `json:"type"`
	Features []string           `json:"features"`
	Metrics  map[string]float64 `json:"metrics"`
	LoadedAt time.Time          `json:"loaded_at"`
}

// ToJSON converts the transaction to JSON
func (t *Transaction) ToJSON() ([]byte, error) {
	return json.Marshal(t)
}

// FromJSON creates a transaction from JSON
func (t *Transaction) FromJSON(data []byte) error {
	return json.Unmarshal(data, t)
}

// GetFeatureVector returns the feature vector for ML models
func (t *Transaction) GetFeatureVector() []float64 {
	return []float64{
		t.Time, t.V1, t.V2, t.V3, t.V4, t.V5, t.V6, t.V7, t.V8, t.V9, t.V10,
		t.V11, t.V12, t.V13, t.V14, t.V15, t.V16, t.V17, t.V18, t.V19, t.V20,
		t.V21, t.V22, t.V23, t.V24, t.V25, t.V26, t.V27, t.V28, t.Amount,
	}
}

// Validate validates the transaction data
func (t *Transaction) Validate() error {
	if t.Amount < 0 {
		return &ValidationError{Field: "amount", Message: "amount must be non-negative"}
	}
	if t.Time < 0 {
		return &ValidationError{Field: "time", Message: "time must be non-negative"}
	}
	return nil
}

// ValidationError represents a validation error
type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return e.Field + ": " + e.Message
}
