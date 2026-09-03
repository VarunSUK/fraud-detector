package ml

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"fraud-detection-inference/internal/models"
	"github.com/sirupsen/logrus"
)

// Predictor interface for ML model predictions
type Predictor interface {
	Predict(transaction *models.Transaction) (float64, int, error)
	Explain(transaction *models.Transaction) (*models.ExplainResponse, error)
	GetModelInfo() *models.ModelInfo
	IsLoaded() bool
}

// RuleBasedPredictor implements rule-based fraud detection
type RuleBasedPredictor struct {
	logger *logrus.Logger
}

// NewRuleBasedPredictor creates a new rule-based predictor
func NewRuleBasedPredictor(logger *logrus.Logger) *RuleBasedPredictor {
	return &RuleBasedPredictor{
		logger: logger,
	}
}

// Predict implements rule-based fraud detection
func (r *RuleBasedPredictor) Predict(transaction *models.Transaction) (float64, int, error) {
	score := 0.0

	// Rule 1: High amount transactions
	if transaction.Amount > 10000 {
		score += 0.3
	}

	// Rule 2: Very high amount transactions
	if transaction.Amount > 50000 {
		score += 0.4
	}

	// Rule 3: Unusual time (night transactions)
	hour := int(transaction.Time/3600) % 24
	if hour < 6 || hour > 22 {
		score += 0.2
	}

	// Rule 4: High PCA feature values (anomaly detection)
	pcaFeatures := []float64{
		transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
		transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
		transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
		transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
		transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
		transaction.V26, transaction.V27, transaction.V28,
	}

	// Check for extreme PCA values
	for _, v := range pcaFeatures {
		if v > 3 || v < -3 { // More than 3 standard deviations
			score += 0.1
		}
	}

	// Normalize score to 0-1 range
	if score > 1.0 {
		score = 1.0
	}

	prediction := 0
	if score > 0.5 {
		prediction = 1
	}

	return score, prediction, nil
}

// Explain provides explanation for rule-based predictions
func (r *RuleBasedPredictor) Explain(transaction *models.Transaction) (*models.ExplainResponse, error) {
	score, prediction, err := r.Predict(transaction)
	if err != nil {
		return nil, err
	}

	contributions := []models.FeatureContribution{}

	// Amount contribution
	if transaction.Amount > 10000 {
		contributions = append(contributions, models.FeatureContribution{
			Feature:      "amount_high",
			Value:        transaction.Amount,
			Importance:   0.3,
			Contribution: 0.3,
		})
	}

	// Time contribution
	hour := int(transaction.Time/3600) % 24
	if hour < 6 || hour > 22 {
		contributions = append(contributions, models.FeatureContribution{
			Feature:      "unusual_time",
			Value:        float64(hour),
			Importance:   0.2,
			Contribution: 0.2,
		})
	}

	// PCA anomaly contribution
	pcaFeatures := []float64{
		transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
		transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
		transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
		transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
		transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
		transaction.V26, transaction.V27, transaction.V28,
	}

	anomalyCount := 0
	for _, v := range pcaFeatures {
		if v > 3 || v < -3 {
			anomalyCount++
		}
	}

	if anomalyCount > 0 {
		contributions = append(contributions, models.FeatureContribution{
			Feature:      "pca_anomalies",
			Value:        float64(anomalyCount),
			Importance:   0.1,
			Contribution: float64(anomalyCount) * 0.1,
		})
	}

	return &models.ExplainResponse{
		TransactionID:        transaction.TransactionID,
		Score:                score,
		Prediction:           prediction,
		FeatureContributions: contributions,
		Model:                "rule_based",
		Timestamp:            time.Now(),
		ProcessingMs:         0,
	}, nil
}

// GetModelInfo returns information about the rule-based model
func (r *RuleBasedPredictor) GetModelInfo() *models.ModelInfo {
	return &models.ModelInfo{
		Name:     "rule_based",
		Type:     "rule_based",
		Features: []string{"amount", "time", "pca_features"},
		Metrics: map[string]float64{
			"accuracy": 0.85, // Estimated accuracy
		},
		LoadedAt: time.Now(),
	}
}

// IsLoaded returns true if the model is loaded
func (r *RuleBasedPredictor) IsLoaded() bool {
	return true
}

// EnsemblePredictor combines multiple predictors
type EnsemblePredictor struct {
	predictors []Predictor
	weights    []float64
	logger     *logrus.Logger
	mu         sync.RWMutex
}

// NewEnsemblePredictor creates a new ensemble predictor
func NewEnsemblePredictor(logger *logrus.Logger) *EnsemblePredictor {
	return &EnsemblePredictor{
		predictors: make([]Predictor, 0),
		weights:    make([]float64, 0),
		logger:     logger,
	}
}

// AddPredictor adds a predictor to the ensemble
func (e *EnsemblePredictor) AddPredictor(predictor Predictor, weight float64) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.predictors = append(e.predictors, predictor)
	e.weights = append(e.weights, weight)

	e.logger.WithFields(logrus.Fields{
		"predictor": predictor.GetModelInfo().Name,
		"weight":    weight,
	}).Info("Added predictor to ensemble")
}

// Predict combines predictions from all predictors
func (e *EnsemblePredictor) Predict(transaction *models.Transaction) (float64, int, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(e.predictors) == 0 {
		return 0.0, 0, fmt.Errorf("no predictors in ensemble")
	}

	var totalScore float64
	var totalWeight float64

	for i, predictor := range e.predictors {
		if !predictor.IsLoaded() {
			continue
		}

		score, _, err := predictor.Predict(transaction)
		if err != nil {
			e.logger.WithError(err).WithField("predictor", i).Warn("Predictor failed")
			continue
		}

		totalScore += score * e.weights[i]
		totalWeight += e.weights[i]
	}

	if totalWeight == 0 {
		return 0.0, 0, fmt.Errorf("no valid predictions")
	}

	finalScore := totalScore / totalWeight
	prediction := 0
	if finalScore > 0.5 {
		prediction = 1
	}

	return finalScore, prediction, nil
}

// Explain provides explanation from the ensemble
func (e *EnsemblePredictor) Explain(transaction *models.Transaction) (*models.ExplainResponse, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	// Prefer the most recently added (highest-fidelity) loaded predictor, e.g. the
	// real ML service over the rule-based fallback, which is added first.
	for i := len(e.predictors) - 1; i >= 0; i-- {
		if e.predictors[i].IsLoaded() {
			return e.predictors[i].Explain(transaction)
		}
	}

	return nil, fmt.Errorf("no loaded predictors for explanation")
}

// GetModelInfo returns information about the ensemble
func (e *EnsemblePredictor) GetModelInfo() *models.ModelInfo {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var modelNames []string
	for _, predictor := range e.predictors {
		if predictor.IsLoaded() {
			modelNames = append(modelNames, predictor.GetModelInfo().Name)
		}
	}

	return &models.ModelInfo{
		Name:     "ensemble",
		Type:     "ensemble",
		Features: []string{"combined_features"},
		Metrics: map[string]float64{
			"num_models": float64(len(modelNames)),
		},
		LoadedAt: time.Now(),
	}
}

// IsLoaded returns true if at least one predictor is loaded
func (e *EnsemblePredictor) IsLoaded() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()

	for _, predictor := range e.predictors {
		if predictor.IsLoaded() {
			return true
		}
	}
	return false
}

// ModelManager manages multiple ML models
type ModelManager struct {
	models       map[string]Predictor
	mlService    *MLServicePredictor
	logger       *logrus.Logger
	modelsDir    string
	mlServiceURL string
	mu           sync.RWMutex
}

// NewModelManager creates a new model manager
func NewModelManager(logger *logrus.Logger, modelsDir string, mlServiceURL string) *ModelManager {
	return &ModelManager{
		models:       make(map[string]Predictor),
		logger:       logger,
		modelsDir:    modelsDir,
		mlServiceURL: mlServiceURL,
	}
}

// LoadModels loads all available models from the models directory
func (m *ModelManager) LoadModels() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Always add rule-based predictor
	ruleBased := NewRuleBasedPredictor(m.logger)
	m.models["rule_based"] = ruleBased

	// Create ensemble predictor
	ensemble := NewEnsemblePredictor(m.logger)
	ensemble.AddPredictor(ruleBased, 0.3)

	// Try to load ML models from Python training
	if err := m.loadMLModels(ensemble); err != nil {
		m.logger.WithError(err).Warn("Failed to load ML models, using rule-based only")
	}

	m.models["ensemble"] = ensemble

	m.logger.WithField("num_models", len(m.models)).Info("Models loaded successfully")
	return nil
}

// loadMLModels wires in the real LightGBM/XGBoost models served by the
// Python ml-serving sidecar (see python-ml/src/serve.py). The predictor
// degrades gracefully if the sidecar is unreachable or has no models loaded.
func (m *ModelManager) loadMLModels(ensemble *EnsemblePredictor) error {
	mlService := NewMLServicePredictor(m.logger, m.mlServiceURL)
	ensemble.AddPredictor(mlService, 0.7)

	// Registered independently (not just inside the ensemble) so it can be
	// listed on its own and used directly for credit decisioning, which needs
	// account context the generic Predictor interface doesn't carry.
	m.models["ml_service"] = mlService
	m.mlService = mlService

	return nil
}

// GetPredictor returns a predictor by name
func (m *ModelManager) GetPredictor(name string) (Predictor, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	predictor, exists := m.models[name]
	return predictor, exists
}

// GetMLServicePredictor returns the ml-serving-backed predictor directly, for
// capabilities (credit decisioning, case review) that aren't part of the
// generic Predictor interface.
func (m *ModelManager) GetMLServicePredictor() (*MLServicePredictor, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.mlService, m.mlService != nil
}

// GetAvailableModels returns a list of available model names
func (m *ModelManager) GetAvailableModels() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var models []string
	for name := range m.models {
		models = append(models, name)
	}
	return models
}

// mlServiceHealthCheckInterval controls how often MLServicePredictor
// re-checks the sidecar's /health endpoint instead of hitting it on every call.
const mlServiceHealthCheckInterval = 5 * time.Second

// MLServicePredictor calls out to the Python ml-serving sidecar (see
// python-ml/src/serve.py), which loads the real LightGBM/XGBoost models
// trained by python-ml/train.py.
type MLServicePredictor struct {
	logger     *logrus.Logger
	baseURL    string
	httpClient *http.Client

	mu              sync.RWMutex
	healthy         bool
	loadedModels    []string
	lastHealthCheck time.Time
}

// NewMLServicePredictor creates a predictor backed by the ml-serving sidecar at baseURL.
func NewMLServicePredictor(logger *logrus.Logger, baseURL string) *MLServicePredictor {
	p := &MLServicePredictor{
		logger:     logger,
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{Timeout: 5 * time.Second},
	}
	p.refreshHealth()
	return p
}

type mlServiceHealthResponse struct {
	Status       string   `json:"status"`
	ModelsLoaded []string `json:"models_loaded"`
}

func (p *MLServicePredictor) refreshHealth() {
	resp, err := p.httpClient.Get(p.baseURL + "/health")

	p.mu.Lock()
	defer p.mu.Unlock()
	p.lastHealthCheck = time.Now()

	if err != nil {
		p.logger.WithError(err).Warn("ml-service health check failed")
		p.healthy = false
		p.loadedModels = nil
		return
	}
	defer resp.Body.Close()

	var health mlServiceHealthResponse
	if resp.StatusCode != http.StatusOK || json.NewDecoder(resp.Body).Decode(&health) != nil {
		p.healthy = false
		p.loadedModels = nil
		return
	}

	p.healthy = health.Status == "healthy" && len(health.ModelsLoaded) > 0
	p.loadedModels = health.ModelsLoaded
}

// IsLoaded reports whether the ml-serving sidecar currently has models loaded.
func (p *MLServicePredictor) IsLoaded() bool {
	p.mu.RLock()
	stale := time.Since(p.lastHealthCheck) > mlServiceHealthCheckInterval
	p.mu.RUnlock()

	if stale {
		p.refreshHealth()
	}

	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.healthy
}

type mlServicePredictResponse struct {
	Score      float64 `json:"score"`
	Prediction int     `json:"prediction"`
}

// Predict sends the transaction to the ml-serving sidecar's /predict endpoint.
func (p *MLServicePredictor) Predict(transaction *models.Transaction) (float64, int, error) {
	body, err := json.Marshal(transaction)
	if err != nil {
		return 0, 0, fmt.Errorf("marshal transaction: %w", err)
	}

	resp, err := p.httpClient.Post(p.baseURL+"/predict", "application/json", bytes.NewReader(body))
	if err != nil {
		return 0, 0, fmt.Errorf("ml-service request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, 0, fmt.Errorf("ml-service returned status %d", resp.StatusCode)
	}

	var result mlServicePredictResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, 0, fmt.Errorf("decode ml-service response: %w", err)
	}

	return result.Score, result.Prediction, nil
}

type mlServiceExplainResponse struct {
	Score                float64                      `json:"score"`
	Prediction           int                          `json:"prediction"`
	FeatureContributions []models.FeatureContribution `json:"feature_contributions"`
	ModelScores          map[string]float64           `json:"model_scores"`
}

// Explain sends the transaction to the ml-serving sidecar's /explain endpoint.
func (p *MLServicePredictor) Explain(transaction *models.Transaction) (*models.ExplainResponse, error) {
	body, err := json.Marshal(transaction)
	if err != nil {
		return nil, fmt.Errorf("marshal transaction: %w", err)
	}

	resp, err := p.httpClient.Post(p.baseURL+"/explain", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ml-service request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ml-service returned status %d", resp.StatusCode)
	}

	var result mlServiceExplainResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode ml-service response: %w", err)
	}

	return &models.ExplainResponse{
		TransactionID:        transaction.TransactionID,
		Score:                result.Score,
		Prediction:           result.Prediction,
		FeatureContributions: result.FeatureContributions,
		ModelScores:          result.ModelScores,
		Model:                "ml_service",
		Timestamp:            time.Now(),
	}, nil
}

// GetModelInfo returns information about the models currently loaded by the sidecar.
func (p *MLServicePredictor) GetModelInfo() *models.ModelInfo {
	p.mu.RLock()
	defer p.mu.RUnlock()

	return &models.ModelInfo{
		Name:     "ml_service",
		Type:     "ml",
		Features: []string{"v1..v28", "time", "amount"},
		Metrics: map[string]float64{
			"num_models_loaded": float64(len(p.loadedModels)),
		},
		LoadedAt: p.lastHealthCheck,
	}
}

type mlServiceDecisionResponse struct {
	TransactionID             string                           `json:"transaction_id"`
	FraudScore                float64                          `json:"fraud_score"`
	Action                    string                           `json:"action"`
	RiskTier                  string                           `json:"risk_tier"`
	ReasonCodes               []string                         `json:"reason_codes"`
	CreditLimitRecommendation models.CreditLimitRecommendation `json:"credit_limit_recommendation"`
	Narrative                 string                           `json:"narrative"`
	FeatureContributions      []models.FeatureContribution     `json:"feature_contributions"`
	ModelScores               map[string]float64               `json:"model_scores"`
}

// Decide sends the transaction and account context to the ml-serving sidecar's
// /decision endpoint, which applies the credit risk policy and records the
// outcome to its audit log. This isn't part of the Predictor interface: only
// the ml-serving-backed predictor supports credit decisioning today.
func (p *MLServicePredictor) Decide(transaction *models.Transaction, account *models.AccountContext) (*models.DecisionResponse, error) {
	payload := struct {
		Transaction *models.Transaction    `json:"transaction"`
		Account     *models.AccountContext `json:"account"`
	}{Transaction: transaction, Account: account}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal decision request: %w", err)
	}

	resp, err := p.httpClient.Post(p.baseURL+"/decision", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ml-service request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ml-service returned status %d", resp.StatusCode)
	}

	var result mlServiceDecisionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode ml-service response: %w", err)
	}

	return &models.DecisionResponse{
		TransactionID:             result.TransactionID,
		FraudScore:                result.FraudScore,
		Action:                    result.Action,
		RiskTier:                  result.RiskTier,
		ReasonCodes:               result.ReasonCodes,
		CreditLimitRecommendation: result.CreditLimitRecommendation,
		Narrative:                 result.Narrative,
		FeatureContributions:      result.FeatureContributions,
		ModelScores:               result.ModelScores,
		Timestamp:                 time.Now(),
	}, nil
}

// GetAnalyticsSummary fetches the live approval-funnel and score-decile
// breakdown from the ml-serving sidecar's audit log.
func (p *MLServicePredictor) GetAnalyticsSummary() (*models.AnalyticsSummary, error) {
	resp, err := p.httpClient.Get(p.baseURL + "/analytics/summary")
	if err != nil {
		return nil, fmt.Errorf("ml-service request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ml-service returned status %d", resp.StatusCode)
	}

	var result models.AnalyticsSummary
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode ml-service response: %w", err)
	}

	return &result, nil
}

// ListCases fetches the pending human-review queue from the ml-serving sidecar.
func (p *MLServicePredictor) ListCases() ([]models.Case, error) {
	resp, err := p.httpClient.Get(p.baseURL + "/cases")
	if err != nil {
		return nil, fmt.Errorf("ml-service request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ml-service returned status %d", resp.StatusCode)
	}

	var result models.CasesResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode ml-service response: %w", err)
	}

	return result.Cases, nil
}

// ResolveCase records an analyst's verdict on a pending case via the ml-serving sidecar.
func (p *MLServicePredictor) ResolveCase(caseID int64, req *models.ResolveCaseRequest) error {
	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal resolve request: %w", err)
	}

	url := fmt.Sprintf("%s/cases/%d/resolve", p.baseURL, caseID)
	resp, err := p.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("ml-service request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return errCaseNotFound
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ml-service returned status %d", resp.StatusCode)
	}

	return nil
}

var errCaseNotFound = fmt.Errorf("case not found")

// IsErrCaseNotFound reports whether err indicates the sidecar returned 404 for a case.
func IsErrCaseNotFound(err error) bool {
	return err == errCaseNotFound
}
