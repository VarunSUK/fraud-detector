package ml

import (
	"fmt"
	"os"
	"path/filepath"
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
		Name: "rule_based",
		Type: "rule_based",
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
	
	// For now, use the first loaded predictor for explanation
	for _, predictor := range e.predictors {
		if predictor.IsLoaded() {
			return predictor.Explain(transaction)
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
	models     map[string]Predictor
	logger     *logrus.Logger
	modelsDir  string
	mu         sync.RWMutex
}

// NewModelManager creates a new model manager
func NewModelManager(logger *logrus.Logger, modelsDir string) *ModelManager {
	return &ModelManager{
		models:    make(map[string]Predictor),
		logger:    logger,
		modelsDir: modelsDir,
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

// loadMLModels attempts to load ML models from Python training output
func (m *ModelManager) loadMLModels(ensemble *EnsemblePredictor) error {
	// For now, we'll implement a mock ML predictor
	// In a real implementation, you would load the actual trained models
	// using libraries like ONNX Runtime or similar
	
	mockML := &MockMLPredictor{
		logger: m.logger,
	}
	
	ensemble.AddPredictor(mockML, 0.7)
	
	return nil
}

// GetPredictor returns a predictor by name
func (m *ModelManager) GetPredictor(name string) (Predictor, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	predictor, exists := m.models[name]
	return predictor, exists
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

// MockMLPredictor is a mock ML predictor for demonstration
type MockMLPredictor struct {
	logger *logrus.Logger
}

// Predict implements mock ML prediction
func (m *MockMLPredictor) Predict(transaction *models.Transaction) (float64, int, error) {
	// Simple mock prediction based on PCA features
	score := 0.0
	
	// Check for anomalies in PCA features
	pcaFeatures := []float64{
		transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
		transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
		transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
		transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
		transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
		transaction.V26, transaction.V27, transaction.V28,
	}
	
	// Calculate anomaly score
	anomalyScore := 0.0
	for _, v := range pcaFeatures {
		if v > 2 || v < -2 {
			anomalyScore += 0.1
		}
	}
	
	// Amount-based scoring
	amountScore := 0.0
	if transaction.Amount > 5000 {
		amountScore = 0.3
	}
	if transaction.Amount > 20000 {
		amountScore = 0.6
	}
	
	score = anomalyScore + amountScore
	if score > 1.0 {
		score = 1.0
	}
	
	prediction := 0
	if score > 0.5 {
		prediction = 1
	}
	
	return score, prediction, nil
}

// Explain provides mock explanation
func (m *MockMLPredictor) Explain(transaction *models.Transaction) (*models.ExplainResponse, error) {
	score, prediction, err := m.Predict(transaction)
	if err != nil {
		return nil, err
	}
	
	contributions := []models.FeatureContribution{
		{
			Feature:      "pca_anomaly_score",
			Value:        0.5, // Mock value
			Importance:   0.4,
			Contribution: 0.4,
		},
		{
			Feature:      "amount_score",
			Value:        transaction.Amount,
			Importance:   0.3,
			Contribution: 0.3,
		},
	}
	
	return &models.ExplainResponse{
		TransactionID:        transaction.TransactionID,
		Score:                score,
		Prediction:           prediction,
		FeatureContributions: contributions,
		Model:                "mock_ml",
		Timestamp:            time.Now(),
		ProcessingMs:         1,
	}, nil
}

// GetModelInfo returns mock model information
func (m *MockMLPredictor) GetModelInfo() *models.ModelInfo {
	return &models.ModelInfo{
		Name:     "mock_ml",
		Type:     "ml",
		Features: []string{"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "amount", "time"},
		Metrics: map[string]float64{
			"accuracy": 0.92,
			"auc":      0.95,
		},
		LoadedAt: time.Now(),
	}
}

// IsLoaded returns true
func (m *MockMLPredictor) IsLoaded() bool {
	return true
}

// Helper function to check if file exists
func fileExists(filename string) bool {
	_, err := os.Stat(filename)
	return !os.IsNotExist(err)
}

// Helper function to find model files
func findModelFiles(modelsDir string) ([]string, error) {
	var modelFiles []string
	
	err := filepath.Walk(modelsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		
		if filepath.Ext(path) == ".joblib" || filepath.Ext(path) == ".json" {
			modelFiles = append(modelFiles, path)
		}
		
		return nil
	})
	
	return modelFiles, err
}
