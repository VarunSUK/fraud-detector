package ml

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"fraud-detection-inference/internal/models"
	"github.com/sirupsen/logrus"
)

func testLogger() *logrus.Logger {
	logger := logrus.New()
	logger.SetOutput(nowhere{})
	return logger
}

// nowhere discards everything written to it, keeping test output clean.
type nowhere struct{}

func (nowhere) Write(p []byte) (int, error) { return len(p), nil }

func TestRuleBasedPredictor_Predict(t *testing.T) {
	predictor := NewRuleBasedPredictor(testLogger())

	tests := []struct {
		name     string
		txn      *models.Transaction
		wantPred int
		minScore float64
	}{
		{"low risk small daytime transaction", &models.Transaction{Amount: 50, Time: 12 * 3600}, 0, 0},
		{"very high amount", &models.Transaction{Amount: 60000, Time: 12 * 3600}, 1, 0.5},
		{"night time transaction", &models.Transaction{Amount: 50, Time: 2 * 3600}, 0, 0.2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score, prediction, err := predictor.Predict(tt.txn)
			if err != nil {
				t.Fatalf("Predict() error = %v", err)
			}
			if prediction != tt.wantPred {
				t.Errorf("prediction = %d, want %d (score=%v)", prediction, tt.wantPred, score)
			}
			if score < tt.minScore {
				t.Errorf("score = %v, want >= %v", score, tt.minScore)
			}
			if score > 1.0 {
				t.Errorf("score = %v, want <= 1.0", score)
			}
		})
	}
}

func TestRuleBasedPredictor_Explain(t *testing.T) {
	predictor := NewRuleBasedPredictor(testLogger())
	txn := &models.Transaction{Amount: 60000, Time: 2 * 3600, TransactionID: "txn_1"}

	resp, err := predictor.Explain(txn)
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}
	if resp.TransactionID != "txn_1" {
		t.Errorf("TransactionID = %q, want txn_1", resp.TransactionID)
	}
	if len(resp.FeatureContributions) == 0 {
		t.Error("expected at least one feature contribution for a high-amount, night-time transaction")
	}
}

func TestRuleBasedPredictor_IsLoaded(t *testing.T) {
	if !NewRuleBasedPredictor(testLogger()).IsLoaded() {
		t.Error("rule-based predictor should always report loaded")
	}
}

// fakePredictor lets ensemble tests control score/loaded state directly.
type fakePredictor struct {
	name    string
	score   float64
	loaded  bool
	failErr error
}

func (f *fakePredictor) Predict(*models.Transaction) (float64, int, error) {
	if f.failErr != nil {
		return 0, 0, f.failErr
	}
	pred := 0
	if f.score > 0.5 {
		pred = 1
	}
	return f.score, pred, nil
}

func (f *fakePredictor) Explain(txn *models.Transaction) (*models.ExplainResponse, error) {
	return &models.ExplainResponse{TransactionID: txn.TransactionID, Score: f.score, Model: f.name}, nil
}

func (f *fakePredictor) GetModelInfo() *models.ModelInfo {
	return &models.ModelInfo{Name: f.name, Type: "fake"}
}

func (f *fakePredictor) IsLoaded() bool { return f.loaded }

func TestEnsemblePredictor_Predict_WeightedAverage(t *testing.T) {
	ensemble := NewEnsemblePredictor(testLogger())
	ensemble.AddPredictor(&fakePredictor{name: "a", score: 1.0, loaded: true}, 0.3)
	ensemble.AddPredictor(&fakePredictor{name: "b", score: 0.0, loaded: true}, 0.7)

	score, prediction, err := ensemble.Predict(&models.Transaction{})
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	wantScore := 0.3 // (1.0*0.3 + 0.0*0.7) / (0.3+0.7)
	if score < wantScore-1e-9 || score > wantScore+1e-9 {
		t.Errorf("score = %v, want %v", score, wantScore)
	}
	if prediction != 0 {
		t.Errorf("prediction = %d, want 0", prediction)
	}
}

func TestEnsemblePredictor_Predict_SkipsUnloadedAndFailing(t *testing.T) {
	ensemble := NewEnsemblePredictor(testLogger())
	ensemble.AddPredictor(&fakePredictor{name: "unloaded", score: 1.0, loaded: false}, 1.0)
	ensemble.AddPredictor(&fakePredictor{name: "failing", loaded: true, failErr: errFake}, 1.0)
	ensemble.AddPredictor(&fakePredictor{name: "healthy", score: 0.8, loaded: true}, 1.0)

	score, prediction, err := ensemble.Predict(&models.Transaction{})
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}
	if score != 0.8 {
		t.Errorf("score = %v, want 0.8 (only the healthy predictor should count)", score)
	}
	if prediction != 1 {
		t.Errorf("prediction = %d, want 1", prediction)
	}
}

func TestEnsemblePredictor_Predict_NoPredictors(t *testing.T) {
	ensemble := NewEnsemblePredictor(testLogger())
	if _, _, err := ensemble.Predict(&models.Transaction{}); err == nil {
		t.Error("expected error when ensemble has no predictors")
	}
}

func TestEnsemblePredictor_Explain_PrefersLastLoaded(t *testing.T) {
	ensemble := NewEnsemblePredictor(testLogger())
	ensemble.AddPredictor(&fakePredictor{name: "rule_based", loaded: true}, 0.3)
	ensemble.AddPredictor(&fakePredictor{name: "ml_service", loaded: true}, 0.7)

	resp, err := ensemble.Explain(&models.Transaction{})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}
	if resp.Model != "ml_service" {
		t.Errorf("Explain() picked model %q, want ml_service (should prefer the higher-fidelity loaded predictor)", resp.Model)
	}
}

func TestEnsemblePredictor_Explain_FallsBackWhenPreferredUnloaded(t *testing.T) {
	ensemble := NewEnsemblePredictor(testLogger())
	ensemble.AddPredictor(&fakePredictor{name: "rule_based", loaded: true}, 0.3)
	ensemble.AddPredictor(&fakePredictor{name: "ml_service", loaded: false}, 0.7)

	resp, err := ensemble.Explain(&models.Transaction{})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}
	if resp.Model != "rule_based" {
		t.Errorf("Explain() picked model %q, want rule_based fallback", resp.Model)
	}
}

var errFake = &fakeError{"fake predictor failure"}

type fakeError struct{ msg string }

func (e *fakeError) Error() string { return e.msg }

func TestMLServicePredictor_HealthyPredictAndExplain(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			json.NewEncoder(w).Encode(mlServiceHealthResponse{Status: "healthy", ModelsLoaded: []string{"lightgbm", "xgboost"}})
		case "/predict":
			json.NewEncoder(w).Encode(mlServicePredictResponse{Score: 0.9, Prediction: 1})
		case "/explain":
			json.NewEncoder(w).Encode(mlServiceExplainResponse{
				Score: 0.9, Prediction: 1,
				FeatureContributions: []models.FeatureContribution{{Feature: "v14", Value: -5, Importance: 0.4, Contribution: 2.0}},
				ModelScores:          map[string]float64{"lightgbm": 0.92, "xgboost": 0.88, "isolation_forest": 0.7},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	predictor := NewMLServicePredictor(testLogger(), server.URL)

	if !predictor.IsLoaded() {
		t.Fatal("expected predictor to report loaded when sidecar is healthy")
	}

	score, prediction, err := predictor.Predict(&models.Transaction{Amount: 100})
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}
	if score != 0.9 || prediction != 1 {
		t.Errorf("Predict() = (%v, %v), want (0.9, 1)", score, prediction)
	}

	explainResp, err := predictor.Explain(&models.Transaction{TransactionID: "txn_1"})
	if err != nil {
		t.Fatalf("Explain() error = %v", err)
	}
	if explainResp.TransactionID != "txn_1" {
		t.Errorf("TransactionID = %q, want txn_1", explainResp.TransactionID)
	}
	if len(explainResp.FeatureContributions) != 1 {
		t.Errorf("FeatureContributions len = %d, want 1", len(explainResp.FeatureContributions))
	}
	if len(explainResp.ModelScores) != 3 {
		t.Errorf("ModelScores len = %d, want 3 (lightgbm, xgboost, isolation_forest)", len(explainResp.ModelScores))
	}
	if explainResp.ModelScores["isolation_forest"] != 0.7 {
		t.Errorf("ModelScores[isolation_forest] = %v, want 0.7", explainResp.ModelScores["isolation_forest"])
	}
}

func TestMLServicePredictor_DecisionAndCases(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/decision":
			json.NewEncoder(w).Encode(mlServiceDecisionResponse{
				TransactionID: "txn_1",
				FraudScore:    0.6,
				Action:        "step_up_review",
				RiskTier:      "medium",
				ReasonCodes:   []string{"ELEVATED_FRAUD_SCORE"},
				CreditLimitRecommendation: models.CreditLimitRecommendation{
					Current: 5000, Recommended: 5000, AdjustmentPct: 0,
				},
				Narrative:   "txn_1 scored 0.60 and was routed to manual review.",
				ModelScores: map[string]float64{"lightgbm": 0.6},
			})
		case r.URL.Path == "/cases":
			json.NewEncoder(w).Encode(models.CasesResponse{
				Cases: []models.Case{{ID: 1, TransactionID: "txn_1", Action: "step_up_review"}},
			})
		case r.URL.Path == "/cases/1/resolve":
			json.NewEncoder(w).Encode(map[string]interface{}{"case_id": 1, "verdict": "approve"})
		case r.URL.Path == "/cases/999/resolve":
			http.NotFound(w, r)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	predictor := NewMLServicePredictor(testLogger(), server.URL)

	decision, err := predictor.Decide(&models.Transaction{Amount: 500}, &models.AccountContext{CreditLimit: 5000})
	if err != nil {
		t.Fatalf("Decide() error = %v", err)
	}
	if decision.Action != "step_up_review" || decision.TransactionID != "txn_1" {
		t.Errorf("Decide() = %+v, unexpected fields", decision)
	}
	if decision.Narrative == "" {
		t.Error("expected a non-empty narrative")
	}

	cases, err := predictor.ListCases()
	if err != nil {
		t.Fatalf("ListCases() error = %v", err)
	}
	if len(cases) != 1 || cases[0].TransactionID != "txn_1" {
		t.Errorf("ListCases() = %+v, want one case for txn_1", cases)
	}

	if err := predictor.ResolveCase(1, &models.ResolveCaseRequest{Verdict: "approve"}); err != nil {
		t.Errorf("ResolveCase(1) error = %v", err)
	}

	err = predictor.ResolveCase(999, &models.ResolveCaseRequest{Verdict: "approve"})
	if err == nil || !IsErrCaseNotFound(err) {
		t.Errorf("ResolveCase(999) error = %v, want a not-found error", err)
	}
}

func TestMLServicePredictor_DegradedWhenNoModelsLoaded(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(mlServiceHealthResponse{Status: "degraded", ModelsLoaded: []string{}})
	}))
	defer server.Close()

	predictor := NewMLServicePredictor(testLogger(), server.URL)
	if predictor.IsLoaded() {
		t.Error("expected predictor to report not loaded when sidecar has no models")
	}
}

func TestMLServicePredictor_ConcurrentIsLoadedDoesNotStampede(t *testing.T) {
	var healthCheckCount int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			atomic.AddInt64(&healthCheckCount, 1)
			time.Sleep(20 * time.Millisecond) // simulate a slow sidecar under load
			json.NewEncoder(w).Encode(mlServiceHealthResponse{Status: "healthy", ModelsLoaded: []string{"lightgbm"}})
		}
	}))
	defer server.Close()

	predictor := NewMLServicePredictor(testLogger(), server.URL)
	// Force the cache stale so the next round of IsLoaded() calls all race to refresh.
	predictor.mu.Lock()
	predictor.lastHealthCheck = time.Time{}
	predictor.mu.Unlock()
	atomic.StoreInt64(&healthCheckCount, 0)

	var wg sync.WaitGroup
	const concurrency = 50
	wg.Add(concurrency)
	for i := 0; i < concurrency; i++ {
		go func() {
			defer wg.Done()
			predictor.IsLoaded()
		}()
	}
	wg.Wait()

	// NewMLServicePredictor already made one call at construction; a stale
	// cache hit by 50 concurrent goroutines should trigger exactly one more,
	// not up to 50 (the thundering-herd bug this test guards against).
	count := atomic.LoadInt64(&healthCheckCount)
	if count != 1 {
		t.Errorf("health check called %d times for 50 concurrent stale reads, want exactly 1", count)
	}
}

func TestMLServicePredictor_UnreachableSidecar(t *testing.T) {
	predictor := NewMLServicePredictor(testLogger(), "http://127.0.0.1:1")
	if predictor.IsLoaded() {
		t.Error("expected predictor to report not loaded when sidecar is unreachable")
	}
	if _, _, err := predictor.Predict(&models.Transaction{}); err == nil {
		t.Error("expected Predict() to error when sidecar is unreachable")
	}
}

func TestModelManager_LoadModels(t *testing.T) {
	// No ml-serving sidecar running at this address, so it should degrade to rule-based only.
	manager := NewModelManager(testLogger(), "models", "http://127.0.0.1:1")
	if err := manager.LoadModels(); err != nil {
		t.Fatalf("LoadModels() error = %v", err)
	}

	ruleBased, ok := manager.GetPredictor("rule_based")
	if !ok || !ruleBased.IsLoaded() {
		t.Error("expected rule_based predictor to be present and loaded")
	}

	ensemble, ok := manager.GetPredictor("ensemble")
	if !ok {
		t.Fatal("expected ensemble predictor to be present")
	}
	if !ensemble.IsLoaded() {
		t.Error("expected ensemble to still be loaded via the rule-based fallback")
	}

	score, _, err := ensemble.Predict(&models.Transaction{Amount: 100, Time: 1000})
	if err != nil {
		t.Fatalf("ensemble.Predict() error = %v", err)
	}
	if score < 0 || score > 1 {
		t.Errorf("score = %v, want within [0,1]", score)
	}
}
