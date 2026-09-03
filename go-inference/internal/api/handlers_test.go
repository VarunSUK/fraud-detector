package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"fraud-detection-inference/internal/ml"
	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func testHandlers(t *testing.T) *Handlers {
	t.Helper()
	// No ml-serving sidecar reachable in tests, so this exercises the rule-based fallback path.
	return testHandlersWithMLService(t, "http://127.0.0.1:1")
}

func testHandlersWithMLService(t *testing.T, mlServiceURL string) *Handlers {
	t.Helper()
	logger := logrus.New()
	logger.SetOutput(nowhere{})

	manager := ml.NewModelManager(logger, "models", mlServiceURL)
	if err := manager.LoadModels(); err != nil {
		t.Fatalf("LoadModels() error = %v", err)
	}
	return NewHandlers(manager, logger, "test")
}

type nowhere struct{}

func (nowhere) Write(p []byte) (int, error) { return len(p), nil }

func doRequest(handler gin.HandlerFunc, method, path string, body interface{}) *httptest.ResponseRecorder {
	return doRequestRoute(handler, method, path, path, body)
}

// doRequestRoute lets the registered route pattern (e.g. "/cases/:id/resolve")
// differ from the actual request path (e.g. "/cases/1/resolve").
func doRequestRoute(handler gin.HandlerFunc, method, routePath, requestPath string, body interface{}) *httptest.ResponseRecorder {
	router := gin.New()
	router.Handle(method, routePath, handler)

	var reqBody *bytes.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		reqBody = bytes.NewReader(b)
	} else {
		reqBody = bytes.NewReader(nil)
	}

	req := httptest.NewRequest(method, requestPath, reqBody)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)
	return w
}

func TestHealthHandler(t *testing.T) {
	h := testHandlers(t)
	w := doRequest(h.HealthHandler, http.MethodGet, "/health", nil)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if resp["status"] != "healthy" {
		t.Errorf("status field = %v, want healthy", resp["status"])
	}
}

func TestScoreHandler_ValidTransaction(t *testing.T) {
	h := testHandlers(t)
	body := map[string]interface{}{
		"transaction": map[string]interface{}{
			"time":           1000,
			"amount":         60000,
			"transaction_id": "txn_1",
		},
	}
	w := doRequest(h.ScoreHandler, http.MethodPost, "/api/v1/score", body)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if resp["transaction_id"] != "txn_1" {
		t.Errorf("transaction_id = %v, want txn_1", resp["transaction_id"])
	}
	if resp["model"] != "ensemble" {
		t.Errorf("model = %v, want ensemble (default)", resp["model"])
	}
}

func TestScoreHandler_InvalidJSON(t *testing.T) {
	h := testHandlers(t)
	w := doRequest(h.ScoreHandler, http.MethodPost, "/api/v1/score", map[string]interface{}{})

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d (missing required transaction field)", w.Code, http.StatusBadRequest)
	}
}

func TestScoreHandler_NegativeAmountFailsValidation(t *testing.T) {
	h := testHandlers(t)
	body := map[string]interface{}{
		"transaction": map[string]interface{}{
			"time":   1000,
			"amount": -50,
		},
	}
	w := doRequest(h.ScoreHandler, http.MethodPost, "/api/v1/score", body)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusBadRequest)
	}
}

func TestScoreHandler_UnknownModel(t *testing.T) {
	h := testHandlers(t)
	body := map[string]interface{}{
		"transaction": map[string]interface{}{
			"time":   1000,
			"amount": 50,
		},
	}
	w := doRequest(h.ScoreHandler, http.MethodPost, "/api/v1/score?model=nonexistent", body)

	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusNotFound)
	}
}

func TestModelsHandler(t *testing.T) {
	h := testHandlers(t)
	w := doRequest(h.ModelsHandler, http.MethodGet, "/api/v1/models", nil)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	count, ok := resp["count"].(float64)
	if !ok || count < 2 {
		t.Errorf("expected at least 2 models (rule_based, ensemble), got %v", resp["count"])
	}
}

func TestExplainHandler_ValidTransaction(t *testing.T) {
	h := testHandlers(t)
	body := map[string]interface{}{
		"transaction": map[string]interface{}{
			"time":           2 * 3600,
			"amount":         60000,
			"transaction_id": "txn_explain",
		},
	}
	w := doRequest(h.ExplainHandler, http.MethodPost, "/api/v1/explain", body)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if resp["transaction_id"] != "txn_explain" {
		t.Errorf("transaction_id = %v, want txn_explain", resp["transaction_id"])
	}
}

func TestMetricsHandler_ReturnsPrometheusExpositionFormat(t *testing.T) {
	h := testHandlers(t)
	w := doRequest(h.MetricsHandler, http.MethodGet, "/metrics", nil)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}
	body := w.Body.String()
	if !strings.Contains(body, "# HELP fraud_detection_requests_total") {
		preview := body
		if len(preview) > 200 {
			preview = preview[:200]
		}
		t.Errorf("expected Prometheus exposition format with fraud_detection_requests_total, got: %s", preview)
	}
}

func TestDecisionHandler_ReturnsServiceUnavailableWhenMLServiceDown(t *testing.T) {
	h := testHandlers(t) // ml-service unreachable
	body := map[string]interface{}{
		"transaction": map[string]interface{}{"time": 1000, "amount": 500},
		"account":     map[string]interface{}{"credit_limit": 5000},
	}
	w := doRequest(h.DecisionHandler, http.MethodPost, "/api/v1/decision", body)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d (ml-service unreachable)", w.Code, http.StatusServiceUnavailable)
	}
}

func TestDecisionHandler_InvalidTransactionFailsValidation(t *testing.T) {
	h := testHandlers(t)
	body := map[string]interface{}{
		"transaction": map[string]interface{}{"time": 1000, "amount": -50},
	}
	w := doRequest(h.DecisionHandler, http.MethodPost, "/api/v1/decision", body)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusBadRequest)
	}
}

func fakeMLServiceServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/health":
			w.Write([]byte(`{"status":"healthy","models_loaded":["lightgbm","xgboost","isolation_forest"]}`))
		case r.URL.Path == "/decision":
			w.Write([]byte(`{
				"transaction_id": "txn_case",
				"fraud_score": 0.6,
				"action": "step_up_review",
				"risk_tier": "medium",
				"reason_codes": ["ELEVATED_FRAUD_SCORE"],
				"credit_limit_recommendation": {"current": 5000, "recommended": 5000, "adjustment_pct": 0},
				"narrative": "txn_case scored 0.60 and was routed to manual review.",
				"feature_contributions": [],
				"model_scores": {"lightgbm": 0.6}
			}`))
		case r.URL.Path == "/cases":
			w.Write([]byte(`{"cases": [{"id": 1, "transaction_id": "txn_case", "action": "step_up_review"}]}`))
		case r.URL.Path == "/cases/1/resolve":
			w.Write([]byte(`{"case_id": 1, "verdict": "approve"}`))
		case r.URL.Path == "/cases/999/resolve":
			http.NotFound(w, r)
		case r.URL.Path == "/analytics/summary":
			w.Write([]byte(`{
				"funnel": [{"action": "approve", "transaction_count": 10, "pct_of_volume": 100, "total_amount": 500, "avg_fraud_score": 0.1}],
				"score_deciles": [{"score_decile": 0, "transaction_count": 10, "confirmed_fraud_count": 0, "fraud_rate_pct": 0}]
			}`))
		default:
			http.NotFound(w, r)
		}
	}))
}

func TestDecisionHandler_Success(t *testing.T) {
	server := fakeMLServiceServer(t)
	defer server.Close()

	h := testHandlersWithMLService(t, server.URL)
	body := map[string]interface{}{
		"transaction": map[string]interface{}{"time": 1000, "amount": 500, "transaction_id": "txn_case"},
		"account":     map[string]interface{}{"credit_limit": 5000},
	}
	w := doRequest(h.DecisionHandler, http.MethodPost, "/api/v1/decision", body)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if resp["action"] != "step_up_review" {
		t.Errorf("action = %v, want step_up_review", resp["action"])
	}
	if resp["narrative"] == "" || resp["narrative"] == nil {
		t.Error("expected a non-empty narrative")
	}
}

func TestCasesHandler_Success(t *testing.T) {
	server := fakeMLServiceServer(t)
	defer server.Close()

	h := testHandlersWithMLService(t, server.URL)
	w := doRequest(h.CasesHandler, http.MethodGet, "/api/v1/cases", nil)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	cases, ok := resp["cases"].([]interface{})
	if !ok || len(cases) != 1 {
		t.Errorf("cases = %v, want 1 case", resp["cases"])
	}
}

func TestResolveCaseHandler_Success(t *testing.T) {
	server := fakeMLServiceServer(t)
	defer server.Close()

	h := testHandlersWithMLService(t, server.URL)
	body := map[string]interface{}{"verdict": "approve", "is_actual_fraud": false}
	w := doRequestRoute(h.ResolveCaseHandler, http.MethodPost, "/api/v1/cases/:id/resolve", "/api/v1/cases/1/resolve", body)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}
}

func TestResolveCaseHandler_NotFound(t *testing.T) {
	server := fakeMLServiceServer(t)
	defer server.Close()

	h := testHandlersWithMLService(t, server.URL)
	body := map[string]interface{}{"verdict": "approve"}
	w := doRequestRoute(h.ResolveCaseHandler, http.MethodPost, "/api/v1/cases/:id/resolve", "/api/v1/cases/999/resolve", body)

	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusNotFound)
	}
}

func TestAnalyticsSummaryHandler_Success(t *testing.T) {
	server := fakeMLServiceServer(t)
	defer server.Close()

	h := testHandlersWithMLService(t, server.URL)
	w := doRequest(h.AnalyticsSummaryHandler, http.MethodGet, "/api/v1/analytics/summary", nil)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	funnel, ok := resp["funnel"].([]interface{})
	if !ok || len(funnel) != 1 {
		t.Errorf("funnel = %v, want 1 row", resp["funnel"])
	}
}

func TestAnalyticsSummaryHandler_ServiceUnavailable(t *testing.T) {
	h := testHandlers(t) // ml-service unreachable
	w := doRequest(h.AnalyticsSummaryHandler, http.MethodGet, "/api/v1/analytics/summary", nil)

	// ml_service is registered even when unreachable (health-check driven),
	// so this exercises the "sidecar down" -> 500 path via GetAnalyticsSummary's error.
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusInternalServerError)
	}
}

func TestResolveCaseHandler_InvalidCaseID(t *testing.T) {
	h := testHandlers(t)
	body := map[string]interface{}{"verdict": "approve"}
	w := doRequestRoute(h.ResolveCaseHandler, http.MethodPost, "/api/v1/cases/:id/resolve", "/api/v1/cases/not-a-number/resolve", body)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusBadRequest)
	}
}
