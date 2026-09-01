package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
	logger := logrus.New()
	logger.SetOutput(nowhere{})

	// No ml-serving sidecar reachable in tests, so this exercises the rule-based fallback path.
	manager := ml.NewModelManager(logger, "models", "http://127.0.0.1:1")
	if err := manager.LoadModels(); err != nil {
		t.Fatalf("LoadModels() error = %v", err)
	}
	return NewHandlers(manager, logger, "test")
}

type nowhere struct{}

func (nowhere) Write(p []byte) (int, error) { return len(p), nil }

func doRequest(handler gin.HandlerFunc, method, path string, body interface{}) *httptest.ResponseRecorder {
	router := gin.New()
	router.Handle(method, path, handler)

	var reqBody *bytes.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		reqBody = bytes.NewReader(b)
	} else {
		reqBody = bytes.NewReader(nil)
	}

	req := httptest.NewRequest(method, path, reqBody)
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
