package main

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"fraud-detection-inference/internal/api"
	"fraud-detection-inference/internal/ml"
	"github.com/sirupsen/logrus"
)

// TestSetupRouter_DoesNotPanic guards against duplicate route registrations
// (e.g. GET /metrics registered twice), which makes Gin panic at startup.
func TestSetupRouter_DoesNotPanic(t *testing.T) {
	logger := logrus.New()
	logger.SetOutput(discard{})

	manager := ml.NewModelManager(logger, "models", "http://127.0.0.1:1")
	if err := manager.LoadModels(); err != nil {
		t.Fatalf("LoadModels() error = %v", err)
	}
	handlers := api.NewHandlers(manager, logger, "test")

	router := setupRouter(handlers, logger)

	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("GET /health status = %d, want %d", w.Code, http.StatusOK)
	}
}

type discard struct{}

func (discard) Write(p []byte) (int, error) { return len(p), nil }
