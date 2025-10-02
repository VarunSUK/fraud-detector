package api

import (
	"net/http"
	"strconv"
	"time"

	"fraud-detection-inference/internal/ml"
	"fraud-detection-inference/internal/models"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/sirupsen/logrus"
)

// Prometheus metrics
var (
	requestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "fraud_detection_requests_total",
			Help: "Total number of fraud detection requests",
		},
		[]string{"endpoint", "method", "status"},
	)

	requestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "fraud_detection_request_duration_seconds",
			Help:    "Duration of fraud detection requests",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"endpoint", "method"},
	)

	predictionsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "fraud_detection_predictions_total",
			Help: "Total number of fraud predictions",
		},
		[]string{"model", "prediction"},
	)

	modelLoadTime = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "fraud_detection_model_load_time_seconds",
			Help: "Time taken to load models",
		},
		[]string{"model"},
	)
)

// Handlers contains all HTTP handlers
type Handlers struct {
	modelManager *ml.ModelManager
	logger       *logrus.Logger
	version      string
}

// NewHandlers creates a new handlers instance
func NewHandlers(modelManager *ml.ModelManager, logger *logrus.Logger, version string) *Handlers {
	return &Handlers{
		modelManager: modelManager,
		logger:       logger,
		version:      version,
	}
}

// ScoreHandler handles fraud scoring requests
func (h *Handlers) ScoreHandler(c *gin.Context) {
	start := time.Now()
	
	// Parse request
	var req models.ScoreRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.WithError(err).Error("Invalid request format")
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:     "Invalid request format",
			Message:   err.Error(),
			Timestamp: time.Now(),
		})
		requestsTotal.WithLabelValues("score", "POST", "400").Inc()
		return
	}
	
	// Validate transaction
	if err := req.Transaction.Validate(); err != nil {
		h.logger.WithError(err).Error("Transaction validation failed")
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:     "Transaction validation failed",
			Message:   err.Error(),
			Timestamp: time.Now(),
		})
		requestsTotal.WithLabelValues("score", "POST", "400").Inc()
		return
	}
	
	// Get model name from query parameter (default to ensemble)
	modelName := c.DefaultQuery("model", "ensemble")
	
	// Get predictor
	predictor, exists := h.modelManager.GetPredictor(modelName)
	if !exists {
		h.logger.WithField("model", modelName).Error("Model not found")
		c.JSON(http.StatusNotFound, models.ErrorResponse{
			Error:     "Model not found",
			Message:   "Requested model is not available",
			Timestamp: time.Now(),
		})
		requestsTotal.WithLabelValues("score", "POST", "404").Inc()
		return
	}
	
	// Check if model is loaded
	if !predictor.IsLoaded() {
		h.logger.WithField("model", modelName).Error("Model not loaded")
		c.JSON(http.StatusServiceUnavailable, models.ErrorResponse{
			Error:     "Model not loaded",
			Message:   "Requested model is not loaded",
			Timestamp: time.Now(),
		})
		requestsTotal.WithLabelValues("score", "POST", "503").Inc()
		return
	}
	
	// Make prediction
	predictionStart := time.Now()
	score, prediction, err := predictor.Predict(req.Transaction)
	predictionDuration := time.Since(predictionStart)
	
	if err != nil {
		h.logger.WithError(err).WithField("model", modelName).Error("Prediction failed")
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{
			Error:     "Prediction failed",
			Message:   err.Error(),
			Timestamp: time.Now(),
		})
		requestsTotal.WithLabelValues("score", "POST", "500").Inc()
		return
	}
	
	// Create response
	response := models.ScoreResponse{
		TransactionID: req.Transaction.TransactionID,
		Score:         score,
		Prediction:    prediction,
		Probability:   score,
		Model:         modelName,
		Timestamp:     time.Now(),
		ProcessingMs:  predictionDuration.Milliseconds(),
	}
	
	// Log prediction
	h.logger.WithFields(logrus.Fields{
		"transaction_id": req.Transaction.TransactionID,
		"model":          modelName,
		"score":          score,
		"prediction":     prediction,
		"processing_ms":  predictionDuration.Milliseconds(),
	}).Info("Fraud prediction completed")
	
	// Update metrics
	predictionsTotal.WithLabelValues(modelName, strconv.Itoa(prediction)).Inc()
	requestsTotal.WithLabelValues("score", "POST", "200").Inc()
	requestDuration.WithLabelValues("score", "POST").Observe(time.Since(start).Seconds())
	
	c.JSON(http.StatusOK, response)
}

// ExplainHandler handles prediction explanation requests
func (h *Handlers) ExplainHandler(c *gin.Context) {
	start := time.Now()
	
	// Parse request
	var req models.ExplainRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.WithError(err).Error("Invalid request format")
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:     "Invalid request format",
			Message:   err.Error(),
			Timestamp: time.Now(),
		})
		requestsTotal.WithLabelValues("explain", "POST", "400").Inc()
		return
	}
	
	// Validate transaction
	if err := req.Transaction.Validate(); err != nil {
		h.logger.WithError(err).Error("Transaction validation failed")
		c.JSON(http.StatusBadRequest, models.ErrorResponse{
			Error:     "Transaction validation failed",
			Message:   err.Error(),
			Timestamp: time.Now(),
		})
		requestsTotal.WithLabelValues("explain", "POST", "400").Inc()
		return
	}
	
	// Get model name from query parameter (default to ensemble)
	modelName := c.DefaultQuery("model", "ensemble")
	
	// Get predictor
	predictor, exists := h.modelManager.GetPredictor(modelName)
	if !exists {
		h.logger.WithField("model", modelName).Error("Model not found")
		c.JSON(http.StatusNotFound, models.ErrorResponse{
			Error:     "Model not found",
			Message:   "Requested model is not available",
			Timestamp: time.Now(),
		})
		requestsTotal.WithLabelValues("explain", "POST", "404").Inc()
		return
	}
	
	// Check if model is loaded
	if !predictor.IsLoaded() {
		h.logger.WithField("model", modelName).Error("Model not loaded")
		c.JSON(http.StatusServiceUnavailable, models.ErrorResponse{
			Error:     "Model not loaded",
			Message:   "Requested model is not loaded",
			Timestamp: time.Now(),
		})
		requestsTotal.WithLabelValues("explain", "POST", "503").Inc()
		return
	}
	
	// Get explanation
	explanationStart := time.Now()
	response, err := predictor.Explain(req.Transaction)
	explanationDuration := time.Since(explanationStart)
	
	if err != nil {
		h.logger.WithError(err).WithField("model", modelName).Error("Explanation failed")
		c.JSON(http.StatusInternalServerError, models.ErrorResponse{
			Error:     "Explanation failed",
			Message:   err.Error(),
			Timestamp: time.Now(),
		})
		requestsTotal.WithLabelValues("explain", "POST", "500").Inc()
		return
	}
	
	// Update processing time
	response.ProcessingMs = explanationDuration.Milliseconds()
	response.Model = modelName
	
	// Log explanation request
	h.logger.WithFields(logrus.Fields{
		"transaction_id": req.Transaction.TransactionID,
		"model":          modelName,
		"processing_ms":  explanationDuration.Milliseconds(),
	}).Info("Prediction explanation completed")
	
	// Update metrics
	requestsTotal.WithLabelValues("explain", "POST", "200").Inc()
	requestDuration.WithLabelValues("explain", "POST").Observe(time.Since(start).Seconds())
	
	c.JSON(http.StatusOK, response)
}

// HealthHandler handles health check requests
func (h *Handlers) HealthHandler(c *gin.Context) {
	start := time.Now()
	
	// Get available models
	availableModels := h.modelManager.GetAvailableModels()
	
	// Check if at least one model is loaded
	status := "healthy"
	for _, modelName := range availableModels {
		predictor, exists := h.modelManager.GetPredictor(modelName)
		if exists && predictor.IsLoaded() {
			status = "healthy"
			break
		}
		status = "unhealthy"
	}
	
	response := models.HealthResponse{
		Status:    status,
		Timestamp: time.Now(),
		Version:   h.version,
		Models:    availableModels,
	}
	
	// Update metrics
	requestsTotal.WithLabelValues("health", "GET", "200").Inc()
	requestDuration.WithLabelValues("health", "GET").Observe(time.Since(start).Seconds())
	
	c.JSON(http.StatusOK, response)
}

// ModelsHandler returns information about available models
func (h *Handlers) ModelsHandler(c *gin.Context) {
	start := time.Now()
	
	availableModels := h.modelManager.GetAvailableModels()
	var modelInfos []models.ModelInfo
	
	for _, modelName := range availableModels {
		predictor, exists := h.modelManager.GetPredictor(modelName)
		if exists {
			modelInfos = append(modelInfos, *predictor.GetModelInfo())
		}
	}
	
	// Update metrics
	requestsTotal.WithLabelValues("models", "GET", "200").Inc()
	requestDuration.WithLabelValues("models", "GET").Observe(time.Since(start).Seconds())
	
	c.JSON(http.StatusOK, gin.H{
		"models": modelInfos,
		"count":  len(modelInfos),
	})
}

// MetricsHandler exposes Prometheus metrics
func (h *Handlers) MetricsHandler(c *gin.Context) {
	// This would typically use promhttp.Handler() in a real implementation
	// For now, return a simple response
	c.JSON(http.StatusOK, gin.H{
		"message": "Metrics endpoint - use /metrics for Prometheus format",
	})
}

// CORSMiddleware handles CORS
func CORSMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	}
}

