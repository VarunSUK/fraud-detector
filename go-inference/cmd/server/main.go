package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"fraud-detection-inference/internal/api"
	"fraud-detection-inference/internal/ml"
	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

const (
	DefaultPort         = "8080"
	DefaultModelsDir    = "/app/models"
	DefaultLogLevel     = "info"
	DefaultMLServiceURL = "http://localhost:8000"
)

func main() {
	// Initialize logger
	logger := logrus.New()
	logger.SetFormatter(&logrus.JSONFormatter{})

	// Set log level
	logLevel := getEnv("LOG_LEVEL", DefaultLogLevel)
	level, err := logrus.ParseLevel(logLevel)
	if err != nil {
		logger.WithError(err).Warn("Invalid log level, using info")
		level = logrus.InfoLevel
	}
	logger.SetLevel(level)

	// Get configuration
	port := getEnv("PORT", DefaultPort)
	modelsDir := getEnv("MODELS_DIR", DefaultModelsDir)
	mlServiceURL := getEnv("ML_SERVICE_URL", DefaultMLServiceURL)
	version := getEnv("VERSION", "1.0.0")

	logger.WithFields(logrus.Fields{
		"port":           port,
		"models_dir":     modelsDir,
		"ml_service_url": mlServiceURL,
		"version":        version,
		"log_level":      logLevel,
	}).Info("Starting fraud detection inference service")

	// Initialize model manager
	modelManager := ml.NewModelManager(logger, modelsDir, mlServiceURL)

	// Load models
	if err := modelManager.LoadModels(); err != nil {
		logger.WithError(err).Fatal("Failed to load models")
	}

	// Initialize handlers
	handlers := api.NewHandlers(modelManager, logger, version)

	// Setup router
	router := setupRouter(handlers, logger)

	// Create HTTP server
	server := &http.Server{
		Addr:         ":" + port,
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in goroutine
	go func() {
		logger.WithField("port", port).Info("Starting HTTP server")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.WithError(err).Fatal("Failed to start server")
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logger.WithError(err).Fatal("Server forced to shutdown")
	}

	logger.Info("Server exited")
}

func setupRouter(handlers *api.Handlers, logger *logrus.Logger) *gin.Engine {
	// Set Gin mode
	if gin.Mode() == gin.DebugMode {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// Middleware
	router.Use(gin.Recovery())
	router.Use(api.CORSMiddleware())
	router.Use(loggingMiddleware(logger))

	// Health check endpoint
	router.GET("/health", handlers.HealthHandler)
	router.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "pong"})
	})

	// API routes
	v1 := router.Group("/api/v1")
	{
		v1.POST("/score", handlers.ScoreHandler)
		v1.POST("/explain", handlers.ExplainHandler)
		v1.GET("/models", handlers.ModelsHandler)
		v1.GET("/metrics", handlers.MetricsHandler)
		v1.POST("/decision", handlers.DecisionHandler)
		v1.GET("/cases", handlers.CasesHandler)
		v1.POST("/cases/:id/resolve", handlers.ResolveCaseHandler)
		v1.GET("/analytics/summary", handlers.AnalyticsSummaryHandler)
	}

	// Legacy routes for backward compatibility
	router.POST("/score", handlers.ScoreHandler)
	router.POST("/explain", handlers.ExplainHandler)
	router.GET("/models", handlers.ModelsHandler)
	router.GET("/metrics", handlers.MetricsHandler)

	return router
}

func loggingMiddleware(logger *logrus.Logger) gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		logger.WithFields(logrus.Fields{
			"status":     param.StatusCode,
			"method":     param.Method,
			"path":       param.Path,
			"ip":         param.ClientIP,
			"user_agent": param.Request.UserAgent(),
			"latency":    param.Latency,
			"error":      param.ErrorMessage,
		}).Info("HTTP request")

		return ""
	})
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// Initialize function for testing
func init() {
	// Set default environment variables if not set
	if os.Getenv("PORT") == "" {
		os.Setenv("PORT", DefaultPort)
	}
	if os.Getenv("MODELS_DIR") == "" {
		os.Setenv("MODELS_DIR", DefaultModelsDir)
	}
	if os.Getenv("LOG_LEVEL") == "" {
		os.Setenv("LOG_LEVEL", DefaultLogLevel)
	}
}
