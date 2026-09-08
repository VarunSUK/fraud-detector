# Multi-stage Dockerfile for Go inference service
FROM golang:1.19-alpine AS builder

# Set working directory
WORKDIR /app

# Install dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/server

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS requests
RUN apk --no-cache add ca-certificates tzdata

# Create non-root user
RUN adduser -D -s /bin/sh appuser

# Set working directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/main .

# No models/ directory to copy: this service no longer reads model files
# from local disk (MODELS_DIR is a vestigial config value, unused by
# predictor.go) -- real models are loaded and served by the ml-serving
# sidecar over HTTP. A COPY here used to reference a directory that never
# existed in the repo, which broke this image's build outright.

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Default environment variables
ENV PORT=8080
ENV MODELS_DIR=/app/models
ENV LOG_LEVEL=info
ENV VERSION=1.0.0

# Run the application
CMD ["./main"]

