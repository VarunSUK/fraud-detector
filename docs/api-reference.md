# API Reference

Two HTTP services make up the API surface:

- **go-inference** (default `:8080`) -- the public-facing service. Everything under this heading is served here unless noted otherwise.
- **ml-serving sidecar** (default `:8000`) -- the Python FastAPI service go-inference calls internally. Documented separately at the bottom since most consumers should talk to go-inference, not this directly.

All request/response bodies are JSON. There is no authentication in this repo (see [Security Considerations](../README.md#-security-considerations) in the README).

## go-inference

### `GET /health`
Health check. `status` is `"healthy"` if at least one predictor (rule-based, ml_service, or ensemble) reports loaded.

```json
{ "status": "healthy", "timestamp": "2026-01-01T12:00:00Z", "version": "1.0.0", "models": ["rule_based", "ml_service", "ensemble"] }
```

### `GET /api/v1/models`
Lists loaded models and their metadata (feature list, metrics, load time). `ml_service`'s `metrics.num_models_loaded` reflects how many of lightgbm/xgboost/isolation_forest the sidecar actually has loaded.

### `POST /api/v1/score`
Scores a transaction with the ensemble (rule-based + ml_service, weighted). Query param `?model=` selects `rule_based`, `ml_service`, or `ensemble` (default) explicitly.

**Request:**
```json
{ "transaction": { "time": 90000, "amount": 2500, "transaction_id": "txn_1", "v14": -3.2, "v4": 2.1 } }
```
`v1`..`v28` default to `0` if omitted. `time` and `amount` are required (non-negative).

**Response:**
```json
{ "transaction_id": "txn_1", "score": 0.74, "prediction": 1, "probability": 0.74, "model": "ensemble", "timestamp": "...", "processing_ms": 12 }
```

### `POST /api/v1/explain`
Same request shape as `/score`. Returns real SHAP-value feature attributions from whichever predictor is loaded (prefers `ml_service` over `rule_based`), plus `model_scores` -- the per-model components (`lightgbm`, `xgboost`, `isolation_forest`) that were blended into the score, when backed by the sidecar.

```json
{
  "transaction_id": "txn_1", "score": 0.74, "prediction": 1,
  "feature_contributions": [ { "feature": "V14", "value": -3.2, "importance": 2.98, "contribution": 2.98 } ],
  "model_scores": { "lightgbm": 0.91, "xgboost": 0.71, "isolation_forest": 0.47 },
  "model": "ml_service", "timestamp": "...", "processing_ms": 45
}
```
`contribution` is signed: positive pushes toward fraud, negative toward legitimate. Sorted by `|contribution|` descending.

### `POST /api/v1/decision`
Scores the transaction, applies the credit risk policy (see [ml-pipeline.md](./ml-pipeline.md#credit-risk-decisioning)), records it to the audit log, and returns an actionable result. Requires the ml-serving sidecar (`503` if unreachable or has no models loaded).

**Request:**
```json
{
  "transaction": { "time": 90000, "amount": 2500, "transaction_id": "txn_1", "v14": -3.2, "v4": 2.1 },
  "account": { "credit_limit": 5000, "current_balance": 1000, "account_age_days": 400, "delinquent_payments_count": 0, "avg_monthly_spend": 800 }
}
```
`account` is optional; omitted fields default to `0`.

**Response:** see [README § Credit Risk Decision](../README.md#-credit-risk-decision) for a full example. Key fields: `action` (`approve`/`step_up_review`/`decline`), `risk_tier`, `reason_codes`, `credit_limit_recommendation`, `narrative`, `feature_contributions`, `model_scores`.

### `GET /api/v1/cases`
Lists pending `step_up_review` decisions awaiting an analyst verdict (`analyst_verdict IS NULL`), newest first. Optional `?limit=` (default 50).

### `POST /api/v1/cases/:id/resolve`
Records an analyst's verdict.
```json
{ "verdict": "approve", "is_actual_fraud": false }
```
`verdict` must be `"approve"` or `"decline"`. `404` if the case doesn't exist. `is_actual_fraud` is optional -- set it when the true outcome is known, since it's what the analytics queries key off of.

### `GET /api/v1/analytics/summary`
Live approval funnel and score-decile breakdown -- see [README § Analytics Summary](../README.md#-analytics-summary). For deeper analysis (threshold trade-offs, review queue aging), run `python analytics/run_report.py` directly against the audit database; that's intentionally not exposed as an API (see [ml-pipeline.md](./ml-pipeline.md) for why).

### `GET /metrics`
Real Prometheus exposition format (`promhttp.Handler()`), not a JSON stub. Scraped by `monitoring/prometheus/prometheus.yml` at `inference-api:8080/metrics`. Exposes `fraud_detection_requests_total`, `fraud_detection_request_duration_seconds`, `fraud_detection_predictions_total`, `fraud_detection_model_load_time_seconds`, plus the Go runtime/process default collectors.

Legacy unversioned aliases (`/score`, `/explain`, `/models`, `/metrics`) exist for backward compatibility and behave identically to their `/api/v1/...` counterparts.

## ml-serving sidecar

Called by go-inference's `MLServicePredictor`; documented here for anyone running or debugging the sidecar directly.

| Endpoint | Notes |
|---|---|
| `GET /health` | `{"status": "healthy"\|"degraded", "models_loaded": [...]}`. `"healthy"` requires at least one of lightgbm/xgboost loaded. |
| `POST /predict` | Same transaction shape as go-inference's `/score`. Returns `{transaction_id, score, prediction, model_scores}`. |
| `POST /explain` | Same as `/predict` plus `feature_contributions` (real SHAP values from the LightGBM model). |
| `POST /decision` | `{transaction, account}` -> full decision, same shape go-inference proxies. |
| `GET /cases`, `POST /cases/:id/resolve` | Same as the go-inference proxies. |
| `GET /analytics/summary` | Same as the go-inference proxy. |

All request/response field names in the sidecar are lowercase (`v1`..`v28`, `time`, `amount`) even though the underlying models are trained on the creditcard dataset's uppercase columns (`V1`..`V28`, `Time`, `Amount`) -- `serve.py`'s `to_creditcard_frame()` does that translation so the wire format matches go-inference's `Transaction` JSON tags exactly.
