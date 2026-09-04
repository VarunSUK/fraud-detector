# Troubleshooting

## "Models not found" / ml-serving startup shows `models_loaded: []`

`serve.py` looks for `{MODELS_DIR}/lightgbm_metadata.json` etc. and logs a warning (not an error) if a model's artifacts are missing, then reports `"status": "degraded"` from `/health`. Causes, in order of likelihood:

1. **Nothing has been trained yet.** Run `python scripts/seed_audit_log.py --models-dir <dir> --train` (or `python-ml/train.py`) first.
2. **`MODELS_DIR` points somewhere the training script didn't write to.** They must match exactly -- in Docker Compose this is the shared `ml_models` volume; locally, it's whatever `--models-dir` you passed to training vs. `MODELS_DIR` you set for `serve.py`.
3. **Only some artifacts are present** (e.g. `_model.joblib` exists but `_base_model.joblib` doesn't). This happens if you're loading models saved by an older version of `models.py` before `base_model` was split out for SHAP. Retrain.

## go-inference reports `ensemble` loaded but scores never move / `/api/v1/decision` returns 503

Check `GET /health` on the Go service: if `models` only shows `["rule_based", "ensemble"]` and not `ml_service`, the sidecar is unreachable or `IsLoaded()` is failing its health check. `MLServicePredictor` caches its health status for 5 seconds (`mlServiceHealthCheckInterval`) -- a transient sidecar restart can cause one stale read. Check:
- `ML_SERVICE_URL` is set correctly (defaults to `http://localhost:8000`, which is wrong for Docker Compose/K8s -- see `docker-compose.yml`/Helm's env for the right in-network value).
- The sidecar's own `/health` returns `"status": "healthy"` (not `"degraded"`) -- `/api/v1/decision` specifically requires this, not just "reachable."

## `CalibratedClassifierCV` skipped ("Skipping probability calibration...")

Logged as a warning, not an error, from `FraudDetectionModel.train()`. Happens when the validation split has only one class present -- calibration needs both. This is a training-data-size problem, not a bug: increase the dataset size (`scripts/seed_audit_log.py --num-transactions`) or the minority class rate. The model still works with uncalibrated scores, just less trustworthy as literal probabilities.

## SHAP warning: "LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray"

Harmless version-compatibility warning from the `shap` library, not from this codebase -- `explain_prediction()` already handles both the list-of-arrays and single-array return shapes (`if isinstance(shap_values, list): shap_values = shap_values[1]`). Seen in test output; doesn't indicate a real problem.

## "X does not have valid feature names, but LGBMClassifier was fitted with feature names"

sklearn `UserWarning` from calling `.predict()`/`.predict_proba()` with a NumPy array after fitting on a DataFrame. Cosmetic -- `FeatureEngineer.prepare_features()` intentionally returns a NumPy array (post-scaling), and column-name tracking isn't needed downstream. Safe to ignore.

## Review queue is empty / analytics dashboard shows no score-decile data

Two different things need to be true:
- **Review queue** needs at least one transaction to score in the `[REVIEW_THRESHOLD, DECLINE_THRESHOLD)` band (`0.45`-`0.85` by default) *and* not yet have an `analyst_verdict`. Use the frontend's "Load fraud-like sample" a few times, or check `analytics/sql/approval_funnel.sql`'s `step_up_review` row -- if it's `0`, the model isn't producing scores in that band for whatever you're sending it.
- **Score-decile chart** needs `is_actual_fraud` populated, which only happens when a case is resolved (`POST /cases/:id/resolve` with `is_actual_fraud` set) or via `seed_audit_log.py`'s backdated historical rows. A fresh audit log with only pending cases will show an empty decile chart -- that's correct, not a bug.

## `docker-compose up` — frontend can't reach the API (CORS or connection refused)

The frontend image bakes `VITE_API_BASE_URL` in at *build* time (see [deployment.md](./deployment.md)). If you change the published port or hostname of `inference-api`, you must rebuild the frontend image (`docker-compose build frontend`), not just restart the container -- changing the env var on a running container does nothing, since it's already compiled into the JS bundle. Separately, Go's `CORSMiddleware` allows all origins (`Access-Control-Allow-Origin: *`), so a genuine CORS *error* in the browser console (as opposed to a plain connection failure) usually means the request never reached `inference-api` at all -- check the URL the frontend is actually calling (browser devtools Network tab) against where `inference-api` is actually listening.

## Helm: pods stuck `CrashLoopBackOff` with permission errors

Check whether the failure is from the chart's hardened `securityContext` (`readOnlyRootFilesystem`, `runAsNonRoot`) hitting a path the container needs to write to that isn't one of the `emptyDir`-mounted paths already declared (`/tmp` on the Python services, `/var/cache/nginx`+`/var/run`+`/tmp` on the frontend). If you've forked the chart to add a new component that writes to disk, it'll need the same treatment -- see [deployment.md § Security context](./deployment.md#security-context-and-the-frontend-image).

## Long tail latency under concurrent load

`tests/load-test.js` (`k6 run tests/load-test.js`) surfaced this during development: a run at only 20 concurrent VUs showed a good median (~190ms) but occasional extreme outliers. Root cause was a thundering-herd bug in `MLServicePredictor.IsLoaded()` -- every goroutine that observed a stale health-check cache fired its own redundant HTTP call to the sidecar's `/health` at once, instead of one goroutine refreshing while others waited. Fixed by serializing refreshes through `refreshHealthOnce()` (see `predictor.go`); a regression test (`TestMLServicePredictor_ConcurrentIsLoadedDoesNotStampede`) asserts 50 concurrent stale reads produce exactly one health check. If you see similar tail-latency spikes under load, run `go test -race ./...` first -- that class of bug is what to look for.

## `pytest` is slow / trains real models in test fixtures

`python-ml/tests/conftest.py`'s `trained_models_dir` fixture is session-scoped (trained once, reused across all tests that need it) specifically to keep this fast -- if tests seem to retrain per-test, check that a new test isn't accidentally requesting a differently-parameterized fixture instead of reusing the shared one.
