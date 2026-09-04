# ML Pipeline

## Overview

```
train.py / scripts/seed_audit_log.py
        │
        ▼
train_ensemble_models()  (python-ml/src/models.py)
        │
        ├─► LightGBM  ──┐
        ├─► XGBoost   ──┼─► saved to MODELS_DIR/{name}_model.joblib, _base_model.joblib,
        └─► IsolationForest   _features.joblib, _metadata.json
                            │
                            ▼
                    ml-serving sidecar (serve.py) loads all three at startup
                            │
                            ▼
                    ModelStore.predict_score() -- weighted ensemble
                            │
                            ▼
                    go-inference calls over HTTP (MLServicePredictor)
```

## Data

Two schemas are used, and they are **not interchangeable**:

- **Creditcard schema** (`V1`..`V28`, `Time`, `Amount`, `Class`) -- what the real [Kaggle creditcard.csv dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) uses, and what `serve.py`, go-inference's `Transaction`, and the whole serving path expect. `python-ml/src/synthetic_creditcard.py` generates data in this shape without needing the 150MB real file -- deliberately with class overlap (not a clean linear separation), because a toy model with AUC ≈ 1.0 is a modeling red flag, not something to reproduce.
- **Synthetic schema** (`merchant`, `card_type`, `hour`, `day_of_week`, ...) -- what `python-ml/src/data_generator.py`'s `TransactionGenerator` produces. `FeatureEngineer` and `FraudDetectionModel` both support this via `dataset_type='synthetic'`, and it's exercised by the unit tests, but **models trained on it cannot be served by `serve.py`** -- it hardcodes the creditcard column shape. If you want an end-to-end system on the synthetic schema, you'd need to extend `serve.py`'s `to_creditcard_frame`-equivalent to build the synthetic feature row instead.

## Feature Engineering (`FeatureEngineer` in `models.py`)

For the creditcard schema: log/sqrt/percentile transforms of `Amount`, cyclical (`sin`/`cos`) encodings of hour-of-day and day-of-week derived from `Time`, aggregate statistics across the 28 PCA components (`sum`, `mean`, `std`, `max`, `min`), and an amount z-score. ~49 features total, scaled with `StandardScaler` fit on the training split only.

## Models

### Supervised: LightGBM + XGBoost
Each is trained independently with early stopping on a held-out validation split (`train_test_split` twice: test, then val from the remainder). XGBoost additionally uses `scale_pos_weight` computed from the training split's class balance. Both use SMOTE-compatible imbalanced-learn imports, though the current training path relies on `class_weight='balanced'` / `scale_pos_weight` rather than resampling.

**Probability calibration**: the raw fitted booster (`self.base_model`) is wrapped with `CalibratedClassifierCV(FrozenEstimator(base_model), method='sigmoid')`, fit on the validation split. `self.model` (the calibrated version) is what backs `predict()`/`predict_proba()`. This matters because raw GBM outputs are ranking scores, not probabilities -- a credit risk policy that says "auto-decline above 0.9" needs 0.9 to actually mean something. If calibration fails (e.g. a validation split with only one class, which can happen on tiny toy datasets), it falls back to the uncalibrated model rather than crashing.

### Unsupervised: Isolation Forest (`AnomalyDetector`)
Fits on the same processed features, ignoring labels. Its `decision_function` output is flipped and squashed through a logistic function into `(0, 1)` (higher = more anomalous). This exists because supervised models can only recognize fraud patterns present in historical labels -- an anomaly detector flags transactions that simply look unlike anything seen before, catching novel patterns supervised models miss. It's evaluated against true labels (`auc_vs_labels` in the training metrics) purely for visibility, never for training.

### Ensemble weighting (`serve.py`)
```python
ENSEMBLE_WEIGHTS = {"lightgbm": 0.4, "xgboost": 0.4, "isolation_forest": 0.2}
```
A weighted average of calibrated probabilities and the (uncalibrated) anomaly score. The anomaly detector gets a smaller vote because it isn't a calibrated fraud probability -- this mirrors how production fraud systems blend heterogeneous signals (classifiers, anomaly scores, rules) into one decision score rather than trusting any single model. See [examples/custom-models.md](../examples/custom-models.md) to add a fourth signal.

## Explainability

`explain_prediction()` uses `shap.TreeExplainer` on the raw (uncalibrated) booster -- calibration wrappers don't expose tree structure, so SHAP needs the base model. Contributions are signed (positive = pushes toward fraud) and ranked by `|contribution|`. This replaced an earlier heuristic (`feature_importance * abs(value)`) that wasn't a real attribution method.

## Credit Risk Decisioning

`credit_decisioning.py`'s `decide()` is a rule-based policy layer on top of the ensemble score -- not a model. Score cutoffs (`REVIEW_THRESHOLD=0.45`, `DECLINE_THRESHOLD=0.85`) are starting points, not tuned constants; `analytics/sql/threshold_tradeoff.sql` is how you'd actually size a change to them against realized outcomes before pushing it. The policy also reacts to `amount` (large transactions get a lower review bar) and `account.delinquent_payments_count` (tightens both thresholds and the credit-limit recommendation). See [docs/api-reference.md](./api-reference.md#post-apiv1decision) for the request/response shape.

## Why analytics isn't just another API

`analytics/sql/*.sql` (threshold trade-off, review queue aging) are deliberately *not* wrapped as API endpoints, unlike the funnel/decile summary which is (`/api/v1/analytics/summary`, for the dashboard). The distinction: the dashboard needs a small, stable set of pre-aggregated numbers refreshed on every page load; ad hoc risk analysis needs the actual SQL in front of an analyst who might change the threshold list, add a cohort filter, or join in a new column next week. Wrapping every possible query as an endpoint would mean shipping a query builder; running `analytics/run_report.py` against the real database is simpler and more honest about what it is.

## Training entry points

- `python-ml/train.py` -- CLI training script. `--generate-data` always uses the *synthetic* schema (a pre-existing quirk: it ignores `--dataset-type` when generating data), so it won't produce a model `serve.py` can load. Use `--data-file creditcard.csv --dataset-type creditcard` for a servable model.
- `scripts/seed_audit_log.py --train` -- trains on the creditcard-compatible synthetic generator and is what `serve.py` actually expects; also seeds the audit log with realistic historical decisions. This is the recommended path for local development (see the README Quick Start).
