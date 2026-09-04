# Adding a Custom Model to the Ensemble

The ensemble in `python-ml/src/serve.py` is intentionally not a fixed pair -- it's a `ModelStore` that loads whatever's present under `MODELS_DIR` and blends it by name. This walks through adding a fourth model (a plain logistic regression, as a fast/interpretable baseline) end to end.

## 1. Train and save it

Anything with `.fit()` / `.predict_proba()` and a way to serialize itself works. Reuse `FeatureEngineer` so it gets the same processed features as the other models:

```python
# python-ml/src/models.py -- alongside FraudDetectionModel, or in a new module
from sklearn.linear_model import LogisticRegression

class LogisticBaseline:
    def __init__(self, dataset_type: str = 'creditcard'):
        self.feature_engineer = FeatureEngineer(dataset_type=dataset_type)
        self.model = None

    def train(self, X, y):
        X_processed = self.feature_engineer.prepare_features(X, fit=True)
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.model.fit(X_processed, y)

    def predict(self, X):
        X_processed = self.feature_engineer.prepare_features(X, fit=False)
        return self.model.predict(X_processed), self.model.predict_proba(X_processed)[:, 1]

    def save(self, filepath):
        import os, joblib
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, f"{filepath}_model.joblib")
        self.feature_engineer.save(f"{filepath}_features.joblib")

    def load(self, filepath):
        import joblib
        self.model = joblib.load(f"{filepath}_model.joblib")
        self.feature_engineer.load(f"{filepath}_features.joblib")
```

Train and save it to the same `MODELS_DIR` the other models use, under its own name prefix (e.g. `MODELS_DIR/logistic`).

## 2. Load it in `ModelStore`

In `serve.py`:
```python
SUPERVISED_MODEL_NAMES = ("lightgbm", "xgboost", "logistic")
```
`ModelStore._load()` already loops over `SUPERVISED_MODEL_NAMES` and constructs a `FraudDetectionModel(model_type=name)` for each -- if your new model isn't a `FraudDetectionModel`, you'll need a small branch there to construct the right class instead (`LogisticBaseline()` in this example) while keeping everything else (the `try/except` around loading, the `loaded_names` property, `is_ready()`) unchanged.

## 3. Give it a weight

```python
ENSEMBLE_WEIGHTS = {"lightgbm": 0.35, "xgboost": 0.35, "logistic": 0.1, "isolation_forest": 0.2}
```
Weights don't need to sum to 1 -- `predict_score()` divides by `weight_total`, so they're relative. Whatever isn't in `ENSEMBLE_WEIGHTS` defaults to weight `1.0` (see the `.get(name, 1.0)` fallback), so an unlisted model would dominate the average -- always add new models here explicitly.

## 4. That's it for scoring

`/predict`, `/explain`, and `/decision` all call `ModelStore.predict_score()`, so they automatically pick up the new component -- you'll see `"logistic": 0.42` show up in `model_scores` in every response without touching the route handlers. `/health`'s `models_loaded` list picks it up too via `loaded_names`.

## What you get for free vs. what you don't

**Free**: ensemble blending, `model_scores` in API responses, `/health` reporting, Go's `MLServicePredictor` and the frontend's model-breakdown chart (it renders whatever keys are in `model_scores`, no frontend changes needed).

**Not free**: `explain()` (used for `/explain` and `/decision`'s SHAP output) currently only picks `lightgbm` or falls back to the first loaded supervised model (`self.supervised_models.get("lightgbm") or next(iter(...))`). A model without SHAP support (like the logistic regression above) won't break anything as a *scoring* component, but it won't contribute feature-level explanations unless you extend `ModelStore.explain()` to pick a preferred explainer per model type (e.g. `shap.LinearExplainer` for logistic regression, `shap.TreeExplainer` for tree models).
