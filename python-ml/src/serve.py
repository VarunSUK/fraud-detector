"""
FastAPI serving layer for trained fraud detection models.

Loads the LightGBM/XGBoost/Isolation-Forest artifacts produced by train.py
from MODELS_DIR and exposes them over HTTP so other services (e.g. the Go
inference service) can request real scores instead of a hardcoded mock.
"""

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import AnomalyDetector, FraudDetectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPERVISED_MODEL_NAMES = ("lightgbm", "xgboost")
ANOMALY_MODEL_NAME = "isolation_forest"

# Weighted blend of the supervised (calibrated-probability) models and the
# unsupervised anomaly detector. The anomaly score isn't a calibrated fraud
# probability, so it's given a smaller vote -- this mirrors how production
# fraud systems combine heterogeneous signals (classifiers, anomaly scores,
# rules) into one decision score rather than trusting any single model.
ENSEMBLE_WEIGHTS = {"lightgbm": 0.4, "xgboost": 0.4, ANOMALY_MODEL_NAME: 0.2}


class TransactionPayload(BaseModel):
    """Mirrors go-inference's Transaction JSON shape (creditcard dataset features)."""

    time: float = 0.0
    amount: float = 0.0
    transaction_id: str = ""
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    v5: float = 0.0
    v6: float = 0.0
    v7: float = 0.0
    v8: float = 0.0
    v9: float = 0.0
    v10: float = 0.0
    v11: float = 0.0
    v12: float = 0.0
    v13: float = 0.0
    v14: float = 0.0
    v15: float = 0.0
    v16: float = 0.0
    v17: float = 0.0
    v18: float = 0.0
    v19: float = 0.0
    v20: float = 0.0
    v21: float = 0.0
    v22: float = 0.0
    v23: float = 0.0
    v24: float = 0.0
    v25: float = 0.0
    v26: float = 0.0
    v27: float = 0.0
    v28: float = 0.0


def to_creditcard_frame(payload: TransactionPayload) -> pd.DataFrame:
    """Build the single-row DataFrame FeatureEngineer expects for the creditcard dataset."""
    row = {"Time": payload.time, "Amount": payload.amount}
    for i in range(1, 29):
        row[f"V{i}"] = getattr(payload, f"v{i}")
    return pd.DataFrame([row])


class ModelStore:
    """Loads and holds the trained models available under models_dir, and
    combines their outputs into a single weighted ensemble score."""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.supervised_models: Dict[str, FraudDetectionModel] = {}
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self._load()

    def _load(self) -> None:
        for name in SUPERVISED_MODEL_NAMES:
            path_prefix = os.path.join(self.models_dir, name)
            metadata_path = f"{path_prefix}_metadata.json"
            if not os.path.exists(metadata_path):
                logger.warning("Model artifacts not found for %s at %s", name, path_prefix)
                continue
            try:
                model = FraudDetectionModel(model_type=name)
                model.load_model(path_prefix)
                self.supervised_models[name] = model
                logger.info("Loaded %s model from %s", name, path_prefix)
            except Exception:
                logger.exception("Failed to load %s model from %s", name, path_prefix)

        anomaly_prefix = os.path.join(self.models_dir, ANOMALY_MODEL_NAME)
        if os.path.exists(f"{anomaly_prefix}_features.joblib"):
            try:
                detector = AnomalyDetector()
                detector.load(anomaly_prefix)
                self.anomaly_detector = detector
                logger.info("Loaded anomaly detector from %s", anomaly_prefix)
            except Exception:
                logger.exception("Failed to load anomaly detector from %s", anomaly_prefix)
        else:
            logger.warning("Anomaly detector artifacts not found at %s", anomaly_prefix)

    @property
    def loaded_names(self) -> List[str]:
        names = sorted(self.supervised_models.keys())
        if self.anomaly_detector is not None:
            names.append(ANOMALY_MODEL_NAME)
        return names

    def is_ready(self) -> bool:
        # At least one calibrated supervised model is required -- the anomaly
        # detector alone doesn't produce a meaningful fraud probability.
        return len(self.supervised_models) > 0

    def predict_score(self, frame: pd.DataFrame) -> Dict:
        """Returns {"score": weighted ensemble score, "components": {model_name: score}}."""
        components: Dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0

        for name, model in self.supervised_models.items():
            _, proba = model.predict(frame)
            score = float(proba[0])
            components[name] = score
            weight = ENSEMBLE_WEIGHTS.get(name, 1.0)
            weighted_sum += score * weight
            weight_total += weight

        if self.anomaly_detector is not None:
            score = float(self.anomaly_detector.anomaly_score(frame)[0])
            components[ANOMALY_MODEL_NAME] = score
            weight = ENSEMBLE_WEIGHTS.get(ANOMALY_MODEL_NAME, 0.0)
            weighted_sum += score * weight
            weight_total += weight

        if weight_total == 0:
            raise RuntimeError("no models loaded")

        return {"score": weighted_sum / weight_total, "components": components}

    def explain(self, frame: pd.DataFrame) -> Dict:
        model = self.supervised_models.get("lightgbm") or next(iter(self.supervised_models.values()), None)
        if model is None:
            raise RuntimeError("no models loaded")
        return model.explain_prediction(frame, instance_idx=0)


def create_app(models_dir: Optional[str] = None) -> FastAPI:
    """Build a FastAPI app bound to a ModelStore for the given models_dir.

    Exposed as a factory (rather than a bare module-level app) so tests can
    point at an isolated models directory without touching process env/state.
    """
    resolved_dir = models_dir or os.environ.get("MODELS_DIR", "models")
    store = ModelStore(resolved_dir)

    app = FastAPI(title="Fraud Detection ML Service")
    app.state.store = store

    @app.get("/health")
    def health():
        return {
            "status": "healthy" if store.is_ready() else "degraded",
            "models_loaded": store.loaded_names,
        }

    @app.post("/predict")
    def predict(payload: TransactionPayload):
        if not store.is_ready():
            raise HTTPException(status_code=503, detail="no models loaded")

        frame = to_creditcard_frame(payload)
        try:
            result = store.predict_score(frame)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return {
            "transaction_id": payload.transaction_id,
            "score": result["score"],
            "prediction": int(result["score"] > 0.5),
            "model_scores": result["components"],
        }

    @app.post("/explain")
    def explain(payload: TransactionPayload):
        if not store.is_ready():
            raise HTTPException(status_code=503, detail="no models loaded")

        frame = to_creditcard_frame(payload)
        try:
            result = store.predict_score(frame)
            explanation = store.explain(frame)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return {
            "transaction_id": payload.transaction_id,
            "score": result["score"],
            "prediction": int(result["score"] > 0.5),
            "feature_contributions": explanation.get("feature_contributions", []),
            "model_scores": result["components"],
        }

    return app


app = create_app()
