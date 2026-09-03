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

import audit_log
from credit_decisioning import AccountContext, decide
from models import AnomalyDetector, FraudDetectionModel
from narrative import generate as generate_narrative

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


class AccountContextPayload(BaseModel):
    credit_limit: float = 0.0
    current_balance: float = 0.0
    account_age_days: int = 0
    delinquent_payments_count: int = 0
    avg_monthly_spend: float = 0.0


class DecisionRequestPayload(BaseModel):
    transaction: TransactionPayload
    account: AccountContextPayload = AccountContextPayload()


class ResolveCasePayload(BaseModel):
    verdict: str  # "approve" | "decline"
    is_actual_fraud: Optional[bool] = None


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


def create_app(models_dir: Optional[str] = None, db_path: Optional[str] = None) -> FastAPI:
    """Build a FastAPI app bound to a ModelStore for the given models_dir.

    Exposed as a factory (rather than a bare module-level app) so tests can
    point at an isolated models directory / audit database without touching
    process env/state.
    """
    resolved_dir = models_dir or os.environ.get("MODELS_DIR", "models")
    resolved_db_path = db_path or os.environ.get("AUDIT_DB_PATH", audit_log.DEFAULT_DB_PATH)
    store = ModelStore(resolved_dir)
    audit_log.connect(resolved_db_path).close()  # ensure schema exists up front

    app = FastAPI(title="Fraud Detection ML Service")
    app.state.store = store
    app.state.db_path = resolved_db_path

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

    @app.post("/decision")
    def decision(payload: DecisionRequestPayload):
        """Scores the transaction, applies the credit risk policy, records the
        decision to the audit log, and returns an actionable result -- the
        thing a credit risk workflow actually consumes, as opposed to a bare
        probability."""
        if not store.is_ready():
            raise HTTPException(status_code=503, detail="no models loaded")

        frame = to_creditcard_frame(payload.transaction)
        try:
            result = store.predict_score(frame)
            explanation = store.explain(frame)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        account = AccountContext(**payload.account.model_dump())
        anomaly_score = result["components"].get(ANOMALY_MODEL_NAME, 0.0)
        outcome = decide(result["score"], payload.transaction.amount, account, anomaly_score)

        feature_contributions = explanation.get("feature_contributions", [])
        narrative_text = generate_narrative(
            transaction_id=payload.transaction.transaction_id,
            action=outcome.action,
            risk_tier=outcome.risk_tier,
            reason_codes=outcome.reason_codes,
            feature_contributions=feature_contributions,
            fraud_score=result["score"],
        )

        audit_log.record_decision(
            app.state.db_path,
            transaction_id=payload.transaction.transaction_id,
            amount=payload.transaction.amount,
            fraud_score=result["score"],
            action=outcome.action,
            risk_tier=outcome.risk_tier,
            reason_codes=outcome.reason_codes,
            model_scores=result["components"],
            credit_limit_current=outcome.credit_limit_current,
            credit_limit_recommended=outcome.credit_limit_recommended,
        )

        return {
            "transaction_id": payload.transaction.transaction_id,
            "fraud_score": result["score"],
            "action": outcome.action,
            "risk_tier": outcome.risk_tier,
            "reason_codes": outcome.reason_codes,
            "credit_limit_recommendation": {
                "current": outcome.credit_limit_current,
                "recommended": outcome.credit_limit_recommended,
                "adjustment_pct": outcome.credit_limit_adjustment_pct,
            },
            "narrative": narrative_text,
            "feature_contributions": feature_contributions,
            "model_scores": result["components"],
        }

    @app.get("/analytics/summary")
    def analytics_summary():
        """Live version of analytics/sql/approval_funnel.sql and
        loss_rate_by_score_decile.sql, for the dashboard. For deeper ad hoc
        analysis, run analytics/run_report.py against the same database."""
        return {
            "funnel": audit_log.funnel_summary(app.state.db_path),
            "score_deciles": audit_log.score_decile_summary(app.state.db_path),
        }

    @app.get("/cases")
    def list_cases(limit: int = 50):
        """Pending human-review queue: transactions routed to step_up_review
        that haven't been resolved by an analyst yet."""
        return {"cases": audit_log.list_pending_cases(app.state.db_path, limit=limit)}

    @app.post("/cases/{case_id}/resolve")
    def resolve_case(case_id: int, payload: ResolveCasePayload):
        if payload.verdict not in ("approve", "decline"):
            raise HTTPException(status_code=400, detail="verdict must be 'approve' or 'decline'")

        resolved = audit_log.resolve_case(
            app.state.db_path, case_id, verdict=payload.verdict, is_actual_fraud=payload.is_actual_fraud
        )
        if not resolved:
            raise HTTPException(status_code=404, detail="case not found")

        return {"case_id": case_id, "verdict": payload.verdict}

    return app


app = create_app()
