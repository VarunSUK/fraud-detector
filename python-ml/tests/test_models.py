import os

import numpy as np

from data_generator import TransactionGenerator
from models import AnomalyDetector, FraudDetectionModel


def _synthetic_dataset(num_users=150, transactions_per_user=20):
    generator = TransactionGenerator(seed=11)
    df = generator.generate_transaction_stream(num_users=num_users, transactions_per_user=transactions_per_user)
    drop_cols = ["is_fraud", "fraud_type", "transaction_id", "timestamp", "ip_address"]
    X = df.drop(columns=drop_cols)
    y = df["is_fraud"]
    return X, y


def test_train_lightgbm_produces_valid_metrics():
    X, y = _synthetic_dataset()
    model = FraudDetectionModel(model_type="lightgbm", dataset_type="synthetic")

    metrics = model.train(X, y)

    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert model.feature_importance is not None


def test_predict_returns_matching_shapes_and_valid_probabilities():
    X, y = _synthetic_dataset()
    model = FraudDetectionModel(model_type="lightgbm", dataset_type="synthetic")
    model.train(X, y)

    y_pred, y_pred_proba = model.predict(X)

    assert len(y_pred) == len(X)
    assert len(y_pred_proba) == len(X)
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1))
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_save_and_load_model_round_trip(tmp_path):
    X, y = _synthetic_dataset()
    model = FraudDetectionModel(model_type="lightgbm", dataset_type="synthetic")
    model.train(X, y)

    filepath = os.path.join(tmp_path, "lightgbm")
    model.save_model(filepath)

    assert os.path.exists(f"{filepath}_model.joblib")
    assert os.path.exists(f"{filepath}_features.joblib")
    assert os.path.exists(f"{filepath}_metadata.json")

    reloaded = FraudDetectionModel(model_type="lightgbm", dataset_type="synthetic")
    reloaded.load_model(filepath)

    _, original_proba = model.predict(X)
    _, reloaded_proba = reloaded.predict(X)
    np.testing.assert_allclose(original_proba, reloaded_proba)


def test_explain_prediction_returns_sorted_contributions():
    # Contributions are real (signed) SHAP values: positive pushes toward fraud,
    # negative pushes toward legitimate. They're ranked by magnitude of impact,
    # not by raw sign, so the |contribution| sequence should be non-increasing.
    X, y = _synthetic_dataset()
    model = FraudDetectionModel(model_type="lightgbm", dataset_type="synthetic")
    model.train(X, y)

    explanation = model.explain_prediction(X, instance_idx=0)

    contributions = explanation["feature_contributions"]
    assert len(contributions) > 0
    magnitudes = [abs(c["contribution"]) for c in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for c in contributions:
        assert c["importance"] == abs(c["contribution"])


def test_explain_prediction_works_after_save_and_load_round_trip(tmp_path):
    # Regression test: save_model used to JSON-dump feature_importance with
    # default=str, which silently stringified the numpy floats. explain_prediction
    # would then crash multiplying a string by a float on any reloaded model.
    X, y = _synthetic_dataset()
    model = FraudDetectionModel(model_type="lightgbm", dataset_type="synthetic")
    model.train(X, y)

    filepath = os.path.join(tmp_path, "lightgbm")
    model.save_model(filepath)

    reloaded = FraudDetectionModel(model_type="lightgbm", dataset_type="synthetic")
    reloaded.load_model(filepath)

    for importance in reloaded.feature_importance.values():
        assert isinstance(importance, float)

    explanation = reloaded.explain_prediction(X, instance_idx=0)
    assert len(explanation["feature_contributions"]) > 0


def test_train_ensemble_models_writes_artifacts_for_both_model_types(trained_models_dir):
    for name in ("lightgbm", "xgboost"):
        prefix = os.path.join(trained_models_dir, name)
        assert os.path.exists(f"{prefix}_model.joblib")
        assert os.path.exists(f"{prefix}_base_model.joblib")
        assert os.path.exists(f"{prefix}_features.joblib")
        assert os.path.exists(f"{prefix}_metadata.json")

    isolation_prefix = os.path.join(trained_models_dir, "isolation_forest")
    assert os.path.exists(f"{isolation_prefix}_model.joblib")
    assert os.path.exists(f"{isolation_prefix}_features.joblib")

    assert os.path.exists(os.path.join(trained_models_dir, "ensemble_results.json"))


def test_anomaly_detector_scores_are_bounded_and_unsupervised(tmp_path):
    X, y = _synthetic_dataset()

    detector = AnomalyDetector(dataset_type="synthetic")
    metrics = detector.train(X, y)

    assert 0.0 <= metrics["auc_vs_labels"] <= 1.0

    scores = detector.anomaly_score(X)
    assert len(scores) == len(X)
    assert np.all((scores > 0) & (scores < 1))


def test_anomaly_detector_save_and_load_round_trip(tmp_path):
    X, _ = _synthetic_dataset()

    detector = AnomalyDetector(dataset_type="synthetic")
    detector.train(X)

    filepath = os.path.join(tmp_path, "isolation_forest")
    detector.save(filepath)

    reloaded = AnomalyDetector(dataset_type="synthetic")
    reloaded.load(filepath)

    np.testing.assert_allclose(detector.anomaly_score(X), reloaded.anomaly_score(X))


def test_calibrated_model_predict_proba_matches_predict():
    # The calibrated model (self.model) should be what backs predict()/predict_proba(),
    # not the raw uncalibrated booster (self.base_model).
    X, y = _synthetic_dataset()
    model = FraudDetectionModel(model_type="lightgbm", dataset_type="synthetic")
    model.train(X, y)

    assert model.model is not None
    assert model.base_model is not None
    assert model.model is not model.base_model

    y_pred, y_pred_proba = model.predict(X)
    assert np.array_equal(y_pred, (y_pred_proba > 0.5).astype(int))
