from fastapi.testclient import TestClient

from serve import TransactionPayload, create_app, to_creditcard_frame


def test_health_degraded_when_no_models(tmp_path):
    app = create_app(models_dir=str(tmp_path))
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["models_loaded"] == []


def test_predict_returns_503_when_no_models(tmp_path):
    app = create_app(models_dir=str(tmp_path))
    client = TestClient(app)

    resp = client.post("/predict", json={"time": 1000, "amount": 50})

    assert resp.status_code == 503


def test_explain_returns_503_when_no_models(tmp_path):
    app = create_app(models_dir=str(tmp_path))
    client = TestClient(app)

    resp = client.post("/explain", json={"time": 1000, "amount": 50})

    assert resp.status_code == 503


def test_health_healthy_when_models_loaded(trained_models_dir):
    app = create_app(models_dir=trained_models_dir)
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert set(body["models_loaded"]) == {"lightgbm", "xgboost", "isolation_forest"}


def test_predict_returns_score_and_prediction(trained_models_dir):
    app = create_app(models_dir=trained_models_dir)
    client = TestClient(app)

    payload = {"time": 90000, "amount": 2500, "transaction_id": "txn_1", "v14": -4.0, "v4": 3.0}
    resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == "txn_1"
    assert 0.0 <= body["score"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert set(body["model_scores"].keys()) == {"lightgbm", "xgboost", "isolation_forest"}
    for component_score in body["model_scores"].values():
        assert 0.0 <= component_score <= 1.0


def test_predict_flags_fraud_like_transaction_as_higher_risk(trained_models_dir):
    app = create_app(models_dir=trained_models_dir)
    client = TestClient(app)

    normal = {"time": 1000, "amount": 25, "v14": 0.0, "v4": 0.0}
    fraud_like = {"time": 1000, "amount": 2500, "v14": -4.0, "v4": 3.0}

    normal_score = client.post("/predict", json=normal).json()["score"]
    fraud_score = client.post("/predict", json=fraud_like).json()["score"]

    assert fraud_score > normal_score


def test_explain_returns_feature_contributions(trained_models_dir):
    app = create_app(models_dir=trained_models_dir)
    client = TestClient(app)

    payload = {"time": 90000, "amount": 2500, "transaction_id": "txn_2", "v14": -4.0, "v4": 3.0}
    resp = client.post("/explain", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == "txn_2"
    assert len(body["feature_contributions"]) > 0
    for contribution in body["feature_contributions"]:
        assert set(contribution.keys()) == {"feature", "value", "importance", "contribution"}
    assert set(body["model_scores"].keys()) == {"lightgbm", "xgboost", "isolation_forest"}


def test_to_creditcard_frame_maps_lowercase_fields_to_uppercase_columns():
    payload = TransactionPayload(time=5.0, amount=10.0, v1=0.5, v28=-0.5)

    frame = to_creditcard_frame(payload)

    assert frame.loc[0, "Time"] == 5.0
    assert frame.loc[0, "Amount"] == 10.0
    assert frame.loc[0, "V1"] == 0.5
    assert frame.loc[0, "V28"] == -0.5
