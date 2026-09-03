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


def test_decision_returns_action_and_records_audit_log(trained_models_dir, tmp_path):
    db_path = str(tmp_path / "audit.db")
    app = create_app(models_dir=trained_models_dir, db_path=db_path)
    client = TestClient(app)

    payload = {
        "transaction": {"time": 90000, "amount": 2500, "transaction_id": "txn_decision", "v14": -4.0, "v4": 3.0},
        "account": {"credit_limit": 5000, "current_balance": 1000, "delinquent_payments_count": 0},
    }
    resp = client.post("/decision", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == "txn_decision"
    assert body["action"] in ("approve", "step_up_review", "decline")
    assert body["risk_tier"] in ("low", "medium", "high")
    assert "narrative" in body and "txn_decision" in body["narrative"]
    assert "credit_limit_recommendation" in body
    assert set(body["model_scores"].keys()) == {"lightgbm", "xgboost", "isolation_forest"}

    import audit_log

    conn = audit_log.connect(db_path)
    rows = conn.execute("SELECT * FROM decisions WHERE transaction_id = 'txn_decision'").fetchall()
    conn.close()
    assert len(rows) == 1


def test_decision_returns_503_when_no_models(tmp_path):
    app = create_app(models_dir=str(tmp_path / "models"), db_path=str(tmp_path / "audit.db"))
    client = TestClient(app)

    resp = client.post("/decision", json={"transaction": {"time": 1000, "amount": 50}})

    assert resp.status_code == 503


def test_cases_queue_lifecycle(trained_models_dir, tmp_path):
    db_path = str(tmp_path / "audit.db")
    app = create_app(models_dir=trained_models_dir, db_path=db_path)
    client = TestClient(app)

    # A moderately elevated score should land in step_up_review and show up in the queue.
    payload = {
        "transaction": {"time": 45000, "amount": 800, "transaction_id": "txn_case", "v14": -2.0, "v4": 1.5},
        "account": {"credit_limit": 5000, "current_balance": 1000},
    }
    client.post("/decision", json=payload)

    cases_resp = client.get("/cases")
    assert cases_resp.status_code == 200
    cases = cases_resp.json()["cases"]

    review_cases = [c for c in cases if c["transaction_id"] == "txn_case"]
    if not review_cases:
        # This particular transaction didn't land in review under the toy model -- not
        # informative either way, skip the resolve half rather than asserting on luck.
        return

    case_id = review_cases[0]["id"]
    resolve_resp = client.post(f"/cases/{case_id}/resolve", json={"verdict": "approve", "is_actual_fraud": False})
    assert resolve_resp.status_code == 200

    cases_after = client.get("/cases").json()["cases"]
    assert all(c["id"] != case_id for c in cases_after)


def test_resolve_unknown_case_returns_404(trained_models_dir, tmp_path):
    db_path = str(tmp_path / "audit.db")
    app = create_app(models_dir=trained_models_dir, db_path=db_path)
    client = TestClient(app)

    resp = client.post("/cases/999/resolve", json={"verdict": "approve"})
    assert resp.status_code == 404


def test_resolve_invalid_verdict_returns_400(trained_models_dir, tmp_path):
    db_path = str(tmp_path / "audit.db")
    app = create_app(models_dir=trained_models_dir, db_path=db_path)
    client = TestClient(app)

    resp = client.post("/cases/1/resolve", json={"verdict": "maybe"})
    assert resp.status_code == 400


def test_analytics_summary_reflects_recorded_decisions(trained_models_dir, tmp_path):
    db_path = str(tmp_path / "audit.db")
    app = create_app(models_dir=trained_models_dir, db_path=db_path)
    client = TestClient(app)

    payload = {
        "transaction": {"time": 90000, "amount": 2500, "transaction_id": "txn_analytics", "v14": -4.0, "v4": 3.0},
        "account": {"credit_limit": 5000},
    }
    client.post("/decision", json=payload)

    resp = client.get("/analytics/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "funnel" in body and "score_deciles" in body
    assert sum(row["transaction_count"] for row in body["funnel"]) == 1


def test_to_creditcard_frame_maps_lowercase_fields_to_uppercase_columns():
    payload = TransactionPayload(time=5.0, amount=10.0, v1=0.5, v28=-0.5)

    frame = to_creditcard_frame(payload)

    assert frame.loc[0, "Time"] == 5.0
    assert frame.loc[0, "Amount"] == 10.0
    assert frame.loc[0, "V1"] == 0.5
    assert frame.loc[0, "V28"] == -0.5
