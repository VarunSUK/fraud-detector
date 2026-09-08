import os

import audit_log


def test_record_and_list_pending_cases(tmp_path):
    db_path = str(tmp_path / "audit.db")

    audit_log.record_decision(
        db_path,
        transaction_id="txn_review",
        amount=500,
        fraud_score=0.6,
        action="step_up_review",
        risk_tier="medium",
        reason_codes=["ELEVATED_FRAUD_SCORE"],
        model_scores={"lightgbm": 0.6, "xgboost": 0.55, "isolation_forest": 0.4},
        credit_limit_current=5000,
        credit_limit_recommended=5000,
    )
    audit_log.record_decision(
        db_path,
        transaction_id="txn_approved",
        amount=50,
        fraud_score=0.1,
        action="approve",
        risk_tier="low",
        reason_codes=["WITHIN_POLICY"],
        model_scores={"lightgbm": 0.1, "xgboost": 0.1, "isolation_forest": 0.05},
        credit_limit_current=5000,
        credit_limit_recommended=5000,
    )

    pending = audit_log.list_pending_cases(db_path)

    assert len(pending) == 1
    assert pending[0]["transaction_id"] == "txn_review"
    assert pending[0]["reason_codes"] == ["ELEVATED_FRAUD_SCORE"]
    assert pending[0]["model_scores"]["lightgbm"] == 0.6
    assert pending[0]["analyst_verdict"] is None


def test_resolve_case_removes_it_from_pending(tmp_path):
    db_path = str(tmp_path / "audit.db")

    case_id = audit_log.record_decision(
        db_path,
        transaction_id="txn_review",
        amount=500,
        fraud_score=0.6,
        action="step_up_review",
        risk_tier="medium",
        reason_codes=["ELEVATED_FRAUD_SCORE"],
        model_scores={"lightgbm": 0.6},
        credit_limit_current=5000,
        credit_limit_recommended=5000,
    )

    resolved = audit_log.resolve_case(db_path, case_id, verdict="approve", is_actual_fraud=False)

    assert resolved is True
    assert audit_log.list_pending_cases(db_path) == []


def test_resolve_nonexistent_case_returns_false(tmp_path):
    db_path = str(tmp_path / "audit.db")
    audit_log.connect(db_path).close()  # ensure schema exists

    assert audit_log.resolve_case(db_path, 999, verdict="approve") is False


def test_funnel_and_decile_summaries(tmp_path):
    db_path = str(tmp_path / "audit.db")

    audit_log.record_decision(
        db_path, transaction_id="t1", amount=50, fraud_score=0.1, action="approve",
        risk_tier="low", reason_codes=["WITHIN_POLICY"], model_scores={"lightgbm": 0.1},
        credit_limit_current=5000, credit_limit_recommended=5000, is_actual_fraud=False,
    )
    audit_log.record_decision(
        db_path, transaction_id="t2", amount=4000, fraud_score=0.9, action="decline",
        risk_tier="high", reason_codes=["HIGH_FRAUD_SCORE"], model_scores={"lightgbm": 0.9},
        credit_limit_current=5000, credit_limit_recommended=3500, is_actual_fraud=True,
    )
    audit_log.record_decision(
        db_path, transaction_id="t3", amount=500, fraud_score=0.6, action="step_up_review",
        risk_tier="medium", reason_codes=["ELEVATED_FRAUD_SCORE"], model_scores={"lightgbm": 0.6},
        credit_limit_current=5000, credit_limit_recommended=5000,
    )

    funnel = audit_log.funnel_summary(db_path)
    assert {row["action"] for row in funnel} == {"approve", "decline", "step_up_review"}
    approve_row = next(row for row in funnel if row["action"] == "approve")
    assert approve_row["transaction_count"] == 1

    deciles = audit_log.score_decile_summary(db_path)
    # Only the two labeled (resolved) transactions should appear; the pending
    # step_up_review case has no is_actual_fraud yet.
    assert sum(row["transaction_count"] for row in deciles) == 2
    high_decile = next(row for row in deciles if row["score_decile"] == 9)
    assert high_decile["confirmed_fraud_count"] == 1


def _record_at(db_path, transaction_id, fraud_score, created_at):
    audit_log.record_decision(
        db_path,
        transaction_id=transaction_id,
        amount=100,
        fraud_score=fraud_score,
        action="approve" if fraud_score < 0.5 else "decline",
        risk_tier="low",
        reason_codes=["WITHIN_POLICY"],
        model_scores={"lightgbm": fraud_score},
        credit_limit_current=5000,
        credit_limit_recommended=5000,
        created_at=created_at,
    )


def test_score_drift_summary_stable_when_distributions_match(tmp_path):
    db_path = str(tmp_path / "audit.db")
    now = 1_700_000_000.0
    recent_start = now - 7 * 86400
    baseline_start = recent_start - 7 * 86400

    # Same low-score distribution in both windows.
    for i in range(20):
        _record_at(db_path, f"baseline_{i}", 0.1, baseline_start + i * 60)
        _record_at(db_path, f"recent_{i}", 0.1, recent_start + i * 60)

    drift = audit_log.score_drift_summary(db_path, now=now)

    assert drift["interpretation"] == "stable"
    assert drift["psi"] < 0.1
    assert drift["baseline_count"] == 20
    assert drift["recent_count"] == 20


def test_score_drift_summary_significant_shift_when_distributions_diverge(tmp_path):
    db_path = str(tmp_path / "audit.db")
    now = 1_700_000_000.0
    recent_start = now - 7 * 86400
    baseline_start = recent_start - 7 * 86400

    # Baseline is all low scores; recent is all high scores -- a real drift.
    for i in range(20):
        _record_at(db_path, f"baseline_{i}", 0.05, baseline_start + i * 60)
        _record_at(db_path, f"recent_{i}", 0.95, recent_start + i * 60)

    drift = audit_log.score_drift_summary(db_path, now=now)

    assert drift["interpretation"] == "significant_shift"
    assert drift["psi"] >= 0.25
    low_decile = next(d for d in drift["deciles"] if d["score_decile"] == 0)
    high_decile = next(d for d in drift["deciles"] if d["score_decile"] == 9)
    assert low_decile["baseline_pct"] == 100.0
    assert high_decile["recent_pct"] == 100.0


def test_score_drift_summary_insufficient_data_when_windows_empty(tmp_path):
    db_path = str(tmp_path / "audit.db")
    audit_log.connect(db_path).close()

    drift = audit_log.score_drift_summary(db_path)

    assert drift["interpretation"] == "insufficient_data"
    assert drift["baseline_count"] == 0
    assert drift["recent_count"] == 0
