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
