from credit_decisioning import AccountContext, decide


def test_low_score_approves_by_default():
    account = AccountContext(credit_limit=5000, current_balance=1000)
    decision = decide(fraud_score=0.1, amount=50, account=account)

    assert decision.action == "approve"
    assert decision.risk_tier == "low"
    assert decision.reason_codes == ["WITHIN_POLICY"]


def test_high_score_declines():
    account = AccountContext(credit_limit=5000, current_balance=1000)
    decision = decide(fraud_score=0.95, amount=50, account=account)

    assert decision.action == "decline"
    assert decision.risk_tier == "high"
    assert "HIGH_FRAUD_SCORE" in decision.reason_codes


def test_mid_score_routes_to_review():
    account = AccountContext(credit_limit=5000, current_balance=1000)
    decision = decide(fraud_score=0.6, amount=50, account=account)

    assert decision.action == "step_up_review"
    assert decision.risk_tier == "medium"


def test_large_transaction_lowers_review_bar():
    account = AccountContext(credit_limit=5000, current_balance=1000)
    # 0.35 alone would approve, but a large amount should push it to review.
    small = decide(fraud_score=0.35, amount=100, account=account)
    large = decide(fraud_score=0.35, amount=5000, account=account)

    assert small.action == "approve"
    assert large.action == "step_up_review"
    assert "LARGE_TRANSACTION_AMOUNT" in large.reason_codes


def test_delinquency_tightens_thresholds_and_credit_limit():
    clean_account = AccountContext(credit_limit=5000, current_balance=1000)
    delinquent_account = AccountContext(credit_limit=5000, current_balance=1000, delinquent_payments_count=2)

    clean_decision = decide(fraud_score=0.75, amount=50, account=clean_account)
    delinquent_decision = decide(fraud_score=0.75, amount=50, account=delinquent_account)

    assert clean_decision.action == "step_up_review"
    assert delinquent_decision.action == "decline"
    assert "RECENT_DELINQUENCY" in delinquent_decision.reason_codes
    assert delinquent_decision.credit_limit_recommended < delinquent_account.credit_limit


def test_anomaly_score_adds_reason_code_without_changing_action():
    account = AccountContext(credit_limit=5000, current_balance=1000)
    decision = decide(fraud_score=0.1, amount=50, account=account, anomaly_score=0.9)

    assert decision.action == "approve"
    assert "ANOMALY_DETECTED" in decision.reason_codes


def test_high_utilization_with_low_risk_grows_credit_limit():
    account = AccountContext(credit_limit=1000, current_balance=900)  # 90% utilization
    decision = decide(fraud_score=0.05, amount=50, account=account)

    assert decision.action == "approve"
    assert "HIGH_UTILIZATION" in decision.reason_codes
    assert decision.credit_limit_recommended > decision.credit_limit_current


def test_zero_credit_limit_does_not_divide_by_zero():
    account = AccountContext(credit_limit=0, current_balance=0)
    decision = decide(fraud_score=0.9, amount=50, account=account)

    assert decision.credit_limit_adjustment_pct == 0.0
    assert decision.credit_limit_recommended == 0.0
