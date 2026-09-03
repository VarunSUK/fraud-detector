from narrative import generate


def test_generate_includes_key_facts():
    text = generate(
        transaction_id="txn_123",
        action="decline",
        risk_tier="high",
        reason_codes=["HIGH_FRAUD_SCORE", "RECENT_DELINQUENCY"],
        feature_contributions=[
            {"feature": "V14", "value": -4.2, "importance": 2.1, "contribution": 2.1},
            {"feature": "Amount", "value": 5000, "importance": 1.5, "contribution": 1.5},
            {"feature": "V4", "value": 3.1, "importance": -0.3, "contribution": -0.3},
        ],
        fraud_score=0.91,
    )

    assert "txn_123" in text
    assert "0.91" in text
    assert "declined" in text
    assert "high" in text
    assert "HIGH_FRAUD_SCORE" in text
    assert "V14" in text


def test_generate_handles_no_contributions():
    text = generate(
        transaction_id="txn_1",
        action="approve",
        risk_tier="low",
        reason_codes=[],
        feature_contributions=[],
        fraud_score=0.05,
    )

    assert "approved" in text
    assert "no dominant feature" in text
    assert "none" in text


def test_generate_limits_to_top_n():
    contributions = [{"feature": f"f{i}", "value": 0, "importance": i, "contribution": float(i)} for i in range(10)]
    text = generate(
        transaction_id="txn_1",
        action="approve",
        risk_tier="low",
        reason_codes=[],
        feature_contributions=contributions,
        fraud_score=0.05,
        top_n=2,
    )

    # Highest-magnitude contributions (f9, f8) should be named; low ones should not.
    assert "f9" in text
    assert "f8" in text
    assert "f0" not in text
