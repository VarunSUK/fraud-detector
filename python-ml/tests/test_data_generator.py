from data_generator import TransactionGenerator, FraudPattern


def test_generate_transaction_stream_shape():
    generator = TransactionGenerator(seed=42)
    df = generator.generate_transaction_stream(num_users=10, transactions_per_user=5)

    assert len(df) == 50
    for col in ("transaction_id", "amount", "merchant", "is_fraud", "hour", "day_of_week"):
        assert col in df.columns


def test_generate_transaction_stream_is_deterministic_with_seed():
    df1 = TransactionGenerator(seed=7).generate_transaction_stream(num_users=5, transactions_per_user=5)
    df2 = TransactionGenerator(seed=7).generate_transaction_stream(num_users=5, transactions_per_user=5)

    assert df1["amount"].tolist() == df2["amount"].tolist()
    assert df1["is_fraud"].tolist() == df2["is_fraud"].tolist()


def test_derived_features_present_and_finite():
    generator = TransactionGenerator(seed=1)
    df = generator.generate_transaction_stream(num_users=10, transactions_per_user=5)

    for col in ("amount_log", "hour_sin", "hour_cos", "amount_to_avg_ratio"):
        assert col in df.columns
        assert df[col].notna().all()


def test_fraud_pattern_amount_anomaly():
    class FakeTxn:
        amount = 5000
        avg_amount = 100

    assert FraudPattern.amount_anomaly(FakeTxn()) is True


def test_fraud_pattern_time_anomaly():
    class FakeTxn:
        hour = 3

    assert FraudPattern.time_anomaly(FakeTxn()) is True

    class FakeTxnDaytime:
        hour = 14

    assert FraudPattern.time_anomaly(FakeTxnDaytime()) is False
