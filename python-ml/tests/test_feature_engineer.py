import numpy as np

from data_generator import TransactionGenerator
from models import FeatureEngineer


def _synthetic_frame(num_users=10, transactions_per_user=5):
    generator = TransactionGenerator(seed=3)
    df = generator.generate_transaction_stream(num_users=num_users, transactions_per_user=transactions_per_user)
    feature_cols = [
        "amount", "merchant", "card_type", "hour", "day_of_week", "is_weekend",
        "previous_transactions", "avg_amount", "max_amount", "location_country", "device_type",
    ]
    return df[feature_cols].copy()


def test_prepare_features_synthetic_shape_and_no_nans():
    engineer = FeatureEngineer(dataset_type="synthetic")
    X = _synthetic_frame()

    features = engineer.prepare_features(X, fit=True)

    assert features.shape[0] == len(X)
    assert features.shape[1] == len(engineer.get_feature_names())
    assert not np.isnan(features).any()


def test_prepare_features_transform_reuses_fitted_encoders():
    engineer = FeatureEngineer(dataset_type="synthetic")
    X = _synthetic_frame()

    engineer.prepare_features(X, fit=True)
    transformed = engineer.prepare_features(X, fit=False)

    assert transformed.shape == (len(X), len(engineer.get_feature_names()))


def test_save_and_load_round_trip(tmp_path):
    engineer = FeatureEngineer(dataset_type="synthetic")
    X = _synthetic_frame()
    engineer.prepare_features(X, fit=True)

    filepath = tmp_path / "feature_engineer.joblib"
    engineer.save(str(filepath))

    reloaded = FeatureEngineer(dataset_type="synthetic")
    reloaded.load(str(filepath))

    assert reloaded.get_feature_names() == engineer.get_feature_names()
