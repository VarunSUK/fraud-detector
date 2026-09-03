import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def trained_models_dir(tmp_path_factory):
    """Trains a tiny lightgbm+xgboost ensemble on synthetic creditcard-shaped
    data (V1..V28, Time, Amount, Class) once and reuses it across tests. This
    matches the schema serve.py and go-inference's Transaction actually use,
    unlike the TransactionGenerator synthetic schema (merchant, card_type, ...).
    """
    from models import train_ensemble_models
    from synthetic_creditcard import generate

    X, y = generate(n=1200, fraud_rate=0.08, seed=42)

    models_dir = tmp_path_factory.mktemp("models")
    train_ensemble_models(X, y, models_dir=str(models_dir), dataset_type="creditcard")
    return str(models_dir)
