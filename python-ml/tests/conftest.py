import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

    rng = np.random.default_rng(42)
    n = 1200
    is_fraud = rng.random(n) < 0.08

    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    # Shift a couple of features for fraud rows so the models have signal to learn.
    data["V14"] = np.where(is_fraud, rng.normal(-4, 1, n), rng.normal(0, 1, n))
    data["V4"] = np.where(is_fraud, rng.normal(3, 1, n), rng.normal(0, 1, n))
    data["Time"] = np.sort(rng.uniform(0, 172800, n))
    data["Amount"] = np.where(is_fraud, rng.uniform(500, 5000, n), rng.uniform(1, 300, n))

    X = pd.DataFrame(data)
    y = pd.Series(is_fraud.astype(int), name="Class")

    models_dir = tmp_path_factory.mktemp("models")
    train_ensemble_models(X, y, models_dir=str(models_dir), dataset_type="creditcard")
    return str(models_dir)
