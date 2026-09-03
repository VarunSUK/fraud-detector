"""
Synthetic creditcard-schema data generator (V1..V28, Time, Amount, Class).

Produces data in the same shape as the real Kaggle creditcard.csv dataset
without needing that 150MB file present, for tests and for seeding the audit
log with realistic historical volume. Two features (V14, V4) get their means
shifted for fraud rows so trained models have a learnable signal, mirroring
how in the real dataset only a handful of PCA components carry most of the
fraud signal.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def generate(n: int = 1200, fraud_rate: float = 0.08, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate

    data = {f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    # Shifted, but with enough spread and overlap that the classes aren't
    # perfectly separable -- a model with AUC ~= 1.0 on toy data is a red flag,
    # not a feature, and gives every downstream report (score deciles, review
    # queue, threshold trade-off) nothing realistic to show.
    data["V14"] = np.where(is_fraud, rng.normal(-2.6, 1.6, n), rng.normal(0, 1, n))
    data["V4"] = np.where(is_fraud, rng.normal(2.0, 1.6, n), rng.normal(0, 1, n))
    data["Time"] = np.sort(rng.uniform(0, 172800, n))
    data["Amount"] = np.where(is_fraud, rng.gamma(2.0, 700, n) + 100, rng.gamma(2.0, 90, n) + 1)

    X = pd.DataFrame(data)
    y = pd.Series(is_fraud.astype(int), name="Class")
    return X, y
