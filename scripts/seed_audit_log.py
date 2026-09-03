#!/usr/bin/env python3
"""
Seeds the audit log with realistic historical decisions so the SQL reports
in analytics/sql/ and the review-queue dashboard have real data without
needing weeks of live traffic first.

Trains a fresh tiny ensemble on synthetic creditcard-shaped data (or loads
one already trained under --models-dir), scores a batch of held-out
transactions with known ground truth, applies the credit risk policy, and
writes each decision to the audit log. Older transactions are backdated and
marked as already resolved by an analyst (ground truth now known); the most
recent step_up_review cases are left pending, simulating an active queue.

Usage:
    python scripts/seed_audit_log.py --db audit_log.db --num-transactions 1500
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python-ml" / "src"))

import audit_log  # noqa: E402
from credit_decisioning import AccountContext, decide  # noqa: E402
from models import AnomalyDetector, FraudDetectionModel, train_ensemble_models  # noqa: E402
from synthetic_creditcard import generate  # noqa: E402

ENSEMBLE_WEIGHTS = {"lightgbm": 0.4, "xgboost": 0.4, "isolation_forest": 0.2}
CREDIT_LIMITS = [1000, 2500, 5000, 10000, 20000]
RECENT_WINDOW_SECONDS = 2 * 86400  # cases newer than this may still be pending review


def ensemble_score(lgb_model, xgb_model, anomaly_detector, frame):
    _, lgb_proba = lgb_model.predict(frame)
    _, xgb_proba = xgb_model.predict(frame)
    components = {
        "lightgbm": float(lgb_proba[0]),
        "xgboost": float(xgb_proba[0]),
        "isolation_forest": float(anomaly_detector.anomaly_score(frame)[0]),
    }
    weighted = sum(components[name] * ENSEMBLE_WEIGHTS[name] for name in components)
    return weighted / sum(ENSEMBLE_WEIGHTS.values()), components


def random_account() -> AccountContext:
    limit = random.choice(CREDIT_LIMITS)
    return AccountContext(
        credit_limit=limit,
        current_balance=random.uniform(0, limit),
        account_age_days=random.randint(1, 2000),
        delinquent_payments_count=random.choices([0, 1, 2], weights=[85, 10, 5])[0],
        avg_monthly_spend=random.uniform(200, limit / 2),
    )


def main():
    parser = argparse.ArgumentParser(description="Seed the audit log with historical decisions")
    parser.add_argument("--models-dir", default="python-ml/models")
    parser.add_argument("--db", default="audit_log.db")
    parser.add_argument("--num-transactions", type=int, default=800)
    parser.add_argument("--days-of-history", type=int, default=30)
    parser.add_argument("--train", action="store_true", help="Train a fresh ensemble even if models-dir has one")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    random.seed(args.seed)

    lightgbm_metadata = os.path.join(args.models_dir, "lightgbm_metadata.json")
    if args.train or not os.path.exists(lightgbm_metadata):
        print("Training a fresh ensemble for seeding...")
        train_X, train_y = generate(n=3000, seed=1)
        train_ensemble_models(train_X, train_y, models_dir=args.models_dir, dataset_type="creditcard")

    lgb_model = FraudDetectionModel(model_type="lightgbm")
    lgb_model.load_model(os.path.join(args.models_dir, "lightgbm"))
    xgb_model = FraudDetectionModel(model_type="xgboost")
    xgb_model.load_model(os.path.join(args.models_dir, "xgboost"))
    anomaly_detector = AnomalyDetector()
    anomaly_detector.load(os.path.join(args.models_dir, "isolation_forest"))

    X, y = generate(n=args.num_transactions, seed=args.seed)

    now = time.time()
    history_seconds = args.days_of_history * 86400
    pending_count = 0

    for i in range(len(X)):
        frame = X.iloc[[i]]
        score, components = ensemble_score(lgb_model, xgb_model, anomaly_detector, frame)

        amount = float(frame["Amount"].iloc[0])
        account = random_account()
        outcome = decide(score, amount, account, components["isolation_forest"])

        # Spread creation times over the trailing window, oldest first.
        created_at = now - history_seconds * (1 - i / len(X))
        is_actual_fraud = bool(y.iloc[i])
        is_recent = created_at > now - RECENT_WINDOW_SECONDS

        leave_pending = outcome.action == "step_up_review" and is_recent
        recorded_fraud_label = None if leave_pending else is_actual_fraud

        row_id = audit_log.record_decision(
            args.db,
            transaction_id=f"seed_{i:05d}",
            amount=amount,
            fraud_score=score,
            action=outcome.action,
            risk_tier=outcome.risk_tier,
            reason_codes=outcome.reason_codes,
            model_scores=components,
            credit_limit_current=outcome.credit_limit_current,
            credit_limit_recommended=outcome.credit_limit_recommended,
            is_actual_fraud=recorded_fraud_label,
            created_at=created_at,
        )

        if leave_pending:
            pending_count += 1
        elif outcome.action == "step_up_review":
            verdict = "decline" if is_actual_fraud else "approve"
            audit_log.resolve_case(
                args.db, row_id, verdict=verdict, is_actual_fraud=is_actual_fraud, resolved_at=created_at + 3600
            )

    print(f"Seeded {len(X)} historical decisions into {args.db} ({pending_count} left pending review)")


if __name__ == "__main__":
    main()
