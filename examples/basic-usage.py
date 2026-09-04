#!/usr/bin/env python3
"""
Basic usage example: score a transaction, get its SHAP explanation, run it
through the credit risk policy, and check the review queue -- against a
live go-inference API.

Prerequisites (see the README Quick Start for the full sequence):
    python scripts/seed_audit_log.py --models-dir python-ml/models --db audit_log.db --train
    (cd python-ml && MODELS_DIR=models AUDIT_DB_PATH=../audit_log.db uvicorn serve:app --app-dir src --port 8000 &)
    (cd go-inference && ML_SERVICE_URL=http://localhost:8000 go run cmd/server/main.go &)

Usage:
    pip install requests
    python examples/basic-usage.py [--base-url http://localhost:8080]
"""

import argparse
import json
import sys

import requests


def pretty(label: str, obj) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(obj, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    args = parser.parse_args()

    session = requests.Session()

    try:
        health = session.get(f"{args.base_url}/health", timeout=5).json()
    except requests.exceptions.ConnectionError:
        print(f"Could not reach {args.base_url} -- is go-inference running? See the docstring above.", file=sys.stderr)
        return 1
    pretty("Health", health)

    # A transaction with fraud-shifted PCA features (see
    # python-ml/src/synthetic_creditcard.py for why V14/V4 in particular).
    transaction = {
        "transaction_id": "example_txn_001",
        "time": 90000,
        "amount": 2500.00,
        "v14": -3.2,
        "v4": 2.1,
    }

    score = session.post(f"{args.base_url}/api/v1/score", json={"transaction": transaction}, timeout=10).json()
    pretty("Score", score)

    explanation = session.post(f"{args.base_url}/api/v1/explain", json={"transaction": transaction}, timeout=10).json()
    top_features = explanation.get("feature_contributions", [])[:5]
    pretty("Top 5 SHAP contributions", top_features)

    decision_request = {
        "transaction": transaction,
        "account": {
            "credit_limit": 5000,
            "current_balance": 1000,
            "account_age_days": 400,
            "delinquent_payments_count": 0,
            "avg_monthly_spend": 800,
        },
    }
    decision_resp = session.post(f"{args.base_url}/api/v1/decision", json=decision_request, timeout=10)
    if decision_resp.status_code == 503:
        print("\n(Skipping /decision and /cases -- ml-serving sidecar isn't reachable from go-inference.)")
        return 0

    decision = decision_resp.json()
    pretty("Decision", {k: v for k, v in decision.items() if k != "feature_contributions"})

    if decision["action"] == "step_up_review":
        cases = session.get(f"{args.base_url}/api/v1/cases", timeout=10).json()
        pretty("Review queue", cases)
        print(f"\n'{transaction['transaction_id']}' is now pending review -- resolve it with:")
        print(f"  POST {args.base_url}/api/v1/cases/<id>/resolve  {{\"verdict\": \"approve\", \"is_actual_fraud\": false}}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
