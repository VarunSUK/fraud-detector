"""
Verifies each analytics/sql/*.sql report actually runs against the audit_log
schema without error, and returns the columns it claims to. This doesn't
assert on business logic (that's exercised by seeding real data and reading
the report, which is what these queries are for) -- it's a guard against SQL
syntax errors and schema drift (e.g. a column renamed in audit_log.py but not
in the .sql files) going unnoticed.

Run from the repo root: python -m pytest analytics/test_run_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python-ml" / "src"))

import pytest  # noqa: E402

import audit_log  # noqa: E402

SQL_DIR = Path(__file__).parent / "sql"
SQL_FILES = sorted(SQL_DIR.glob("*.sql"))


@pytest.fixture
def seeded_db(tmp_path):
    db_path = str(tmp_path / "audit.db")

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
        is_actual_fraud=False,
    )
    audit_log.record_decision(
        db_path,
        transaction_id="txn_declined",
        amount=4000,
        fraud_score=0.9,
        action="decline",
        risk_tier="high",
        reason_codes=["HIGH_FRAUD_SCORE"],
        model_scores={"lightgbm": 0.9, "xgboost": 0.88, "isolation_forest": 0.7},
        credit_limit_current=5000,
        credit_limit_recommended=3500,
        is_actual_fraud=True,
    )
    pending_id = audit_log.record_decision(
        db_path,
        transaction_id="txn_pending_review",
        amount=800,
        fraud_score=0.6,
        action="step_up_review",
        risk_tier="medium",
        reason_codes=["ELEVATED_FRAUD_SCORE"],
        model_scores={"lightgbm": 0.6, "xgboost": 0.55, "isolation_forest": 0.4},
        credit_limit_current=5000,
        credit_limit_recommended=5000,
    )
    assert pending_id > 0

    return db_path


@pytest.mark.parametrize("sql_file", SQL_FILES, ids=lambda f: f.stem)
def test_report_runs_without_error(seeded_db, sql_file):
    conn = audit_log.connect(seeded_db)
    try:
        cursor = conn.execute(sql_file.read_text())
        cursor.fetchall()
        assert cursor.description, f"{sql_file.name} returned no column metadata"
    finally:
        conn.close()


def test_review_queue_aging_only_returns_pending_cases(seeded_db):
    conn = audit_log.connect(seeded_db)
    try:
        rows = conn.execute((SQL_DIR / "review_queue_aging.sql").read_text()).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0]["transaction_id"] == "txn_pending_review"


def test_approval_funnel_covers_all_three_actions(seeded_db):
    conn = audit_log.connect(seeded_db)
    try:
        rows = conn.execute((SQL_DIR / "approval_funnel.sql").read_text()).fetchall()
    finally:
        conn.close()

    actions = {row["action"] for row in rows}
    assert actions == {"approve", "step_up_review", "decline"}
