"""
SQLite-backed audit log of every scored decision.

Persisting decisions (not just returning them over HTTP) is what makes
analytics/sql/*.sql possible, and mirrors a real requirement in credit risk
and compliance work: every automated decision needs to be reconstructable
later -- what score it got, what policy fired, what action was taken, and
(once known) whether it was actually fraud.

A fresh connection is opened per call rather than held open across requests.
That's deliberately simple: SQLite write-concurrency from multiple threads is
easy to get wrong, and this table sees at most a few writes per request in a
demo-scale service -- not a throughput path worth optimizing.
"""

import json
import math
import os
import sqlite3
import time
from typing import Dict, List, Optional

DEFAULT_DB_PATH = os.environ.get("AUDIT_DB_PATH", "audit_log.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT,
    created_at REAL NOT NULL,
    amount REAL NOT NULL,
    fraud_score REAL NOT NULL,
    action TEXT NOT NULL,
    risk_tier TEXT NOT NULL,
    reason_codes TEXT NOT NULL,
    model_scores TEXT NOT NULL,
    credit_limit_current REAL,
    credit_limit_recommended REAL,
    analyst_verdict TEXT,
    resolved_at REAL,
    is_actual_fraud INTEGER
);

CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions(action);
CREATE INDEX IF NOT EXISTS idx_decisions_created_at ON decisions(created_at);
"""


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def record_decision(
    db_path: Optional[str],
    transaction_id: str,
    amount: float,
    fraud_score: float,
    action: str,
    risk_tier: str,
    reason_codes: List[str],
    model_scores: Dict[str, float],
    credit_limit_current: float,
    credit_limit_recommended: float,
    is_actual_fraud: Optional[bool] = None,
    created_at: Optional[float] = None,
) -> int:
    conn = connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO decisions (
                transaction_id, created_at, amount, fraud_score, action, risk_tier,
                reason_codes, model_scores, credit_limit_current, credit_limit_recommended,
                is_actual_fraud
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_id,
                created_at if created_at is not None else time.time(),
                amount,
                fraud_score,
                action,
                risk_tier,
                json.dumps(reason_codes),
                json.dumps(model_scores),
                credit_limit_current,
                credit_limit_recommended,
                None if is_actual_fraud is None else int(is_actual_fraud),
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_pending_cases(db_path: Optional[str] = None, limit: int = 50) -> List[Dict]:
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT * FROM decisions
            WHERE action = 'step_up_review' AND analyst_verdict IS NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def resolve_case(
    db_path: Optional[str],
    case_id: int,
    verdict: str,
    is_actual_fraud: Optional[bool] = None,
    resolved_at: Optional[float] = None,
) -> bool:
    conn = connect(db_path)
    try:
        cursor = conn.execute(
            """
            UPDATE decisions
            SET analyst_verdict = ?, resolved_at = ?, is_actual_fraud = COALESCE(?, is_actual_fraud)
            WHERE id = ?
            """,
            (
                verdict,
                resolved_at if resolved_at is not None else time.time(),
                None if is_actual_fraud is None else int(is_actual_fraud),
                case_id,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def funnel_summary(db_path: Optional[str] = None) -> List[Dict]:
    """Approval funnel: volume and dollar split across approve / step_up_review
    / decline. Same shape as analytics/sql/approval_funnel.sql, exposed live
    for the dashboard rather than requiring an offline SQL run."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                action,
                COUNT(*) AS transaction_count,
                ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM decisions), 2) AS pct_of_volume,
                ROUND(SUM(amount), 2) AS total_amount,
                ROUND(AVG(fraud_score), 4) AS avg_fraud_score
            FROM decisions
            GROUP BY action
            ORDER BY CASE action WHEN 'approve' THEN 1 WHEN 'step_up_review' THEN 2 WHEN 'decline' THEN 3 ELSE 4 END
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def score_decile_summary(db_path: Optional[str] = None) -> List[Dict]:
    """Fraud rate by predicted-score decile, for confirmed (labeled) outcomes
    only. Same shape as analytics/sql/loss_rate_by_score_decile.sql."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                CAST(MIN(fraud_score * 10, 9) AS INTEGER) AS score_decile,
                COUNT(*) AS transaction_count,
                SUM(COALESCE(is_actual_fraud, 0)) AS confirmed_fraud_count,
                ROUND(AVG(COALESCE(is_actual_fraud, 0)) * 100, 2) AS fraud_rate_pct
            FROM decisions
            WHERE is_actual_fraud IS NOT NULL
            GROUP BY score_decile
            ORDER BY score_decile
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _decile_counts(conn: sqlite3.Connection, start: float, end: float) -> Dict[int, int]:
    rows = conn.execute(
        """
        SELECT CAST(MIN(fraud_score * 10, 9) AS INTEGER) AS score_decile, COUNT(*) AS n
        FROM decisions
        WHERE created_at >= ? AND created_at < ?
        GROUP BY score_decile
        """,
        (start, end),
    ).fetchall()
    return {row["score_decile"]: row["n"] for row in rows}


def score_drift_summary(
    db_path: Optional[str] = None,
    recent_days: int = 7,
    baseline_days: int = 7,
    now: Optional[float] = None,
) -> Dict:
    """Population Stability Index (PSI) between a recent window and the
    baseline window immediately before it, bucketed by score decile. PSI is
    the standard metric credit risk teams use to monitor score drift: it
    doesn't need ground-truth fraud labels (unlike score_decile_summary),
    so it can run continuously against live scoring traffic, not just
    resolved cases.

    PSI < 0.1 : stable
    PSI < 0.25: moderate shift, worth investigating
    PSI >= 0.25: significant shift -- the scored population has changed
                 enough that the model's calibration may no longer hold.
    """
    now = now if now is not None else time.time()
    recent_start = now - recent_days * 86400
    baseline_start = recent_start - baseline_days * 86400

    conn = connect(db_path)
    try:
        recent_counts = _decile_counts(conn, recent_start, now)
        baseline_counts = _decile_counts(conn, baseline_start, recent_start)
    finally:
        conn.close()

    recent_total = sum(recent_counts.values())
    baseline_total = sum(baseline_counts.values())

    deciles = []
    psi = 0.0
    epsilon = 1e-4  # avoids log(0) / division by zero for empty buckets

    for decile in range(10):
        recent_pct = recent_counts.get(decile, 0) / recent_total if recent_total else 0.0
        baseline_pct = baseline_counts.get(decile, 0) / baseline_total if baseline_total else 0.0
        r = max(recent_pct, epsilon)
        b = max(baseline_pct, epsilon)
        contribution = (r - b) * math.log(r / b)
        psi += contribution
        deciles.append(
            {
                "score_decile": decile,
                "baseline_pct": round(baseline_pct * 100, 2),
                "recent_pct": round(recent_pct * 100, 2),
            }
        )

    if baseline_total == 0 or recent_total == 0:
        interpretation = "insufficient_data"
    elif psi < 0.1:
        interpretation = "stable"
    elif psi < 0.25:
        interpretation = "moderate_shift"
    else:
        interpretation = "significant_shift"

    return {
        "psi": round(psi, 4),
        "interpretation": interpretation,
        "baseline_count": baseline_total,
        "recent_count": recent_total,
        "deciles": deciles,
    }


def _row_to_dict(row: sqlite3.Row) -> Dict:
    d = dict(row)
    d["reason_codes"] = json.loads(d["reason_codes"])
    d["model_scores"] = json.loads(d["model_scores"])
    d["is_actual_fraud"] = None if d["is_actual_fraud"] is None else bool(d["is_actual_fraud"])
    return d
