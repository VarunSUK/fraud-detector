#!/usr/bin/env python3
"""
Runs the analytics/sql/*.sql reports against the audit log database and
prints each as a plain-text table.

The queries in analytics/sql/ ARE the analysis; this script just executes
and formats them. Run it after scripts/seed_audit_log.py has populated a
database, or after the service has been live long enough to accumulate
real decisions.

Usage:
    python analytics/run_report.py --db ../audit_log.db
    python analytics/run_report.py --db ../audit_log.db --report approval_funnel
"""

import argparse
import sqlite3
import sys
from pathlib import Path

SQL_DIR = Path(__file__).parent / "sql"


def run_report(conn: sqlite3.Connection, sql_file: Path) -> None:
    print(f"\n=== {sql_file.stem} ===")
    cursor = conn.execute(sql_file.read_text())
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()

    if not rows:
        print("(no rows -- likely no labeled outcomes yet; run scripts/seed_audit_log.py)")
        return

    widths = [
        max(len(str(col)), max((len(str(row[i])) for row in rows), default=0))
        for i, col in enumerate(columns)
    ]
    print(" | ".join(col.ljust(w) for col, w in zip(columns, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description="Run credit risk / fraud analytics reports")
    parser.add_argument("--db", default="audit_log.db", help="Path to the audit log SQLite database")
    parser.add_argument("--report", help="Run only this report (filename stem, e.g. approval_funnel)")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Database not found at {args.db}. Run scripts/seed_audit_log.py first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    sql_files = sorted(SQL_DIR.glob("*.sql"))
    if args.report:
        sql_files = [f for f in sql_files if f.stem == args.report]
        if not sql_files:
            print(f"No report named {args.report!r} in {SQL_DIR}", file=sys.stderr)
            sys.exit(1)

    for sql_file in sql_files:
        run_report(conn, sql_file)
    conn.close()


if __name__ == "__main__":
    main()
