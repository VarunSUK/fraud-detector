import { useEffect, useState } from "react";
import { getAnalyticsSummary } from "../api";
import type { AnalyticsSummary } from "../types";
import { DriftBadge } from "./ActionBadge";

const ACTION_COLORS: Record<string, string> = {
  approve: "#2e7d46",
  step_up_review: "#b8860b",
  decline: "#c0392b",
};

export function AnalyticsDashboard() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setSummary(await getAnalyticsSummary());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load analytics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  if (loading) return <div className="card"><p className="muted">Loading analytics...</p></div>;
  if (error) return <div className="card"><p className="error">{error}</p></div>;
  if (!summary) return null;

  const maxFunnelCount = Math.max(...summary.funnel.map((r) => r.transaction_count), 1);

  return (
    <div className="card">
      <div className="card-header">
        <h2>Model &amp; Policy Analytics</h2>
        <button className="btn-secondary" onClick={load}>
          Refresh
        </button>
      </div>
      <p className="muted">
        Live view of analytics/sql/approval_funnel.sql and loss_rate_by_score_decile.sql. Run{" "}
        <code>python analytics/run_report.py</code> for the full ad hoc report set (threshold trade-off,
        review queue aging).
      </p>

      <h3>Approval funnel</h3>
      {summary.funnel.length === 0 ? (
        <p className="muted">No decisions recorded yet. Score some transactions first.</p>
      ) : (
        <div className="funnel-chart">
          {summary.funnel.map((row) => (
            <div className="funnel-row" key={row.action}>
              <span className="funnel-label">{row.action}</span>
              <div className="funnel-track">
                <div
                  className="funnel-bar"
                  style={{
                    width: `${(row.transaction_count / maxFunnelCount) * 100}%`,
                    background: ACTION_COLORS[row.action] ?? "#888",
                  }}
                />
              </div>
              <span className="funnel-value">
                {row.transaction_count} ({row.pct_of_volume}%) · avg score {row.avg_fraud_score}
              </span>
            </div>
          ))}
        </div>
      )}

      <h3>Fraud rate by score decile</h3>
      {summary.score_deciles.length === 0 ? (
        <p className="muted">
          No labeled outcomes yet. Run <code>scripts/seed_audit_log.py</code> for historical data, or
          resolve pending review cases with a fraud verdict.
        </p>
      ) : (
        <div className="decile-chart">
          {summary.score_deciles.map((row) => (
            <div className="decile-bar-container" key={row.score_decile}>
              <div
                className="decile-bar"
                style={{ height: `${Math.max(row.fraud_rate_pct, 2)}%` }}
                title={`${row.fraud_rate_pct}% fraud (${row.confirmed_fraud_count}/${row.transaction_count})`}
              />
              <span className="decile-label">{row.score_decile}</span>
            </div>
          ))}
        </div>
      )}
      <p className="muted small">
        Deciles bucket the fraud score into [0.0-0.1) ... [0.9-1.0]. Fraud rate should rise with decile;
        a decile out of order (or a low decile with meaningfully nonzero fraud) is a calibration
        red flag worth investigating before leaning on the score for policy.
      </p>

      <h3>
        Score drift <DriftBadge interpretation={summary.drift.interpretation} />
      </h3>
      {summary.drift.interpretation === "insufficient_data" ? (
        <p className="muted">
          Not enough history yet to compare a recent window against a baseline window. Run{" "}
          <code>scripts/seed_audit_log.py</code> with a longer <code>--days-of-history</code>, or wait for
          more live traffic.
        </p>
      ) : (
        <>
          <div className="drift-legend">
            <span>
              <span className="drift-legend-swatch" style={{ background: "#9aa5b5" }} />
              baseline ({summary.drift.baseline_count})
            </span>
            <span>
              <span className="drift-legend-swatch" style={{ background: "#4a63d6" }} />
              recent ({summary.drift.recent_count})
            </span>
            <span>PSI {summary.drift.psi}</span>
          </div>
          <div className="drift-chart">
            {summary.drift.deciles.map((row) => (
              <div className="drift-bar-container" key={row.score_decile}>
                <div className="drift-bar-pair">
                  <div className="drift-bar-baseline" style={{ height: `${Math.max(row.baseline_pct, 1)}%` }} />
                  <div className="drift-bar-recent" style={{ height: `${Math.max(row.recent_pct, 1)}%` }} />
                </div>
                <span className="decile-label">{row.score_decile}</span>
              </div>
            ))}
          </div>
        </>
      )}
      <p className="muted small">
        Population Stability Index between the last 7 days and the 7 days before that, bucketed by score
        decile. Below 0.1 is stable, 0.1 to 0.25 is a moderate shift worth investigating, above 0.25 is
        significant enough that the model's calibration may no longer hold against the incoming
        transaction population. This does not need fraud labels, unlike the decile chart above it, so
        it can run continuously against live traffic.
      </p>
    </div>
  );
}
