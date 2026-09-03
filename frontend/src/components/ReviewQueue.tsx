import { useEffect, useState, useCallback } from "react";
import { getCases, resolveCase } from "../api";
import type { Case } from "../types";

export function ReviewQueue() {
  const [cases, setCases] = useState<Case[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resolvingId, setResolvingId] = useState<number | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await getCases();
      setCases(resp.cases);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load review queue");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleResolve = async (caseItem: Case, verdict: "approve" | "decline") => {
    setResolvingId(caseItem.id);
    try {
      await resolveCase(caseItem.id, verdict, verdict === "decline");
      setCases((prev) => prev.filter((c) => c.id !== caseItem.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to resolve case");
    } finally {
      setResolvingId(null);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>Human Review Queue</h2>
        <button className="btn-secondary" onClick={load} disabled={loading}>
          Refresh
        </button>
      </div>

      {error && <p className="error">{error}</p>}
      {loading && <p className="muted">Loading...</p>}
      {!loading && cases.length === 0 && !error && (
        <p className="muted">No pending cases. Score a borderline transaction to populate the queue.</p>
      )}

      <div className="case-list">
        {cases.map((c) => (
          <div className="case-row" key={c.id}>
            <div className="case-info">
              <strong>{c.transaction_id}</strong>
              <span className="muted"> · ${c.amount.toFixed(2)} · score {c.fraud_score.toFixed(2)}</span>
              <div className="reason-codes">
                {c.reason_codes.map((code) => (
                  <span key={code} className="chip">
                    {code}
                  </span>
                ))}
              </div>
            </div>
            <div className="case-actions">
              <button
                className="btn-approve"
                disabled={resolvingId === c.id}
                onClick={() => handleResolve(c, "approve")}
              >
                Approve
              </button>
              <button
                className="btn-decline"
                disabled={resolvingId === c.id}
                onClick={() => handleResolve(c, "decline")}
              >
                Decline
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
