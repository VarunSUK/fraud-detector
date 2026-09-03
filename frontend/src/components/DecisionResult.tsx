import type { DecisionResponse } from "../types";
import { ActionBadge, RiskBadge } from "./ActionBadge";
import { ShapChart } from "./ShapChart";

export function DecisionResult({ decision }: { decision: DecisionResponse }) {
  const { credit_limit_recommendation: limitRec } = decision;
  const limitChanged = limitRec.recommended !== limitRec.current;

  return (
    <div className="card">
      <div className="card-header">
        <h2>Decision: {decision.transaction_id}</h2>
        <div className="badge-row">
          <ActionBadge action={decision.action} />
          <RiskBadge tier={decision.risk_tier} />
        </div>
      </div>

      <p className="narrative">{decision.narrative}</p>

      <div className="stat-grid">
        <div className="stat">
          <span className="stat-label">Fraud score</span>
          <span className="stat-value">{decision.fraud_score.toFixed(3)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Credit limit</span>
          <span className="stat-value">
            ${limitRec.current.toLocaleString()}
            {limitChanged && (
              <>
                {" -> "}${limitRec.recommended.toLocaleString()}
                <span className={limitRec.adjustment_pct >= 0 ? "positive" : "negative"}>
                  {" "}({limitRec.adjustment_pct >= 0 ? "+" : ""}
                  {limitRec.adjustment_pct.toFixed(1)}%)
                </span>
              </>
            )}
          </span>
        </div>
        <div className="stat">
          <span className="stat-label">Latency</span>
          <span className="stat-value">{decision.processing_ms}ms</span>
        </div>
      </div>

      <div className="reason-codes">
        {decision.reason_codes.map((code) => (
          <span key={code} className="chip">
            {code}
          </span>
        ))}
      </div>

      <h3>Model breakdown</h3>
      <div className="model-scores">
        {Object.entries(decision.model_scores).map(([name, score]) => (
          <div className="model-score-row" key={name}>
            <span className="model-name">{name}</span>
            <div className="model-score-bar-track">
              <div className="model-score-bar" style={{ width: `${score * 100}%` }} />
            </div>
            <span className="model-score-value">{score.toFixed(3)}</span>
          </div>
        ))}
      </div>

      <h3>What drove this score (SHAP)</h3>
      <ShapChart contributions={decision.feature_contributions} />
    </div>
  );
}
