import type { FeatureContribution } from "../types";

/** Diverging horizontal bar chart of SHAP-style feature contributions.
 * Positive contributions (push toward fraud) render right/red;
 * negative (push toward legitimate) render left/blue. */
export function ShapChart({
  contributions,
  topN = 8,
}: {
  contributions: FeatureContribution[];
  topN?: number;
}) {
  const top = contributions.slice(0, topN);
  if (top.length === 0) {
    return <p className="muted">No feature contributions available.</p>;
  }

  const maxAbs = Math.max(...top.map((c) => Math.abs(c.contribution)), 0.0001);

  return (
    <div className="shap-chart">
      {top.map((c) => {
        const pct = (Math.abs(c.contribution) / maxAbs) * 50;
        const isPositive = c.contribution >= 0;
        return (
          <div className="shap-row" key={c.feature}>
            <span className="shap-label" title={c.feature}>
              {c.feature}
            </span>
            <div className="shap-track">
              <div className="shap-half shap-half-neg">
                {!isPositive && (
                  <div className="shap-bar shap-bar-neg" style={{ width: `${pct}%` }} />
                )}
              </div>
              <div className="shap-center" />
              <div className="shap-half shap-half-pos">
                {isPositive && (
                  <div className="shap-bar shap-bar-pos" style={{ width: `${pct}%` }} />
                )}
              </div>
            </div>
            <span className="shap-value">{c.contribution >= 0 ? "+" : ""}{c.contribution.toFixed(2)}</span>
          </div>
        );
      })}
      <p className="shap-legend">
        <span className="legend-dot legend-dot-pos" /> pushes toward fraud &nbsp;&nbsp;
        <span className="legend-dot legend-dot-neg" /> pushes toward legitimate
      </p>
    </div>
  );
}
