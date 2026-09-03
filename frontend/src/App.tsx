import { useEffect, useState } from "react";
import "./App.css";
import { getDecision, getHealth } from "./api";
import { AnalyticsDashboard } from "./components/AnalyticsDashboard";
import { DecisionResult } from "./components/DecisionResult";
import { ReviewQueue } from "./components/ReviewQueue";
import { TransactionForm } from "./components/TransactionForm";
import type { AccountContext, DecisionResponse, Transaction } from "./types";

type Tab = "console" | "queue" | "analytics";

function App() {
  const [tab, setTab] = useState<Tab>("console");
  const [decision, setDecision] = useState<DecisionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<"healthy" | "unhealthy" | "unknown">("unknown");

  useEffect(() => {
    getHealth()
      .then((h) => setHealth(h.status === "healthy" ? "healthy" : "unhealthy"))
      .catch(() => setHealth("unhealthy"));
  }, []);

  const handleScore = async (transaction: Transaction, account: AccountContext) => {
    setLoading(true);
    setError(null);
    setDecision(null);
    try {
      const result = await getDecision(transaction, account);
      setDecision(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to score transaction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Fraud &amp; Credit Risk Console</h1>
        <span className={`health-dot health-${health}`} title={`Inference API: ${health}`} />
      </header>

      <nav className="tabs">
        <button className={tab === "console" ? "tab active" : "tab"} onClick={() => setTab("console")}>
          Decision Console
        </button>
        <button className={tab === "queue" ? "tab active" : "tab"} onClick={() => setTab("queue")}>
          Review Queue
        </button>
        <button className={tab === "analytics" ? "tab active" : "tab"} onClick={() => setTab("analytics")}>
          Analytics
        </button>
      </nav>

      <main>
        {tab === "console" && (
          <div className="console-layout">
            <TransactionForm onSubmit={handleScore} loading={loading} />
            {error && <div className="card error-card"><p className="error">{error}</p></div>}
            {decision && <DecisionResult decision={decision} />}
          </div>
        )}
        {tab === "queue" && <ReviewQueue />}
        {tab === "analytics" && <AnalyticsDashboard />}
      </main>
    </div>
  );
}

export default App;
