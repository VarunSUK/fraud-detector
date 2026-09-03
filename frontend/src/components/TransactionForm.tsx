import { useState } from "react";
import type { AccountContext, Transaction } from "../types";
import { emptyAccount, randomAccount, randomTransaction } from "../sampleData";

interface Props {
  onSubmit: (transaction: Transaction, account: AccountContext) => void;
  loading: boolean;
}

export function TransactionForm({ onSubmit, loading }: Props) {
  const [transaction, setTransaction] = useState<Transaction>(() => randomTransaction("normal"));
  const [account, setAccount] = useState<AccountContext>(emptyAccount());
  const [showAdvanced, setShowAdvanced] = useState(false);

  const loadSample = (kind: "normal" | "fraud_like") => {
    setTransaction(randomTransaction(kind));
    setAccount(randomAccount());
  };

  const updateTxnField = (field: string, value: string) => {
    setTransaction((t) => ({ ...t, [field]: field === "transaction_id" ? value : Number(value) }));
  };

  const updateAccountField = (field: keyof AccountContext, value: string) => {
    setAccount((a) => ({ ...a, [field]: Number(value) }));
  };

  return (
    <form
      className="card"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(transaction, account);
      }}
    >
      <div className="card-header">
        <h2>Score a Transaction</h2>
        <div className="sample-buttons">
          <button type="button" className="btn-secondary" onClick={() => loadSample("normal")}>
            Load normal sample
          </button>
          <button type="button" className="btn-secondary" onClick={() => loadSample("fraud_like")}>
            Load fraud-like sample
          </button>
        </div>
      </div>

      <fieldset>
        <legend>Transaction</legend>
        <div className="form-grid">
          <label>
            Transaction ID
            <input
              type="text"
              value={transaction.transaction_id ?? ""}
              onChange={(e) => updateTxnField("transaction_id", e.target.value)}
            />
          </label>
          <label>
            Amount ($)
            <input
              type="number"
              step="0.01"
              value={transaction.amount}
              onChange={(e) => updateTxnField("amount", e.target.value)}
              required
            />
          </label>
          <label>
            Time (seconds since window start)
            <input
              type="number"
              value={transaction.time}
              onChange={(e) => updateTxnField("time", e.target.value)}
              required
            />
          </label>
        </div>
      </fieldset>

      <fieldset>
        <legend>Account context</legend>
        <div className="form-grid">
          <label>
            Credit limit ($)
            <input
              type="number"
              value={account.credit_limit}
              onChange={(e) => updateAccountField("credit_limit", e.target.value)}
            />
          </label>
          <label>
            Current balance ($)
            <input
              type="number"
              value={account.current_balance}
              onChange={(e) => updateAccountField("current_balance", e.target.value)}
            />
          </label>
          <label>
            Account age (days)
            <input
              type="number"
              value={account.account_age_days}
              onChange={(e) => updateAccountField("account_age_days", e.target.value)}
            />
          </label>
          <label>
            Delinquent payments
            <input
              type="number"
              value={account.delinquent_payments_count}
              onChange={(e) => updateAccountField("delinquent_payments_count", e.target.value)}
            />
          </label>
          <label>
            Avg monthly spend ($)
            <input
              type="number"
              value={account.avg_monthly_spend}
              onChange={(e) => updateAccountField("avg_monthly_spend", e.target.value)}
            />
          </label>
        </div>
      </fieldset>

      <button type="button" className="btn-link" onClick={() => setShowAdvanced((s) => !s)}>
        {showAdvanced ? "Hide" : "Show"} raw PCA features (V1-V28)
      </button>

      {showAdvanced && (
        <div className="pca-grid">
          {Array.from({ length: 28 }, (_, i) => i + 1).map((i) => (
            <label key={i} className="pca-field">
              V{i}
              <input
                type="number"
                step="0.01"
                value={Number(transaction[`v${i}`] ?? 0)}
                onChange={(e) => updateTxnField(`v${i}`, e.target.value)}
              />
            </label>
          ))}
        </div>
      )}

      <button type="submit" className="btn-primary" disabled={loading}>
        {loading ? "Scoring..." : "Score Transaction"}
      </button>
    </form>
  );
}
