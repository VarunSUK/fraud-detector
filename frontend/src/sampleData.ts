import type { AccountContext, Transaction } from "./types";

function gaussian(mean: number, stdDev: number): number {
  // Box-Muller transform
  const u1 = Math.random() || 1e-9;
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * stdDev;
}

/** Builds a transaction resembling the synthetic creditcard-shaped training
 * data (see python-ml/src/synthetic_creditcard.py): mostly noise features,
 * with V14/V4/Amount shifted for the "fraud-like" case. */
export function randomTransaction(kind: "normal" | "fraud_like"): Transaction {
  const isFraud = kind === "fraud_like";
  const txn: Transaction = {
    time: Math.round(Math.random() * 172800),
    amount: isFraud
      ? Math.round((100 + Math.random() * 4000) * 100) / 100
      : Math.round((1 + Math.random() * 300) * 100) / 100,
    transaction_id: `txn_${Math.random().toString(36).slice(2, 10)}`,
  };

  for (let i = 1; i <= 28; i++) {
    let value: number;
    if (i === 14) {
      value = isFraud ? gaussian(-2.6, 1.6) : gaussian(0, 1);
    } else if (i === 4) {
      value = isFraud ? gaussian(2.0, 1.6) : gaussian(0, 1);
    } else {
      value = gaussian(0, 1);
    }
    txn[`v${i}`] = Math.round(value * 1000) / 1000;
  }

  return txn;
}

export function randomAccount(): AccountContext {
  const limit = [1000, 2500, 5000, 10000, 20000][Math.floor(Math.random() * 5)];
  return {
    credit_limit: limit,
    current_balance: Math.round(Math.random() * limit),
    account_age_days: Math.round(Math.random() * 2000),
    delinquent_payments_count: Math.random() < 0.15 ? Math.ceil(Math.random() * 2) : 0,
    avg_monthly_spend: Math.round(Math.random() * (limit / 2)),
  };
}

export function emptyAccount(): AccountContext {
  return {
    credit_limit: 5000,
    current_balance: 1000,
    account_age_days: 365,
    delinquent_payments_count: 0,
    avg_monthly_spend: 800,
  };
}
