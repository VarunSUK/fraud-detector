import { describe, expect, it } from "vitest";
import { randomAccount, randomTransaction } from "./sampleData";

describe("randomTransaction", () => {
  it("produces 28 PCA features plus time/amount/transaction_id", () => {
    const txn = randomTransaction("normal");
    for (let i = 1; i <= 28; i++) {
      expect(typeof txn[`v${i}`]).toBe("number");
    }
    expect(typeof txn.time).toBe("number");
    expect(typeof txn.amount).toBe("number");
    expect(txn.transaction_id).toMatch(/^txn_/);
  });

  it("keeps normal-sample amounts within the low range", () => {
    for (let i = 0; i < 20; i++) {
      const txn = randomTransaction("normal");
      expect(txn.amount).toBeGreaterThanOrEqual(1);
      expect(txn.amount).toBeLessThanOrEqual(301);
    }
  });

  it("keeps fraud-like sample amounts within the elevated range", () => {
    for (let i = 0; i < 20; i++) {
      const txn = randomTransaction("fraud_like");
      expect(txn.amount).toBeGreaterThanOrEqual(100);
      expect(txn.amount).toBeLessThanOrEqual(4100);
    }
  });

  it("shifts V14 negative and V4 positive on average for fraud-like samples", () => {
    const n = 200;
    let v14Sum = 0;
    let v4Sum = 0;
    for (let i = 0; i < n; i++) {
      const txn = randomTransaction("fraud_like");
      v14Sum += Number(txn.v14);
      v4Sum += Number(txn.v4);
    }
    expect(v14Sum / n).toBeLessThan(-1);
    expect(v4Sum / n).toBeGreaterThan(1);
  });
});

describe("randomAccount", () => {
  it("keeps current_balance within credit_limit", () => {
    for (let i = 0; i < 20; i++) {
      const account = randomAccount();
      expect(account.current_balance).toBeGreaterThanOrEqual(0);
      expect(account.current_balance).toBeLessThanOrEqual(account.credit_limit);
    }
  });
});
