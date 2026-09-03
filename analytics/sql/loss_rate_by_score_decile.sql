-- Loss rate by predicted-score decile.
--
-- Sanity-checks model calibration against realized outcomes: fraud rate
-- should increase roughly monotonically with score decile. A decile out of
-- order, or a low-score decile with a meaningfully nonzero fraud rate, is a
-- signal the model (or the ensemble weighting) needs revisiting before a
-- policy leans on it harder.
SELECT
    CAST(MIN(fraud_score * 10, 9) AS INTEGER) AS score_decile,
    COUNT(*) AS transaction_count,
    SUM(COALESCE(is_actual_fraud, 0)) AS confirmed_fraud_count,
    ROUND(AVG(COALESCE(is_actual_fraud, 0)) * 100, 2) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(SUM(CASE WHEN is_actual_fraud = 1 THEN amount ELSE 0 END), 2) AS confirmed_fraud_amount
FROM decisions
WHERE is_actual_fraud IS NOT NULL
GROUP BY score_decile
ORDER BY score_decile;
