-- Threshold trade-off analysis.
--
-- For a set of candidate decline thresholds, shows how many transactions
-- would be declined, how much confirmed fraud would be caught vs. missed,
-- and how many good customers would be wrongly declined (false declines,
-- with their dollar exposure). This is the table a policy-change proposal
-- ("move the decline threshold from 0.85 to 0.75") should be built on
-- before it gets pushed, not just intuition about the score.
WITH candidate_thresholds(threshold) AS (
    VALUES (0.3), (0.4), (0.5), (0.6), (0.7), (0.8), (0.9)
),
labeled AS (
    SELECT * FROM decisions WHERE is_actual_fraud IS NOT NULL
)
SELECT
    ct.threshold,
    COUNT(*) FILTER (WHERE l.fraud_score >= ct.threshold) AS would_decline_count,
    COUNT(*) FILTER (WHERE l.fraud_score >= ct.threshold AND l.is_actual_fraud = 1) AS fraud_caught,
    COUNT(*) FILTER (WHERE l.fraud_score < ct.threshold AND l.is_actual_fraud = 1) AS fraud_missed,
    COUNT(*) FILTER (WHERE l.fraud_score >= ct.threshold AND l.is_actual_fraud = 0) AS false_declines,
    ROUND(SUM(CASE WHEN l.fraud_score >= ct.threshold AND l.is_actual_fraud = 0
                   THEN l.amount ELSE 0 END), 2) AS false_decline_amount
FROM candidate_thresholds ct
CROSS JOIN labeled l
GROUP BY ct.threshold
ORDER BY ct.threshold;
