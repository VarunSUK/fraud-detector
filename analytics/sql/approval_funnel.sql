-- Approval funnel: how the current policy splits transaction volume and
-- dollar amount across approve / step_up_review / decline.
--
-- Sizes the operational cost of a policy change before proposing it -- e.g.
-- "the review queue is already 8% of volume; lowering the review threshold
-- another 0.05 would add roughly this many more manual reviews per day."
SELECT
    action,
    COUNT(*) AS transaction_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM decisions), 2) AS pct_of_volume,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(fraud_score), 4) AS avg_fraud_score
FROM decisions
GROUP BY action
ORDER BY
    CASE action WHEN 'approve' THEN 1 WHEN 'step_up_review' THEN 2 WHEN 'decline' THEN 3 ELSE 4 END;
