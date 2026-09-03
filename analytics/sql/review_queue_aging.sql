-- Review queue aging: how long pending step_up_review cases have been
-- sitting unresolved.
--
-- An operational health signal for the manual-review team -- a growing tail
-- of old, unresolved cases means either the queue is under-staffed or the
-- policy is routing too much low-value volume into review.
SELECT
    id,
    transaction_id,
    ROUND((strftime('%s', 'now') - created_at) / 3600.0, 1) AS hours_pending,
    amount,
    fraud_score,
    reason_codes
FROM decisions
WHERE action = 'step_up_review' AND analyst_verdict IS NULL
ORDER BY created_at ASC;
