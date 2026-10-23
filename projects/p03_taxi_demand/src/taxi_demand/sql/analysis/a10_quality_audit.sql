-- Source rows per quality check result.
SELECT check_result, trips, share
FROM quality_audit
ORDER BY trips DESC;
