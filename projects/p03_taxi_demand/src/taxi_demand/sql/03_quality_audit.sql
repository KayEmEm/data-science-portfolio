-- Rows per quality label, with the share of all source rows. Usable trips carry the label 'kept'.
CREATE OR REPLACE TABLE quality_audit AS
SELECT
    coalesce(issue, 'kept') AS check_result,
    count(*) AS trips,
    count(*) / sum(count(*)) OVER () AS share
FROM trips
GROUP BY ALL
ORDER BY trips DESC;
