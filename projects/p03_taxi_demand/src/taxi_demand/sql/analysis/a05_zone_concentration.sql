-- Pickup zones ranked by trips, with their share and the cumulative share of all trips.
WITH zone_totals AS (
    SELECT zone_id, zone, borough, sum(trips) AS trips, sum(total_usd) / sum(trips) AS mean_total_usd
    FROM zone_monthly
    GROUP BY ALL
)
SELECT
    row_number() OVER (ORDER BY trips DESC, zone_id) AS zone_rank,
    zone,
    borough,
    trips,
    mean_total_usd,
    trips / sum(trips) OVER () AS share,
    sum(trips) OVER (ORDER BY trips DESC, zone_id) / sum(trips) OVER () AS cumulative_share
FROM zone_totals
ORDER BY zone_rank;
