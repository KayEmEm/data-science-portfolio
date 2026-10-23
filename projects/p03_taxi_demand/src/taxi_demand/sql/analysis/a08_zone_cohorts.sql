-- Pickup zones grouped by their January volume (quartile 1 is the busiest quarter of zones),
-- with each group's monthly trips per day indexed to January = 100.
WITH january AS (
    SELECT zone_id, trips
    FROM zone_monthly
    WHERE pickup_month = 1
),
cohort AS (
    SELECT zone_id, ntile(4) OVER (ORDER BY trips DESC) AS volume_quartile
    FROM january
),
days AS (
    SELECT pickup_month, count(DISTINCT pickup_date) AS n_days
    FROM fact_hourly
    GROUP BY ALL
),
monthly AS (
    SELECT c.volume_quartile, z.pickup_month, sum(z.trips) / any_value(d.n_days) AS trips_per_day
    FROM zone_monthly AS z
    JOIN cohort AS c USING (zone_id)
    JOIN days AS d USING (pickup_month)
    GROUP BY ALL
)
SELECT
    volume_quartile,
    pickup_month AS month,
    trips_per_day,
    100 * trips_per_day / first(trips_per_day ORDER BY pickup_month) OVER (PARTITION BY volume_quartile)
        AS index_january_100
FROM monthly
ORDER BY volume_quartile, month;
