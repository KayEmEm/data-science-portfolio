-- Mean trips per hour for each weekday (1 Monday to 7 Sunday) and hour of pickup.
WITH days AS (
    SELECT pickup_dow, count(DISTINCT pickup_date) AS n_days
    FROM fact_hourly
    GROUP BY ALL
)
SELECT
    f.pickup_dow AS weekday,
    f.pickup_hour AS hour,
    sum(f.trips) / any_value(d.n_days) AS mean_trips
FROM fact_hourly AS f
JOIN days AS d USING (pickup_dow)
GROUP BY ALL
ORDER BY weekday, hour;
