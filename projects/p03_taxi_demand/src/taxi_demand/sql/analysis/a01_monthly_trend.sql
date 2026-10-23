-- Monthly demand and revenue from usable trips.
SELECT
    pickup_month AS month,
    sum(trips) AS trips,
    sum(trips) / count(DISTINCT pickup_date) AS trips_per_day,
    sum(total_usd) AS total_usd,
    sum(fare_usd) / sum(trips) AS mean_fare_usd,
    sum(tip_usd) FILTER (WHERE payment = 'Credit card')
        / sum(fare_usd) FILTER (WHERE payment = 'Credit card') AS card_tip_rate
FROM fact_hourly
GROUP BY ALL
ORDER BY month;
