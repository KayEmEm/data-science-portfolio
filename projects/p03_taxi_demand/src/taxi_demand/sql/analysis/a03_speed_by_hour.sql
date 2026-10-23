-- Mean trip speed (total distance over total time) by hour, weekdays against weekends.
SELECT
    CASE WHEN pickup_dow <= 5 THEN 'Weekday' ELSE 'Weekend' END AS day_type,
    pickup_hour AS hour,
    sum(distance_mi) / (sum(duration_min) / 60.0) AS mean_speed_mph,
    sum(trips) AS trips
FROM fact_hourly
GROUP BY ALL
ORDER BY day_type, hour;
