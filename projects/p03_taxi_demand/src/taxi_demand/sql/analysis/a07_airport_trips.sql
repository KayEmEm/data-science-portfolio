-- Trips that start or end at an airport zone (1 Newark, 132 JFK, 138 LaGuardia) against all other trips.
SELECT
    CASE
        WHEN pu_zone_id IN (1, 132, 138) OR do_zone_id IN (1, 132, 138) THEN 'Airport'
        ELSE 'Other'
    END AS trip_type,
    count(*) AS trips,
    count(*) / sum(count(*)) OVER () AS share_of_trips,
    sum(total_usd) / sum(sum(total_usd)) OVER () AS share_of_revenue,
    avg(total_usd) AS mean_total_usd,
    avg(distance_mi) AS mean_distance_mi,
    avg(duration_min) AS mean_duration_min
FROM trips
WHERE issue IS NULL
GROUP BY ALL
ORDER BY trip_type;
