-- Demand index by weather class. An index of 1.0 equals the mean of the same weekday in the same month.
SELECT
    weather_class,
    count(*) AS days,
    avg(demand_index) AS mean_demand_index,
    stddev_samp(demand_index) AS sd_demand_index,
    avg(fare_usd / trips) AS mean_fare_usd,
    avg(duration_min / trips) AS mean_duration_min
FROM daily
GROUP BY ALL
ORDER BY mean_demand_index DESC;
