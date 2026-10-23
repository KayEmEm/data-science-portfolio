-- Demand index by weather class, holidays left out. An index of 1.0 equals the mean of the
-- non holiday days with the same weekday in the same month. ci95 is the half width of a normal
-- approximation 95 percent interval for the mean index.
SELECT
    weather_class,
    count(*) AS days,
    avg(demand_index) AS mean_demand_index,
    stddev_samp(demand_index) AS sd_demand_index,
    1.96 * stddev_samp(demand_index) / sqrt(count(*)) AS ci95,
    avg(fare_usd / trips) AS mean_fare_usd,
    avg(duration_min / trips) AS mean_duration_min
FROM daily
WHERE NOT is_holiday
GROUP BY ALL
ORDER BY mean_demand_index DESC;
