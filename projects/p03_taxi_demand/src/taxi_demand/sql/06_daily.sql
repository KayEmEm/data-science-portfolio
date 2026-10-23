-- One row per day: demand, fares and Central Park weather.
-- demand_index is the day's trips divided by the mean for the same weekday in the same month,
-- so that weekly and seasonal patterns do not mask the weather effect.
CREATE OR REPLACE TABLE daily AS
WITH day_totals AS (
    SELECT
        pickup_date,
        any_value(pickup_month) AS pickup_month,
        any_value(pickup_dow) AS pickup_dow,
        sum(trips) AS trips,
        sum(total_usd) AS total_usd,
        sum(fare_usd) AS fare_usd,
        sum(distance_mi) AS distance_mi,
        sum(duration_min) AS duration_min,
        sum(fare_usd) FILTER (WHERE payment = 'Credit card') AS card_fare_usd,
        sum(tip_usd) FILTER (WHERE payment = 'Credit card') AS card_tip_usd
    FROM fact_hourly
    GROUP BY pickup_date
),
weather AS (
    SELECT
        CAST("DATE" AS DATE) AS weather_date,
        CAST(PRCP AS DOUBLE) AS precip_mm,
        CAST(SNOW AS DOUBLE) AS snow_mm,
        CAST(TMAX AS DOUBLE) AS tmax_c,
        CAST(TMIN AS DOUBLE) AS tmin_c
    FROM raw_weather
)
SELECT
    d.*,
    d.trips / avg(d.trips) OVER (PARTITION BY d.pickup_month, d.pickup_dow) AS demand_index,
    w.precip_mm,
    w.snow_mm,
    w.tmax_c,
    w.tmin_c,
    CASE
        WHEN w.precip_mm IS NULL THEN 'no record'
        WHEN w.snow_mm >= 10 THEN 'snow 10 mm or more'
        WHEN w.precip_mm < 1 THEN 'dry, under 1 mm'
        WHEN w.precip_mm < 10 THEN 'rain 1 to 10 mm'
        ELSE 'rain 10 mm or more'
    END AS weather_class
FROM day_totals AS d
LEFT JOIN weather AS w ON w.weather_date = d.pickup_date
ORDER BY d.pickup_date;
