-- Usable trips aggregated to date, hour, pickup borough and payment group.
-- The dashboard filters and sums this table; sums (not means) keep every ratio exact after filtering.
CREATE OR REPLACE TABLE fact_hourly AS
SELECT
    t.pickup_date,
    t.pickup_month,
    t.pickup_hour,
    t.pickup_dow,
    z.borough AS pickup_borough,
    CASE t.payment_type WHEN 1 THEN 'Credit card' WHEN 2 THEN 'Cash' ELSE 'Other' END AS payment,
    count(*) AS trips,
    count(*) FILTER (WHERE t.pu_zone_id IN (1, 132, 138) OR t.do_zone_id IN (1, 132, 138)) AS airport_trips,
    sum(t.fare_usd) AS fare_usd,
    sum(t.tip_usd) AS tip_usd,
    sum(t.total_usd) AS total_usd,
    sum(t.distance_mi) AS distance_mi,
    sum(t.duration_min) AS duration_min
FROM trips AS t
JOIN zones AS z ON z.zone_id = t.pu_zone_id
WHERE t.issue IS NULL
GROUP BY ALL;
