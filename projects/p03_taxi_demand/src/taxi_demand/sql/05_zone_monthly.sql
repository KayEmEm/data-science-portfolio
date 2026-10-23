-- Usable trips per pickup zone and month.
CREATE OR REPLACE TABLE zone_monthly AS
SELECT
    t.pickup_month,
    t.pu_zone_id AS zone_id,
    z.zone,
    z.borough,
    count(*) AS trips,
    sum(t.fare_usd) AS fare_usd,
    sum(t.total_usd) AS total_usd,
    sum(t.distance_mi) AS distance_mi,
    sum(t.fare_usd) FILTER (WHERE t.payment_type = 1) AS card_fare_usd,
    sum(t.tip_usd) FILTER (WHERE t.payment_type = 1) AS card_tip_usd
FROM trips AS t
JOIN zones AS z ON z.zone_id = t.pu_zone_id
WHERE t.issue IS NULL
GROUP BY ALL;
