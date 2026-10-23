-- Credit card trips by pickup borough: tip as a share of fare, and the share of trips with no tip.
SELECT
    z.borough,
    count(*) AS card_trips,
    sum(t.tip_usd) / sum(t.fare_usd) AS card_tip_rate,
    avg(CAST(t.tip_usd = 0 AS INTEGER)) AS zero_tip_share,
    median(t.tip_usd / nullif(t.fare_usd, 0)) FILTER (WHERE t.tip_usd > 0) AS median_tip_rate_when_tipped
FROM trips AS t
JOIN zones AS z ON z.zone_id = t.pu_zone_id
WHERE t.issue IS NULL AND t.payment_type = 1
GROUP BY ALL
ORDER BY card_trips DESC;
