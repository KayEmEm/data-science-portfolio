-- Tip as a share of fare on credit card trips by pickup borough and hour.
-- Cash tips are not recorded in the source, so cash trips are left out.
SELECT
    pickup_borough AS borough,
    pickup_hour AS hour,
    sum(trips) AS card_trips,
    sum(tip_usd) / sum(fare_usd) AS card_tip_rate
FROM fact_hourly
WHERE payment = 'Credit card'
GROUP BY ALL
ORDER BY borough, hour;
