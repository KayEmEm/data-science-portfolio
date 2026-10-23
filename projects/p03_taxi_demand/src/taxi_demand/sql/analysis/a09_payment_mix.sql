-- Share of trips by payment group per month.
SELECT
    pickup_month AS month,
    payment,
    sum(trips) AS trips,
    sum(trips) / sum(sum(trips)) OVER (PARTITION BY pickup_month) AS share
FROM fact_hourly
GROUP BY pickup_month, payment
ORDER BY month, payment;
