-- One row per source trip, typed, with derived time fields and one quality label.
-- issue is NULL for a usable trip, else the first failed check in the order below.
-- Parameters: $year, $max_duration_min, $max_distance_mi, $max_fare_usd, $min_speed_mph, $max_speed_mph.
CREATE OR REPLACE TABLE trips AS
WITH typed AS (
    SELECT
        CAST(VendorID AS INTEGER) AS vendor_id,
        CAST(tpep_pickup_datetime AS TIMESTAMP) AS pickup_at,
        CAST(tpep_dropoff_datetime AS TIMESTAMP) AS dropoff_at,
        CAST(passenger_count AS INTEGER) AS passengers,
        CAST(trip_distance AS DOUBLE) AS distance_mi,
        CAST(RatecodeID AS INTEGER) AS rate_code,
        CAST(PULocationID AS INTEGER) AS pu_zone_id,
        CAST(DOLocationID AS INTEGER) AS do_zone_id,
        CAST(payment_type AS INTEGER) AS payment_type,
        CAST(fare_amount AS DOUBLE) AS fare_usd,
        CAST(tip_amount AS DOUBLE) AS tip_usd,
        CAST(tolls_amount AS DOUBLE) AS tolls_usd,
        CAST(total_amount AS DOUBLE) AS total_usd
    FROM raw_trips
),
derived AS (
    SELECT
        *,
        date_diff('second', pickup_at, dropoff_at) / 60.0 AS duration_min
    FROM typed
)
SELECT
    *,
    CAST(pickup_at AS DATE) AS pickup_date,
    month(pickup_at) AS pickup_month,
    hour(pickup_at) AS pickup_hour,
    isodow(pickup_at) AS pickup_dow,
    CASE
        WHEN pickup_at IS NULL OR dropoff_at IS NULL OR year(pickup_at) <> $year THEN 'pickup outside year'
        WHEN payment_type IN (3, 4, 6) THEN 'no charge, dispute or void'
        WHEN fare_usd < 0 OR tip_usd < 0 OR total_usd < 0 THEN 'negative amount'
        WHEN duration_min <= 0 THEN 'non positive duration'
        WHEN duration_min > $max_duration_min THEN 'duration over limit'
        WHEN distance_mi IS NULL OR distance_mi <= 0 THEN 'zero distance'
        WHEN distance_mi > $max_distance_mi THEN 'distance over limit'
        WHEN fare_usd > $max_fare_usd THEN 'fare over limit'
        WHEN distance_mi / (duration_min / 60.0) < $min_speed_mph THEN 'speed under limit'
        WHEN distance_mi / (duration_min / 60.0) > $max_speed_mph THEN 'speed over limit'
        WHEN pu_zone_id NOT BETWEEN 1 AND 263 OR do_zone_id NOT BETWEEN 1 AND 263 THEN 'unknown zone'
    END AS issue
FROM derived;
