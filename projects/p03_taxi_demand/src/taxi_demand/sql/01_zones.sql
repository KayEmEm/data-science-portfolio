-- Taxi zone lookup. Zones 264 and 265 are the TLC codes for an unknown or out of city location.
CREATE OR REPLACE TABLE zones AS
SELECT
    CAST(LocationID AS INTEGER) AS zone_id,
    coalesce(nullif(trim(Borough), ''), 'Unknown') AS borough,
    coalesce(nullif(trim(Zone), ''), 'Unknown') AS zone,
    coalesce(nullif(trim(service_zone), ''), 'Unknown') AS service_zone
FROM raw_zones;
