# Project 3: New York taxi demand

Describe how yellow taxi demand, fares and tips vary by time, zone and weather in New York City, and present the result in a dashboard for an operations manager.

**Status: in progress.** The pipeline, the SQL query library, the dashboard and the tests run on synthetic files. Results, screenshots and the findings report follow the first run on the real data.

## Dataset and license

* **Trips:** [NYC Taxi and Limousine Commission trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), yellow taxi records for 2024, one Parquet file per month, and the taxi zone lookup table. License and terms of use: to be confirmed and cited here before the results are published.
* **Weather:** [NOAA NCEI daily summaries (GHCN Daily)](https://www.ncei.noaa.gov/) for New York Central Park, station USW00094728: precipitation, snowfall, maximum and minimum temperature, metric units. License: to be confirmed and cited here.

The raw data is not committed to this repository. `make data` downloads it. Where the download hosts are not reachable, files downloaded by hand can be placed with `make ingest SOURCES=<folder>`, which identifies each file by its content.

## Approach

1. **Load and check** (`data.py`, `ingest.py`): download each monthly file, check that every required column is present before any query runs.
2. **Warehouse in SQL** (`sql/01_*.sql` to `sql/06_*.sql`, run by DuckDB): a typed trip table in which each trip carries the first quality check it fails (pickup outside the year, no charge or void, negative amount, non positive or excessive duration, zero or excessive distance, excessive fare, implausible speed, unknown zone); a quality audit; and four summary tables: hourly facts by pickup borough and payment group, pickup zone by month, and one row per day joined with the weather.
3. **Query library** (`sql/analysis/a01_*.sql` to `a10_*.sql`): monthly trend, demand by weekday and hour, speed by hour, demand index by weather class, zone concentration, card tip rate, airport trips, zone volume cohorts, payment mix, quality audit. `make report` runs all ten and writes `reports/query_results.md`, `reports/key_numbers.json` and the figures.
4. **Weather effect:** each day's trips are divided by the mean of the same weekday in the same month (the demand index), so that weekly and seasonal patterns do not mask the weather.
5. **Notebook** (`notebooks/analysis.ipynb`): generated from the query library by `make notebook`, so its cells always match the SQL files.
6. **Dashboard** (`dashboard/app.py`): Streamlit and Plotly in the layout of a BI report: filters for dates, pickup borough and payment group; six headline measures with the change against the preceding period of equal length; tabs for overview, time patterns, zones, weather and data quality. It reads only the small summary tables, so it starts in seconds.
7. **Scheduling** (`dags/taxi_refresh.py`): an Airflow DAG that runs daily, fetches any newly published month, rebuilds the warehouse and refreshes the report tables.

### Analysis choices

* Tip measures use credit card trips only, because the source does not record cash tips.
* The limits for a plausible trip (180 minutes, 100 miles, 500 US dollars, 0.5 to 80 miles per hour) are analysis choices, set in `config.py`, not TLC rules.

## Reproduce

From this folder, with Python 3.11 or later:

```bash
make install            # this project and the shared common package
make all                # data, build, report, notebook
make dashboard          # http://localhost:8501
```

`make smoke` runs the warehouse and the report on synthetic files under `artifacts/smoke/`. Its numbers come from random data and say nothing about New York taxis.

## Layout

```
src/taxi_demand/    config, data, ingest, warehouse, sql/, sql/analysis/, report, notebook, dashboard_data, screenshots, synthetic
dashboard/          Streamlit app
dags/               Airflow DAG
tests/              unit tests, warehouse tests on synthetic files, dashboard render test
reports/            query results, key numbers, figures, findings
```

## Cost

No paid service used: 0 CAD.
