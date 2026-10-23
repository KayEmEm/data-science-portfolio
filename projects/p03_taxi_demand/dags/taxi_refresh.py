"""Airflow DAG: once a day, fetch any newly published TLC month, rebuild the warehouse and the report tables.

The TLC publishes trip files monthly with a delay of about two months, so most daily runs
find nothing new; the download step skips files already present and months not yet published.
Set TAXI_PROJECT_DIR to this project folder on the Airflow worker.
"""

import os
from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PROJECT_DIR = os.environ.get("TAXI_PROJECT_DIR", "/opt/portfolio/projects/p03_taxi_demand")
YEAR = os.environ.get("TAXI_YEAR", "2024")

with DAG(
    dag_id="taxi_refresh",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["portfolio", "p03"],
) as dag:
    fetch = BashOperator(
        task_id="fetch_new_months",
        bash_command=f"cd {PROJECT_DIR} && python -m taxi_demand.data --year {YEAR} --allow-missing",
    )
    build = BashOperator(
        task_id="build_warehouse",
        bash_command=f"cd {PROJECT_DIR} && python -m taxi_demand.warehouse --year {YEAR}",
    )
    report = BashOperator(
        task_id="refresh_report_tables",
        bash_command=f"cd {PROJECT_DIR} && python -m taxi_demand.report",
    )
    fetch >> build >> report
