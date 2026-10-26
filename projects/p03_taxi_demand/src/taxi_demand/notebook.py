"""Write the analysis notebook with nbformat and execute it against the warehouse.

The notebook is generated rather than edited by hand, so its cells always match the query library.
"""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from portfolio_common.config import parse_args

from taxi_demand import config
from taxi_demand.warehouse import ANALYSIS_SQL, analysis_queries

INTRO = """# New York yellow taxi demand: exploratory analysis

This notebook runs each query of the SQL library in `src/taxi_demand/sql/analysis/` against the
DuckDB warehouse that `make build` creates, and shows its result. The written findings are in
`reports/findings.md`; the dashboard is in `dashboard/app.py`.

Every query reads usable trips only. The data quality audit (last section) shows how many source
rows each check removed."""

SETUP = """from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from portfolio_common.plotting import apply_style

from taxi_demand.warehouse import run_query

%matplotlib inline
apply_style()
pd.set_option("display.float_format", "{:,.4g}".format)
con = duckdb.connect(WAREHOUSE, read_only=True)"""

PLOTS = {
    "a01_monthly_trend": """ax = result.plot.bar(
    x="month", y="trips_per_day", legend=False, color="#2a78d6", rot=0
)
ax.set_ylabel("Trips per day")
plt.show()""",
    "a02_hour_weekday": """grid = result.pivot(index="weekday", columns="hour", values="mean_trips")
fig, ax = plt.subplots(figsize=(9, 3.2))
image = ax.imshow(grid, aspect="auto", cmap="Blues")
ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_xlabel("Hour of pickup")
ax.grid(False)
fig.colorbar(image, label="Mean trips per hour")
plt.show()""",
    "a03_speed_by_hour": """speed = result.pivot(index="hour", columns="day_type", values="mean_speed_mph")
ax = speed.plot(color=["#2a78d6", "#eb6834"])
ax.set_ylabel("Mean speed (mph)")
ax.set_ylim(bottom=0)
plt.show()""",
    "a05_zone_concentration": """ax = result.plot(
    x="zone_rank", y="cumulative_share", legend=False, color="#2a78d6"
)
ax.set_ylabel("Cumulative share of trips")
ax.set_xlabel("Pickup zones, ranked by trips")
plt.show()""",
    "a08_zone_cohorts": """cohorts = result.pivot(
    index="month", columns="volume_quartile", values="index_january_100"
)
ax = cohorts.plot(color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])
ax.set_ylabel("Trips per day, January = 100")
ax.legend(title="January volume quartile", frameon=False)
plt.show()""",
}


def describe(name: str) -> str:
    """The leading comment lines of a query file, as prose."""
    lines = (ANALYSIS_SQL / f"{name}.sql").read_text(encoding="utf-8").splitlines()
    return " ".join(line.removeprefix("--").strip() for line in lines if line.startswith("--"))


def build(warehouse: Path) -> nbformat.NotebookNode:
    cells = [new_markdown_cell(INTRO), new_code_cell(f"WAREHOUSE = {str(warehouse)!r}\n" + SETUP)]
    for name in analysis_queries():
        cells.append(new_markdown_cell(f"## {name}\n\n{describe(name)}"))
        cells.append(new_code_cell(f'result = run_query(con, "{name}")\nresult.head(30)'))
        if name in PLOTS:
            cells.append(new_code_cell(PLOTS[name]))
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", type=Path, default=config.WAREHOUSE_PATH)
    parser.add_argument("--out", type=Path, default=config.PROJECT_DIR / "notebooks" / "analysis.ipynb")
    parser.add_argument("--no-execute", action="store_true")
    args = parse_args(parser, __file__)
    notebook = build(args.warehouse.resolve())
    if not args.no_execute:
        NotebookClient(notebook, timeout=600, kernel_name="python3").execute()
    # The warehouse path is local to the machine that ran it; keep the committed notebook portable.
    notebook.cells[1].source = notebook.cells[1].source.replace(
        repr(str(args.warehouse.resolve())), '"../artifacts/taxi.duckdb"'
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.out)
    print(f"notebook written to {args.out}")


if __name__ == "__main__":
    main()
