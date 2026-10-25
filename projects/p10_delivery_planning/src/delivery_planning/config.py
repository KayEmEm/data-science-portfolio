"""Paths, benchmark source, routing settings and the synthetic driver scenario for project 10."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

SEED = 42

# The 56 Solomon instances (100 customers each) and their best known solutions, in the VRPLIB
# format of the PyVRP instance collection (MIT license), pinned to one commit. Each file is checked
# against the SHA256 in ``solomon_files.json``. The collection takes the best known solutions from
# CVRPLIB and follows the DIMACS convention: distances and durations truncated to one decimal, and
# the cost is the total distance, whatever the number of vehicles.
SOURCE_REPO = "https://github.com/PyVRP/Instances"
SOURCE_COMMIT = "7474b068a8effa7f39448b241314b615c9271525"
SOURCE_URL = f"https://raw.githubusercontent.com/PyVRP/Instances/{SOURCE_COMMIT}/VRPTW/Solomon/{{name}}"
MANIFEST_PATH = Path(__file__).with_name("solomon_files.json")
CLASSES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]

# Distances and times are integers in tenths of a unit (DIMACS one decimal truncation), so that
# OR-Tools works on the same numbers as the best known solutions.
SCALE = 10

# Routing configurations of the benchmark. The first two stop at the first solution that the
# construction heuristic finds and are the baselines; the others improve it with guided local search
# until the time limit.
ROUTING_CONFIGS = {
    "cheapest_arc": {"first": "PATH_CHEAPEST_ARC", "metaheuristic": None, "seconds": None},
    "cheapest_insertion": {"first": "PARALLEL_CHEAPEST_INSERTION", "metaheuristic": None, "seconds": None},
    "gls_10s": {
        "first": "PARALLEL_CHEAPEST_INSERTION",
        "metaheuristic": "GUIDED_LOCAL_SEARCH",
        "seconds": 10,
    },
    "gls_60s": {
        "first": "PARALLEL_CHEAPEST_INSERTION",
        "metaheuristic": "GUIDED_LOCAL_SEARCH",
        "seconds": 60,
    },
    "gls_300s": {
        "first": "PARALLEL_CHEAPEST_INSERTION",
        "metaheuristic": "GUIDED_LOCAL_SEARCH",
        "seconds": 300,
    },
}
MAIN_CONFIG = "gls_60s"
WORKERS = 4  # benchmark runs in parallel, one OR-Tools search per CPU core

# A week of deliveries for the shift model: one Solomon instance per day. R101 to R107 share the
# 100 customers and differ in how tight and how many the time windows are. The depot horizon of each
# instance maps onto one operating day from DAY_START to DAY_START + DAY_MINUTES.
WEEK = ["R101", "R102", "R103", "R104", "R105", "R106", "R107"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_START = 8 * 60  # minutes after midnight
DAY_MINUTES = 10 * 60

# Synthetic cost rates in cost units (CU). They set the trade off between distance, vehicles and
# driver hours; they are assumptions of the scenario, not data. One distance unit counts as 1 km.
COST_PER_KM = 0.45
COST_PER_VEHICLE_DAY = 40.0

# Driver rules of the shift model.
CHECK_MINUTES = 15  # vehicle check before and handover after each route
MAX_SPAN_HOURS = 11.0  # first route start to last route end in one shift
MIN_PAID_HOURS = 4.0  # a shift is paid for at least this long
OVERTIME_AFTER = 40.0  # weekly hours above this are paid at OVERTIME_FACTOR
OVERTIME_FACTOR = 1.5
AGENCY_RATE = 60.0  # CU per hour for a route that no own driver covers
AGENCY_MIN_HOURS = 4.0

# The synthetic driver pool: contract types with wage, weekly guarantee (paid whether worked or
# not), weekly maximum and maximum days per week. Each driver has days off and a daily window.
DRIVERS = 30
FULL_TIME_SHARE = 0.6
CONTRACTS = {
    "full_time": {"wage": 32.0, "guarantee": 30.0, "max_hours": 48.0, "max_days": 5},
    "part_time": {"wage": 36.0, "guarantee": 0.0, "max_hours": 30.0, "max_days": 4},
}
# Daily windows in hours after midnight; part time drivers draw one of the three.
FULL_TIME_WINDOW = (6.0, 21.0)
PART_TIME_WINDOWS = {"morning": (6.0, 14.0), "late": (11.0, 21.0), "day": (6.0, 21.0)}
DAYS_OFF = {"full_time": (1, 2), "part_time": (2, 4)}  # range of unavailable days per week

# Shift MIP scenarios of the solver comparison: the first days of the week and the first drivers of
# the pool. The small scenario fits the size limit of Gurobi's pip license.
SCENARIOS = {
    "small": {"days": 2, "drivers": 8},
    "medium": {"days": 4, "drivers": 18},
    "full": {"days": 7, "drivers": 30},
}
SOLVERS = ["highs", "cbc", "gurobi"]
MIP_SECONDS = 300
MIP_GAP = 1e-4

# Fleet size sensitivity: the week is planned again with a cap on vehicles per day, with the
# distance objective and with a cost aware objective that also charges vehicles and driver time.
FLEET_CAPS = [8, 10, 12, 14, 16, 18, 20, 25]
FLEET_SECONDS = 30
DROP_PENALTY = 500.0  # CU per customer left unserved when the fleet is too small to serve all
OBJECTIVES = ["distance", "cost_aware"]

# Quick settings for the tests and ``make smoke``.
QUICK = {
    "configs": ["cheapest_arc", "gls_10s"],
    "seconds": 2,
    "fleet_caps": [4, 6, 10],
    "fleet_seconds": 1,
    "mip_seconds": 10,
    "drivers": 12,
}
