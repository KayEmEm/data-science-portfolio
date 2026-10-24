"""Paths, cleaning rules, feature windows and clustering settings for project 9."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
SEED = 42

# The UCI release: one Excel workbook with a sheet per year. A CSV export of both sheets (as on
# Kaggle) is read as well, and so is a zip that holds either.
SOURCE_NAMES = ["online_retail_II.xlsx", "online_retail_II.csv"]
COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]
# Source spellings of the columns, after ``normalize_name``; the second set is the older Online
# Retail release (UCI 352), which has the same rows for 2010 and 2011.
RENAME = {
    "invoice": "invoice",
    "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "price",
    "unitprice": "price",
    "customerid": "customer_id",
    "country": "country",
}

# Counts as documented on the UCI page, recalled when this project was written; ``make data``
# compares them with the file and records the result, and does not stop on a difference.
DOCUMENTED_ROWS = 1_067_371

# Product rows have a stock code of five digits, with an optional suffix. Other codes are postage,
# fees, discounts, manual adjustments and samples; they are kept out of the product mix and out of
# the spend features.
PRODUCT_CODE = r"^\d{5}"

TRANSACTIONS_PATH = DATA_DIR / "transactions.parquet"
AUDIT_PATH = DATA_DIR / "audit.json"

# The last OUTCOME_DAYS of the data are held out: the features describe each customer up to the
# cutoff, and the segments are checked on what the customers spend after it. With the UCI data
# the cutoff is 2011-09-09 and the outcome window holds the Christmas season of 2011.
OUTCOME_DAYS = 91

# Behavior features of each customer; the skewed ones are log transformed before scaling.
BEHAVIOR = [
    "recency_days",
    "frequency",
    "monetary",
    "mean_basket",
    "tenure_days",
    "distinct_products",
    "return_share",
]
LOG_FEATURES = ["frequency", "monetary", "mean_basket", "distinct_products"]
RFM = ["recency_days", "frequency", "monetary"]
# The category block is scaled so that its total variance equals MIX_WEIGHT times that of the
# behavior block, whatever the number of categories.
MIX_WEIGHT = 0.5

SCALINGS = ["raw", "standard", "robust", "quantile"]
FEATURE_SETS = ["rfm", "rfm_mix"]
K_RANGE = list(range(2, 11))
KMEANS_INIT = 10
GAP_REFERENCES = 20
BOOTSTRAP_FITS = 50
# A k means solution is a candidate when it reproduces under resampling and has no tiny segment.
MIN_STABILITY = 0.80  # mean adjusted Rand index of bootstrap refits against the full fit
MIN_SEGMENT_SHARE = 0.02
SELECTION_K = list(range(3, 9))  # marketing can act on three to eight segments

DBSCAN_MIN_SAMPLES = 10
DBSCAN_QUANTILES = [0.5, 0.75, 0.9, 0.95]  # eps as quantiles of the distance to the 10th neighbor
HDBSCAN_SIZES = [25, 50, 100]
HDBSCAN_MIN_SAMPLES = 10

UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
OUTCOME_BOOTSTRAP = 200

# Quick settings for the tests and ``make smoke``.
QUICK = {
    "k_range": list(range(2, 7)),
    "selection_k": list(range(3, 6)),
    "gap_references": 4,
    "bootstrap_fits": 6,
    "outcome_bootstrap": 20,
    "umap_neighbors": 10,
}
