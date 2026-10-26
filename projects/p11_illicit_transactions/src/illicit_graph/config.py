"""Paths, file layout, temporal split and model settings for project 11."""

from dataclasses import dataclass
from pathlib import Path

from portfolio_common.config import settings as yaml_settings

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"
NEO4J_DIR = PROJECT_DIR / "neo4j"

SOURCE_PAGE = "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
SEED = 42

# The release: three CSV files. The features file has no header row: transaction id, time step,
# then 93 local features and 72 features aggregated over the one hop neighborhood.
FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"
RAW_FILES = [FEATURES_FILE, CLASSES_FILE, EDGES_FILE]
N_LOCAL = 93
N_AGGREGATED = 72
LOCAL = [f"local_{i:02d}" for i in range(1, N_LOCAL + 1)]
AGGREGATED = [f"agg_{i:02d}" for i in range(1, N_AGGREGATED + 1)]
TIME_STEPS = 49

# Counts as documented for the release, recalled when this project was written; ``make data``
# compares them with the files and records the result, and does not stop on a difference.
DOCUMENTED_COUNTS = {"nodes": 203_769, "edges": 234_355, "illicit": 4_545, "licit": 42_019}

# Class codes of the classes file and the label written to the node table.
CLASS_CODES = {"1": 1, "2": 0, "unknown": -1}  # 1 illicit, 0 licit, -1 not labelled
ILLICIT, LICIT, UNKNOWN = 1, 0, -1

# Temporal split. Time steps 1 to 34 are the training period of the published benchmark and 35 to
# 49 its test period; the last four training steps are held out here as validation, for early
# stopping and for the decision threshold. Edges never join two time steps, so each split is its
# own set of graphs and no message passes between periods.
SPLIT_STEPS = {"train": range(1, 31), "validation": range(31, 35), "test": range(35, 50)}
SPLITS = list(SPLIT_STEPS)
# From time step 43 a large dark market was closed; illicit behavior after it differs.
SHUTDOWN_STEP = 43

NODES_PATH = DATA_DIR / "nodes.parquet"  # tx_id, time_step, label, split, features
EDGES_PATH = DATA_DIR / "edges.parquet"  # src, dst as row positions of the node table
AUDIT_PATH = DATA_DIR / "audit.json"

# Network analysis: betweenness is estimated from this many source nodes per time step graph.
BETWEENNESS_SAMPLES = 256
LOUVAIN_RESOLUTION = 1.0
MIN_COMMUNITY = 5  # communities with fewer nodes are not profiled
SUBGRAPH_HOPS = 2  # radius of the subgraphs drawn around illicit transactions
BOOTSTRAP = 200


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class LightGBMSettings:
    n_estimators: int = 1000
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.5
    early_stopping: int = 100


@dataclass(frozen=True)
@yaml_settings(PROJECT_DIR)
class GNNSettings:
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.3
    lr: float = 0.005
    weight_decay: float = 5e-4
    epochs: int = 400
    patience: int = 40
    illicit_weight: float = 0.0  # 0: inverse class frequency of the training labels
    threads: int = 4


GNN_SEEDS = [42, 43, 44]

QUICK_LIGHTGBM = LightGBMSettings(n_estimators=60, num_leaves=7, min_child_samples=5, early_stopping=15)
QUICK_GNN = GNNSettings(hidden=16, epochs=25, patience=10)
