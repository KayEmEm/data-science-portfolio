"""Paths, source layouts, split and detector settings for project 8."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

SOURCE_PAGE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
SEED = 42

# The full records: four CSV files without a header row, and a file that names the 49 columns.
FULL_FILES = ["UNSW-NB15_1.csv", "UNSW-NB15_2.csv", "UNSW-NB15_3.csv", "UNSW-NB15_4.csv"]
FEATURES_FILE = "NUSW-NB15_features.csv"
# Column order of the full records, used when the features file is absent. Names are lower case
# without spaces, as ``normalize_name`` writes them.
FULL_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
    "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt",
    "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]  # fmt: skip
# Names of the full records that differ from the names of the official partition.
FULL_RENAME = {
    "smeansz": "smean",
    "dmeansz": "dmean",
    "sintpkt": "sinpkt",
    "dintpkt": "dinpkt",
    "res_bdy_len": "response_body_len",
}
# The official partition: two CSV files with a header row.
PARTITION_FILES = {"train": "UNSW_NB15_training-set.csv", "test": "UNSW_NB15_testing-set.csv"}

# Counts as documented for the release, recalled when this project was written; ``make data``
# compares them with the files and records the result, and does not stop on a difference.
DOCUMENTED_COUNTS = {
    "full": {"rows": 2_540_044, "attacks": 321_283},
    "partition": {"train": 175_341, "test": 82_332},
}

CATEGORICAL = ["proto", "service", "state"]
# Numeric flow features present in both layouts. Left out: addresses and ports (they name the test
# bed hosts, not the behavior of a flow), start and end times, and the TCP base sequence numbers
# stcpb and dtcpb (random by design).
NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]  # fmt: skip
# Time to live values of the test bed: they may separate attack traffic by the network it came from
# rather than by its behavior, so an ablation drops them.
TTL_FEATURES = ["sttl", "dttl", "ct_state_ttl"]
TOP_CATEGORIES = 12  # most frequent values per categorical column; the rest become "other"

NORMAL = "Normal"
ATTACK_NAMES = {
    "normal": NORMAL,
    "fuzzers": "Fuzzers",
    "analysis": "Analysis",
    "backdoor": "Backdoor",
    "backdoors": "Backdoor",
    "dos": "DoS",
    "exploits": "Exploits",
    "generic": "Generic",
    "reconnaissance": "Reconnaissance",
    "shellcode": "Shellcode",
    "worms": "Worms",
}

FLOWS_PATH = DATA_DIR / "flows.parquet"  # canonical columns, label, attack_cat, split
AUDIT_PATH = DATA_DIR / "audit.json"

# Split of the full records: stratified by attack type, in these shares. The official partition
# keeps its test file and splits its training file into train and validation.
SPLIT_SHARES = {"train": 0.6, "validation": 0.2, "test": 0.2}
PARTITION_VALIDATION_SHARE = 0.2

# Unsupervised detectors see no labels. Their training sample is mostly normal traffic with a small
# share of attacks, as a log from a network that is attacked now and then. The label is used only
# to build this sample and to score the detectors.
TRAIN_ATTACK_SHARE = 0.01
CONTAMINATION_SHARES = [0.0, 0.01, 0.05, 0.2]  # study: how attacks in the training sample hurt
CONTAMINATION_SEEDS = [42, 43, 44]  # each share is drawn and fitted three times
FIT_ROWS = 200_000  # rows of the training sample for most detectors
NEIGHBOR_ROWS = 20_000  # rows for the neighbor based detectors (cost grows with the square)
SUPERVISED_ROWS = 400_000
# The test splits hold many more attacks than a monitored network would see, which inflates PR AUC.
# PR AUC is therefore also reported as if attacks made up this share of the flows.
RESCALED_PREVALENCE = 0.01
TARGET_FPR = 0.01  # alarm threshold: 1 percent of normal validation flows flagged
ALERT_BUDGETS = [100, 1000]  # precision of the top k test alerts
BOOTSTRAP = 200


@dataclass(frozen=True)
class AutoencoderSettings:
    hidden: tuple[int, ...] = (64, 32)
    code: int = 8
    epochs: int = 150
    patience: int = 5
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    holdout: float = 0.1  # share of the training sample kept for early stopping (no labels)
    threads: int = 4


@dataclass(frozen=True)
class ForestSettings:
    n_estimators: int = 300
    max_samples: int = 512


@dataclass(frozen=True)
class SupervisedSettings:
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping: int = 50


LOF_NEIGHBORS = 35
HBOS_BINS = 20

QUICK_AUTOENCODER = AutoencoderSettings(hidden=(16,), code=4, epochs=4, patience=2, batch_size=256)
QUICK_FOREST = ForestSettings(n_estimators=50, max_samples=128)
QUICK_SUPERVISED = SupervisedSettings(n_estimators=40, num_leaves=15, min_child_samples=5, early_stopping=10)
