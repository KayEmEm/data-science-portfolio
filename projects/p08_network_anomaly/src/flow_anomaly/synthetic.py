"""Synthetic flow files in the UNSW-NB15 layouts, for the tests and ``make smoke``.

Their numbers say nothing about UNSW-NB15. Normal flows come from four traffic profiles (web, DNS,
FTP, other TCP), each with its own scale of bytes, packets, durations and connection counts. Each
attack type changes some features of a profile by its own factor: some types differ strongly
(Generic, DoS), others only a little (Fuzzers, Reconnaissance). Most attacks also carry a high
time to live, as an artifact of a test bed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from flow_anomaly.config import FEATURES_FILE, FULL_COLUMNS, FULL_FILES, FULL_RENAME, PARTITION_FILES, SEED

PROFILES = {
    # proto, service, state, share, scale of bytes, packets, duration (seconds)
    "web": ("tcp", "http", "fin", 0.45, 4000.0, 12.0, 0.8),
    "dns": ("udp", "dns", "con", 0.30, 120.0, 2.0, 0.01),
    "ftp": ("tcp", "ftp", "fin", 0.05, 2500.0, 20.0, 2.0),
    "tcp": ("tcp", "-", "fin", 0.20, 1500.0, 8.0, 0.4),
}
# attack type: share of attacks, profile, factor on bytes, packets, duration, connection counts
ATTACKS = {
    "Generic": (0.35, "dns", 0.3, 1.0, 0.1, 6.0),
    "Exploits": (0.25, "web", 3.0, 2.0, 2.0, 2.0),
    "Fuzzers": (0.12, "tcp", 1.3, 1.2, 1.2, 1.3),
    "DoS": (0.10, "web", 0.2, 6.0, 0.2, 8.0),
    "Reconnaissance": (0.08, "tcp", 0.6, 0.8, 0.8, 1.5),
    "Analysis": (0.03, "tcp", 0.5, 0.5, 3.0, 2.0),
    "Backdoor": (0.03, "tcp", 0.5, 0.6, 3.0, 2.0),
    "Shellcode": (0.03, "ftp", 0.4, 0.5, 0.5, 1.0),
    "Worms": (0.01, "web", 5.0, 3.0, 4.0, 3.0),
}
START = 1_421_927_000


def _profile_rows(rng, n: int, profile: str, f_bytes=1.0, f_pkts=1.0, f_dur=1.0, f_ct=1.0) -> dict:
    proto, service, state, _, s_bytes, s_pkts, s_dur = PROFILES[profile]

    def logn(scale, sigma=0.6):
        return scale * rng.lognormal(0.0, sigma, n)

    spkts = np.maximum(1, np.round(logn(s_pkts * f_pkts))).astype(int)
    dpkts = np.maximum(0, np.round(spkts * rng.uniform(0.5, 1.2, n))).astype(int)
    sbytes = np.round(logn(s_bytes * f_bytes)).astype(int) + 40
    dbytes = np.round(logn(s_bytes * f_bytes * 2.0)).astype(int) * (dpkts > 0)
    dur = logn(s_dur * f_dur, 0.8)
    ct = lambda base: np.maximum(1, rng.poisson(base * f_ct, n))  # noqa: E731
    tcp = proto == "tcp"
    return {
        "proto": np.full(n, proto),
        "service": np.full(n, service),
        "state": np.full(n, state),
        "dur": dur,
        "sbytes": sbytes,
        "dbytes": dbytes,
        "sttl": np.full(n, 31),
        "dttl": np.full(n, 29),
        "sloss": rng.binomial(spkts, 0.01),
        "dloss": rng.binomial(dpkts, 0.01),
        "sload": sbytes * 8 / np.maximum(dur, 1e-3),
        "dload": dbytes * 8 / np.maximum(dur, 1e-3),
        "spkts": spkts,
        "dpkts": dpkts,
        "swin": np.where(tcp, 255, 0),
        "dwin": np.where(tcp, 255, 0),
        "stcpb": rng.integers(0, 2**32 - 1, n) * tcp,
        "dtcpb": rng.integers(0, 2**32 - 1, n) * tcp,
        "smeansz": sbytes // spkts,
        "dmeansz": dbytes // np.maximum(dpkts, 1),
        "trans_depth": (service == "http") * rng.integers(0, 3, n),
        "res_bdy_len": (service == "http") * np.round(logn(s_bytes)).astype(int),
        "sjit": logn(20.0 * f_dur, 1.0),
        "djit": logn(15.0 * f_dur, 1.0),
        "sintpkt": dur * 1000 / spkts,
        "dintpkt": dur * 1000 / np.maximum(dpkts, 1),
        "tcprtt": tcp * logn(0.05),
        "synack": tcp * logn(0.025),
        "ackdat": tcp * logn(0.025),
        "is_sm_ips_ports": (rng.random(n) < 0.002).astype(int),
        "ct_state_ttl": rng.integers(0, 3, n),
        "ct_flw_http_mthd": (service == "http") * rng.integers(0, 2, n),
        "is_ftp_login": (service == "ftp") * (rng.random(n) < 0.5),
        "ct_ftp_cmd": (service == "ftp") * rng.integers(0, 3, n),
        "ct_srv_src": ct(4),
        "ct_srv_dst": ct(4),
        "ct_dst_ltm": ct(2),
        "ct_src_ltm": ct(2),
        "ct_src_dport_ltm": ct(1),
        "ct_dst_sport_ltm": ct(1),
        "ct_dst_src_ltm": ct(2),
    }


def make_flows(n_rows: int = 20_000, attack_share: float = 0.12, seed: int = SEED) -> pd.DataFrame:
    """Flows with the column names of the full records."""
    rng = np.random.default_rng(seed)
    n_attacks = int(round(n_rows * attack_share))
    n_normal = n_rows - n_attacks
    parts = []
    shares = np.array([p[3] for p in PROFILES.values()])
    for profile, count in zip(PROFILES, rng.multinomial(n_normal, shares / shares.sum()), strict=True):
        rows = _profile_rows(rng, count, profile)
        rows["attack_cat"] = np.full(count, "")
        parts.append(pd.DataFrame(rows))
    shares = np.array([a[0] for a in ATTACKS.values()])
    counts = rng.multinomial(n_attacks, shares / shares.sum())
    for (name, (_, profile, fb, fp, fd, fc)), count in zip(ATTACKS.items(), counts, strict=True):
        rows = _profile_rows(rng, count, profile, fb, fp, fd, fc)
        high_ttl = rng.random(count) < 0.7
        rows["sttl"] = np.where(high_ttl, 254, rows["sttl"])
        rows["dttl"] = np.where(high_ttl, 252, rows["dttl"])
        rows["ct_state_ttl"] = np.where(high_ttl, rng.integers(1, 7, count), rows["ct_state_ttl"])
        rows["attack_cat"] = np.full(count, name)
        parts.append(pd.DataFrame(rows))
    flows = pd.concat(parts, ignore_index=True)
    flows = flows.iloc[rng.permutation(len(flows))].reset_index(drop=True)
    flows["label"] = (flows["attack_cat"] != "").astype(int)
    flows["srcip"] = np.where(flows["label"] == 1, "175.45.176.1", "59.166.0.1")
    flows["dstip"] = "149.171.126.6"
    flows["sport"] = rng.integers(1024, 65535, len(flows))
    flows["dsport"] = np.select([flows["service"] == "http", flows["service"] == "dns"], [80, 53], 21)
    flows["stime"] = START + np.sort(rng.integers(0, 86_400, len(flows)))
    flows["ltime"] = flows["stime"] + np.ceil(flows["dur"]).astype(int)
    return flows[FULL_COLUMNS]


def write_full(folder: Path, flows: pd.DataFrame) -> Path:
    """Four CSV files without a header row and the features file, as in the full release."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(FULL_FILES, np.array_split(np.arange(len(flows)), len(FULL_FILES)), strict=True):
        part = flows.iloc[rows].copy()
        # The release writes a blank cell for normal flows and spaces around some attack types.
        part["attack_cat"] = part["attack_cat"].replace({"Fuzzers": " Fuzzers ", "Backdoor": "Backdoors"})
        part.to_csv(folder / name, header=False, index=False)
    names = ["srcip", "sport", "dstip", "dsport"] + FULL_COLUMNS[4:]
    names[names.index("ct_src_ltm")] = "ct_src_ ltm"
    names[names.index("label")] = "Label"
    pd.DataFrame(
        {"No.": range(1, len(names) + 1), "Name": names, "Type ": "x", "Description": "synthetic"}
    ).to_csv(folder / FEATURES_FILE, index=False)
    return folder


def write_partition(folder: Path, flows: pd.DataFrame, seed: int = SEED) -> Path:
    """Two CSV files with a header row, as in the official training and testing sets."""
    folder.mkdir(parents=True, exist_ok=True)
    frame = flows.rename(columns=FULL_RENAME).drop(
        columns=["srcip", "sport", "dstip", "dsport", "stime", "ltime"]
    )
    frame["attack_cat"] = frame["attack_cat"].replace("", "Normal")
    frame.insert(0, "id", np.arange(1, len(frame) + 1))
    test = np.random.default_rng(seed).random(len(frame)) < 0.32
    for role, rows in (("train", ~test), ("test", test)):
        frame.loc[rows].to_csv(folder / PARTITION_FILES[role], index=False)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--layout", choices=["full", "partition"], default="full")
    parser.add_argument("--rows", type=int, default=20_000)
    args = parse_args(parser, __file__)
    flows = make_flows(args.rows)
    writer = write_full if args.layout == "full" else write_partition
    print(f"wrote {len(flows):,} synthetic flows to {writer(args.folder, flows)}")


if __name__ == "__main__":
    main()
