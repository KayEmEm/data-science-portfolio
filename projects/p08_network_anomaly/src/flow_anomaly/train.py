"""Fit every detector of every experiment and store its scores on the validation and test flows.

Experiments:

* ``main``: all detectors on a training sample with 1 percent attacks, all features;
* ``contamination_<share>``: Isolation Forest and the autoencoder on samples with other attack shares;
* ``no_ttl``: four detectors without the time to live features (an ablation).

Every finished detector is stored under ``artifacts/<experiment>/`` and skipped when the command
runs again, and the autoencoder saves every epoch, so a stopped run resumes.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flow_anomaly import detectors
from flow_anomaly.config import (
    ARTIFACTS_DIR,
    CONTAMINATION_SHARES,
    DATA_DIR,
    FIT_ROWS,
    NEIGHBOR_ROWS,
    NUMERIC,
    QUICK_AUTOENCODER,
    QUICK_FOREST,
    QUICK_SUPERVISED,
    SEED,
    SUPERVISED_ROWS,
    TRAIN_ATTACK_SHARE,
    TTL_FEATURES,
    AutoencoderSettings,
    ForestSettings,
    SupervisedSettings,
)
from flow_anomaly.data import load
from flow_anomaly.features import FlowPreprocessor

SCORED = ["fit", "validation", "test"]


def experiments() -> dict[str, dict]:
    out = {
        "main": {
            "attack_share": TRAIN_ATTACK_SHARE,
            "numeric": list(NUMERIC),
            "detectors": detectors.UNSUPERVISED + ["lightgbm"],
        }
    }
    for share in CONTAMINATION_SHARES:
        if share != TRAIN_ATTACK_SHARE:
            out[contamination_name(share)] = {
                "attack_share": share,
                "numeric": list(NUMERIC),
                "detectors": ["iforest", "autoencoder"],
            }
    out["no_ttl"] = {
        "attack_share": TRAIN_ATTACK_SHARE,
        "numeric": [c for c in NUMERIC if c not in TTL_FEATURES],
        "detectors": ["iforest", "ecod", "autoencoder", "lightgbm"],
    }
    return out


def contamination_name(share: float) -> str:
    return "main" if share == TRAIN_ATTACK_SHARE else f"contamination_{share:g}"


def fit_sample(flows: pd.DataFrame, share: float, rows: int, seed: int) -> np.ndarray:
    """Row positions of a training sample with the given share of attacks, in random order."""
    rng = np.random.default_rng(seed)
    train = flows["split"].to_numpy() == "train"
    normal = np.flatnonzero(train & (flows["label"].to_numpy() == 0))
    attack = np.flatnonzero(train & (flows["label"].to_numpy() == 1))
    n_attack = int(round(share * rows))
    n_normal = rows - n_attack
    if n_normal > len(normal):
        n_normal = len(normal)
        n_attack = min(len(attack), int(round(share / (1 - share) * n_normal))) if share < 1 else len(attack)
    n_attack = min(n_attack, len(attack))
    chosen = np.concatenate(
        [rng.choice(normal, n_normal, replace=False), rng.choice(attack, n_attack, replace=False)]
    )
    return chosen[rng.permutation(len(chosen))]


def labeled_sample(flows: pd.DataFrame, split: str, rows: int, seed: int) -> np.ndarray:
    positions = np.flatnonzero(flows["split"].to_numpy() == split)
    if len(positions) > rows:
        positions = np.sort(np.random.default_rng(seed).choice(positions, rows, replace=False))
    return positions


class Timer:
    def __init__(self, path: Path):
        self.path = path
        self.times = json.loads(path.read_text()) if path.exists() else {}

    def record(self, name: str, seconds: float) -> None:
        self.times[name] = round(seconds, 1)
        self.path.write_text(json.dumps(self.times, indent=2))


def run_experiment(
    name: str,
    setup: dict,
    flows: pd.DataFrame,
    artifacts: Path,
    forest: ForestSettings,
    ae: AutoencoderSettings,
    supervised: SupervisedSettings,
    fit_rows: int,
    neighbor_rows: int,
    supervised_rows: int,
    force: bool,
) -> None:
    out = artifacts / name
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    timer = Timer(out / "timings.json")
    sample_path = out / "fit_rows.npy"
    if sample_path.exists() and not force:
        sample = np.load(sample_path)
    else:
        sample = fit_sample(flows, setup["attack_share"], fit_rows, SEED)
        np.save(sample_path, sample)
    prep_path = out / "preprocessor.joblib"
    if prep_path.exists() and not force:
        prep = joblib.load(prep_path)
    else:
        prep = FlowPreprocessor(numeric=setup["numeric"]).fit(flows.iloc[sample])
        joblib.dump(prep, prep_path)
    split = flows["split"].to_numpy()
    matrices = {
        "fit": prep.transform(flows.iloc[sample]),
        "validation": prep.transform(flows.loc[split == "validation"]),
        "test": prep.transform(flows.loc[split == "test"]),
    }
    share = float(flows["label"].to_numpy()[sample].mean())
    print(
        f"{name}: training sample {len(sample):,} flows ({share:.2%} attacks), "
        f"{matrices['fit'].shape[1]} features; validation {len(matrices['validation']):,}, "
        f"test {len(matrices['test']):,}"
    )
    (out / "setup.json").write_text(
        json.dumps({**setup, "fit_rows": len(sample), "fit_attack_share": share}, indent=2)
    )

    fitted = {}
    for detector in setup["detectors"]:
        model_path = out / "models" / f"{detector}.joblib"
        done = all((out / "scores" / f"{detector}_{part}.npy").exists() for part in SCORED)
        if done and not force:
            print(f"  {detector}: stored")
            if detector in detectors.ENSEMBLE_MEMBERS and "ensemble" in setup["detectors"]:
                fitted[detector] = joblib.load(model_path)
            continue
        start = time.perf_counter()
        if model_path.exists() and not force:
            model = joblib.load(model_path)
        elif detector == "lightgbm":
            train_rows = labeled_sample(flows, "train", supervised_rows, SEED)
            val_rows = labeled_sample(flows, "validation", supervised_rows // 2, SEED)
            labels = flows["label"].to_numpy()
            model = detectors.Supervised(supervised).fit(
                prep.transform(flows.iloc[train_rows]),
                labels[train_rows],
                prep.transform(flows.iloc[val_rows]),
                labels[val_rows],
            )
        else:
            x_fit = (
                matrices["fit"][:neighbor_rows]
                if detectors.SPECS[detector].sample == "neighbor"
                else matrices["fit"]
            )
            model = detectors.make(detector, forest, ae, out / "models", fitted).fit(x_fit)
        joblib.dump(model, model_path)
        timer.record(f"{detector}_fit", time.perf_counter() - start)
        fitted[detector] = model
        start = time.perf_counter()
        for part in SCORED:
            np.save(out / "scores" / f"{detector}_{part}.npy", model.score(matrices[part]).astype("float64"))
        timer.record(f"{detector}_score", time.perf_counter() - start)
        print(
            f"  {detector}: fit {timer.times[f'{detector}_fit']} s, "
            f"score {timer.times[f'{detector}_score']} s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--only", nargs="*", help="experiments to run (default: all)")
    parser.add_argument("--quick", action="store_true", help="small settings for the smoke run")
    parser.add_argument("--force", action="store_true", help="refit stored detectors")
    args = parser.parse_args()
    flows = load(args.data_dir)
    quick = args.quick
    for name, setup in experiments().items():
        if args.only and name not in args.only:
            continue
        run_experiment(
            name,
            setup,
            flows,
            args.artifacts,
            QUICK_FOREST if quick else ForestSettings(),
            QUICK_AUTOENCODER if quick else AutoencoderSettings(),
            QUICK_SUPERVISED if quick else SupervisedSettings(),
            min(FIT_ROWS, 4_000) if quick else FIT_ROWS,
            min(NEIGHBOR_ROWS, 1_000) if quick else NEIGHBOR_ROWS,
            min(SUPERVISED_ROWS, 8_000) if quick else SUPERVISED_ROWS,
            args.force,
        )


if __name__ == "__main__":
    main()
