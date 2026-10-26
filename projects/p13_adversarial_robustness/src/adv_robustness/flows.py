"""Evasion, poisoning and randomized smoothing on the network flow detectors of project 8.

Targets, built with the code of project 8: the supervised LightGBM model (upper reference) and the
selected unsupervised detector, Isolation Forest, each with its alarm threshold at 1 percent false
alarms on normal validation flows.

* Evasion: an attacker changes its own detected test flows within ``FlowBudget`` (padding, extra
  packets, delays, initial TTL) and keeps the change with the lowest score. Black box: it needs
  only the score of each try. A HopSkipJump attack (Adversarial Robustness Toolbox) with no
  feasibility limits gives the unconstrained reference.
* Defenses against evasion: LightGBM without the TTL features, and LightGBM trained on feasible
  changes of the training attacks (adversarial training).
* Poisoning: flows of one attack type are slipped into the unlabeled training traffic of
  Isolation Forest, so that this type looks normal. Defenses: refit after dropping the highest
  scored training flows, and removal of exact duplicate flows.
* Randomized smoothing (Cohen et al., 2019) of a LightGBM model trained with Gaussian noise on the
  standardized numeric features, with a certified radius per flow.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from flow_anomaly.config import (
    NUMERIC,
    QUICK_FOREST,
    QUICK_SUPERVISED,
    TTL_FEATURES,
    ForestSettings,
    SupervisedSettings,
)
from flow_anomaly.data import load
from flow_anomaly.detectors import Forest, Supervised
from flow_anomaly.features import FlowPreprocessor
from flow_anomaly.metrics import at_threshold, threshold_at_fpr
from flow_anomaly.train import fit_sample, labeled_sample
from portfolio_common.config import parse_args
from portfolio_common.seed import set_seed
from scipy.stats import beta, norm

from adv_robustness.config import (
    ART_ROWS,
    ARTIFACTS_DIR,
    ATTACKED_ROWS,
    DATA_DIR,
    FLOW_TARGET_FPR,
    POISON_SHARES,
    POISON_TYPE,
    SEED,
    TRIM_SHARE,
    FlowBudget,
    SmoothingSettings,
)

MTU_PAYLOAD = 1500  # mean sent packet size cannot exceed this many bytes
NORMAL_TTL_VALUES = 3  # most frequent initial TTL values of normal training flows that the attacker tries
FIT_ROWS = 200_000  # as in project 8


class Model:
    """A fitted detector with its preprocessor and alarm threshold."""

    def __init__(self, name: str, prep: FlowPreprocessor, detector, threshold: float):
        self.name, self.prep, self.detector, self.threshold = name, prep, detector, threshold

    def score(self, flows: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.detector.score(self.prep.transform(flows)), dtype="float64")


def perturb(
    flows: pd.DataFrame, f_bytes: float, f_packets: float, f_duration: float, ttl=None
) -> pd.DataFrame:
    """Flows after padding, extra packets and delays; the derived features are scaled to match.

    Sizes and counts only grow. ``smean`` (mean sent packet size), ``sload`` (sent bits per second)
    and ``sinpkt`` (mean time between sent packets) are scaled by the ratio of their parts, so their
    units do not matter. Jitter is scaled with the duration (an assumption).
    """
    out = flows.copy()
    packets = np.ceil(out["spkts"] * f_packets)
    old_gaps = np.maximum(out["spkts"] - 1, 1)
    new_gaps = np.maximum(packets - 1, 1)
    out["sbytes"] = np.round(out["sbytes"] * f_bytes)
    out["spkts"] = packets
    out["dur"] = out["dur"] * f_duration
    out["smean"] = np.round(out["smean"] * f_bytes * flows["spkts"] / np.maximum(packets, 1))
    out["sload"] = out["sload"] * f_bytes / f_duration
    out["sinpkt"] = out["sinpkt"] * f_duration * old_gaps / new_gaps
    out["sjit"] = out["sjit"] * f_duration
    if ttl is not None:
        out["sttl"] = ttl
    return out


def feasible(flows: pd.DataFrame) -> np.ndarray:
    return (flows["smean"] <= MTU_PAYLOAD).to_numpy()


def grid(budget: FlowBudget, ttl_values: list, groups: str) -> list[tuple]:
    sizes = (
        list(itertools.product(budget.bytes_factor, budget.packets_factor, budget.duration_factor))
        if groups in ("size_timing", "all")
        else [(1.0, 1.0, 1.0)]
    )
    ttls = [None] + (ttl_values if budget.ttl and groups in ("ttl", "all") else [])
    return [(*s, t) for s, t in itertools.product(sizes, ttls)]


def evade(
    model: Model, flows: pd.DataFrame, budget: FlowBudget, ttl_values: list, groups: str
) -> pd.DataFrame:
    """The lowest score each flow reaches within the budget, and whether it falls below the threshold."""
    best = model.score(flows)
    best_change = np.zeros(len(flows), dtype=int)
    changes = grid(budget, ttl_values, groups)
    for i, change in enumerate(changes):
        if change == (1.0, 1.0, 1.0, None):
            continue
        tried = perturb(flows, *change)
        scores = np.where(feasible(tried), model.score(tried), np.inf)
        better = scores < best
        best[better], best_change[better] = scores[better], i
    return pd.DataFrame(
        {"score": best, "evaded": best <= model.threshold, "change": [changes[i] for i in best_change]},
        index=flows.index,
    )


def random_changes(flows: pd.DataFrame, budget: FlowBudget, ttl_values: list, seed: int) -> pd.DataFrame:
    """One random feasible change per flow (unchanged when infeasible), for adversarial training."""
    rng = np.random.default_rng(seed)
    changes = grid(budget, ttl_values, "all")
    pick = rng.integers(len(changes), size=len(flows))
    parts = []
    for i in np.unique(pick):
        part = flows[pick == i]
        tried = perturb(part, *changes[i])
        parts.append(tried.where(pd.Series(feasible(tried), index=part.index), part, axis=0))
    return pd.concat(parts).loc[flows.index]


def fit_supervised(
    flows, prep, settings, seed, train_rows, val_rows, extra: pd.DataFrame | None = None
) -> Model:
    labels = flows["label"].to_numpy()
    train = flows.iloc[train_rows]
    y = labels[train_rows]
    if extra is not None:
        train, y = pd.concat([train, extra]), np.concatenate([y, np.ones(len(extra), dtype=int)])
    detector = Supervised(settings, seed).fit(
        prep.transform(train), y, prep.transform(flows.iloc[val_rows]), labels[val_rows]
    )
    return Model("lightgbm", prep, detector, 0.0)


def calibrate(model: Model, flows: pd.DataFrame) -> Model:
    normal = flows[(flows["split"] == "validation") & (flows["label"] == 0)]
    model.threshold = threshold_at_fpr(model.score(normal), FLOW_TARGET_FPR)
    return model


def clean_metrics(model: Model, test: pd.DataFrame) -> dict:
    return at_threshold(test["label"].to_numpy(), model.score(test), model.threshold)


def detected_attacks(model: Model, test: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    attacks = test[test["label"] == 1]
    detected = attacks[model.score(attacks) > model.threshold]
    if len(detected) > rows:
        detected = detected.sample(rows, random_state=seed)
    return detected


def evasion_rows(model: Model, test: pd.DataFrame, budget: FlowBudget, ttl_values: list, rows: int) -> dict:
    target = detected_attacks(model, test, rows, SEED)
    out = {"attacked": len(target), "groups": {}}
    for groups in ("size_timing", "ttl", "all"):
        found = evade(model, target, budget, ttl_values, groups)
        by_type = found.groupby(target["attack_cat"].astype(str), observed=True)["evaded"].mean()
        out["groups"][groups] = {
            "evasion_rate": float(found["evaded"].mean()) if len(found) else 0.0,
            "by_type": {k: float(v) for k, v in by_type.items()},
        }
        # Recall after the attack: every detected flow of the test file attempts the change.
        attacks = test["label"].to_numpy() == 1
        recall = float((model.score(test[attacks]) > model.threshold).mean())
        out["groups"][groups]["recall_after"] = recall * (1 - out["groups"][groups]["evasion_rate"])
    return out


def hop_skip_jump(model: Model, test: pd.DataFrame, rows: int, quick: bool) -> dict:
    """Unconstrained reference: HopSkipJump in the standardized feature space, any feature may move."""
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import BlackBoxClassifier

    target = detected_attacks(model, test, rows, SEED + 1)
    x = model.prep.transform(target).astype("float64")
    low, high = x.min(axis=0) - 1, x.max(axis=0) + 1
    logit_threshold = np.log(model.threshold / (1 - model.threshold))

    def predict(batch: np.ndarray) -> np.ndarray:
        s = np.clip(model.detector.score(batch.astype("float32")), 1e-9, 1 - 1e-9)
        q = 1 / (1 + np.exp(-(np.log(s / (1 - s)) - logit_threshold)))  # 0.5 at the alarm threshold
        return np.column_stack([1 - q, q])

    classifier = BlackBoxClassifier(
        predict, x.shape[1:], 2, clip_values=(float(low.min()), float(high.max()))
    )
    attack = HopSkipJump(
        classifier,
        targeted=False,
        max_iter=2 if quick else 10,
        max_eval=200 if quick else 1000,
        init_eval=20 if quick else 100,
        verbose=False,
    )
    adversarial = attack.generate(x)
    evaded = predict(adversarial)[:, 1] < 0.5
    distance = np.linalg.norm(adversarial - x, axis=1)
    return {
        "attacked": len(target),
        "evasion_rate": float(evaded.mean()),
        "median_l2": float(np.median(distance[evaded])) if evaded.any() else float("nan"),
    }


def poisoning(flows, test, share_attack: float, settings: ForestSettings, quick: bool) -> dict:
    """Recall of the targeted and other attack types after poisoning, without and with defenses."""
    base_rows = fit_sample(flows, share_attack, 20_000 if quick else FIT_ROWS, SEED)
    train = flows["split"].to_numpy() == "train"
    pool = np.flatnonzero(train & (flows["attack_cat"].astype(str).to_numpy() == POISON_TYPE))
    rng = np.random.default_rng(SEED)
    is_target = test["attack_cat"].astype(str).to_numpy() == POISON_TYPE
    y = test["label"].to_numpy()
    out = {}
    for share in POISON_SHARES:
        n_poison = min(len(pool), int(round(share * len(base_rows) / max(1 - share, 1e-9))))
        rows = np.concatenate([base_rows, rng.choice(pool, n_poison, replace=False)])
        fit = flows.iloc[rows]
        entry = {"poison_flows": int(n_poison)}
        for defense in ("none", "trim", "dedup"):
            prep = FlowPreprocessor().fit(fit)
            x = prep.transform(fit)
            if defense == "dedup":
                x = np.unique(x, axis=0)
            model = Forest(settings, SEED).fit(x)
            if defense == "trim":
                scores = model.score(x)
                keep = scores <= np.quantile(scores, 1 - TRIM_SHARE)
                model = Forest(settings, SEED).fit(x[keep])
            detector = calibrate(Model("iforest", prep, model, 0.0), flows)
            alarm = detector.score(test) > detector.threshold
            entry[defense] = {
                "recall_target": float(alarm[is_target].mean()),
                "recall_other": float(alarm[(y == 1) & ~is_target].mean()),
                "fpr": float(alarm[y == 0].mean()),
            }
        out[f"{share:g}"] = entry
        print(f"  poisoning {share:g}: {POISON_TYPE} recall {entry['none']['recall_target']:.3f}")
    return out


def smoothing(flows, settings: SupervisedSettings, smooth: SmoothingSettings, quick: bool) -> dict:
    """Certified accuracy of smoothed LightGBM models in the standardized numeric feature space."""
    numeric = list(NUMERIC)
    prep = FlowPreprocessor(numeric=numeric, categorical=[]).fit(flows[flows["split"] == "train"])
    rng = np.random.default_rng(SEED)
    train_rows = labeled_sample(flows, "train", 400_000, SEED)
    val_rows = labeled_sample(flows, "validation", 200_000, SEED)
    x_tr, x_va = prep.transform(flows.iloc[train_rows]), prep.transform(flows.iloc[val_rows])
    y_tr, y_va = flows["label"].to_numpy()[train_rows], flows["label"].to_numpy()[val_rows]
    test = flows[flows["split"] == "test"]
    test = test.sample(min(smooth.rows if not quick else 300, len(test)), random_state=SEED)
    x_te, y_te = prep.transform(test), test["label"].to_numpy()
    n = 20 if quick else smooth.samples
    out = {}
    for sigma in smooth.sigmas:
        noisy = x_tr + rng.normal(0, sigma, x_tr.shape).astype("float32")
        base = Supervised(settings, SEED).fit(noisy, y_tr, x_va + rng.normal(0, sigma, x_va.shape), y_va)
        votes = np.zeros(len(x_te))
        for _ in range(n):
            votes += base.score(x_te + rng.normal(0, sigma, x_te.shape).astype("float32")) > 0.5
        top = np.where(votes >= n / 2, 1, 0)
        count = np.where(top == 1, votes, n - votes)
        p_lower = beta.ppf(smooth.alpha, count, n - count + 1)  # one sided Clopper Pearson bound
        radius = np.where(p_lower > 0.5, sigma * norm.ppf(np.clip(p_lower, 0.5, 1 - 1e-12)), -1.0)
        correct = top == y_te
        out[f"{sigma:g}"] = {
            "accuracy": float(correct.mean()),
            "recall": float(top[y_te == 1].mean()),
            "fpr": float(top[y_te == 0].mean()),
            "abstain": float((radius < 0).mean()),
            "certified_accuracy": {
                f"{r:g}": float((correct & (radius >= r)).mean()) for r in (0.0, 0.25, 0.5, 1.0)
            },
        }
        print(f"  smoothing sigma {sigma:g}: accuracy {out[f'{sigma:g}']['accuracy']:.3f}")
    return out


def change_size(flows: pd.DataFrame, budget: FlowBudget) -> float:
    """Median L2 distance, in the smoothing feature space, of the largest feasible change of each flow."""
    prep = FlowPreprocessor(numeric=list(NUMERIC), categorical=[]).fit(flows[flows["split"] == "train"])
    test = flows[(flows["split"] == "test") & (flows["label"] == 1)].head(500)
    largest = perturb(test, max(budget.bytes_factor), max(budget.packets_factor), max(budget.duration_factor))
    return float(np.median(np.linalg.norm(prep.transform(largest) - prep.transform(test), axis=1)))


def run(data_dir: Path, artifacts: Path, quick: bool = False, budget: FlowBudget | None = None) -> dict:
    set_seed(SEED)
    budget = budget or FlowBudget()
    forest = QUICK_FOREST if quick else ForestSettings()
    supervised = QUICK_SUPERVISED if quick else SupervisedSettings()
    flows = load(data_dir / "flows")
    test = flows[flows["split"] == "test"]
    normal_train = flows[(flows["split"] == "train") & (flows["label"] == 0)]
    ttl_values = normal_train["sttl"].value_counts().index[:NORMAL_TTL_VALUES].tolist()
    rows = 300 if quick else ATTACKED_ROWS

    train_rows = labeled_sample(flows, "train", 400_000, SEED)
    val_rows = labeled_sample(flows, "validation", 200_000, SEED)
    fit_rows = fit_sample(flows, 0.01, 20_000 if quick else FIT_ROWS, SEED)
    prep = FlowPreprocessor().fit(flows.iloc[fit_rows])
    no_ttl_prep = FlowPreprocessor(numeric=[c for c in NUMERIC if c not in TTL_FEATURES]).fit(
        flows.iloc[fit_rows]
    )
    train_attacks = flows.iloc[train_rows]
    train_attacks = train_attacks[train_attacks["label"] == 1]
    models = {
        "lightgbm": calibrate(fit_supervised(flows, prep, supervised, SEED, train_rows, val_rows), flows),
        "lightgbm_no_ttl": calibrate(
            fit_supervised(flows, no_ttl_prep, supervised, SEED, train_rows, val_rows), flows
        ),
        "lightgbm_adversarial": calibrate(
            fit_supervised(
                flows,
                prep,
                supervised,
                SEED,
                train_rows,
                val_rows,
                extra=random_changes(train_attacks, budget, ttl_values, SEED),
            ),
            flows,
        ),
        "iforest": calibrate(
            Model("iforest", prep, Forest(forest, SEED).fit(prep.transform(flows.iloc[fit_rows])), 0.0), flows
        ),
    }
    summary = {"ttl_values": [int(v) for v in ttl_values], "models": {}}
    for name, model in models.items():
        result = {
            "clean": clean_metrics(model, test),
            "evasion": evasion_rows(model, test, budget, ttl_values, rows),
        }
        summary["models"][name] = result
        print(
            f"flows {name}: recall {result['clean']['recall']:.3f}, evasion (all) "
            f"{result['evasion']['groups']['all']['evasion_rate']:.3f}"
        )
    summary["hop_skip_jump"] = {
        "lightgbm": hop_skip_jump(models["lightgbm"], test, 40 if quick else ART_ROWS, quick)
    }
    summary["poisoning"] = poisoning(flows, test, 0.01, forest, quick)
    summary["smoothing"] = smoothing(flows, supervised, SmoothingSettings(), quick)
    summary["largest_change_l2"] = change_size(flows, budget)
    out = artifacts / "flows"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--quick", action="store_true", help="small models, for tests and the smoke run")
    args = parse_args(parser, __file__)
    run(args.data_dir, args.artifacts, args.quick)


if __name__ == "__main__":
    main()
