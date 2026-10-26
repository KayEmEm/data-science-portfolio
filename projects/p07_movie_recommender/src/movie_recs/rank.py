"""Train the ranking model on the validation candidates and apply it to both stages.

* Validation: fitted on half A of the users, scored on half B (the half that compares every model).
* Test: refitted on all validation users, applied to the test candidates.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from movie_recs import ranker
from movie_recs.config import ARTIFACTS_DIR, DATA_DIR, QUICK_RANKER, SEED, K, RankerSettings
from movie_recs.data import load
from movie_recs.interactions import make_stage, split_users
from movie_recs.train import Timer, load_lists, save_lists, stage_dir


def stage_frame(name: str, data: dict, artifacts: Path):
    stage = make_stage(data["ratings"], data["split"], name)
    lists = stage_dir(artifacts, name) / "lists"
    users, als_items, als_scores = load_lists(lists / "als.npz")
    tt_users, tt_items, tt_scores = load_lists(lists / "two_tower.npz")
    if not np.array_equal(users, tt_users):
        raise ValueError(f"{name}: ALS and two tower lists cover different users")
    frame = ranker.candidates(users, (als_items, als_scores), (tt_items, tt_scores))
    frame = ranker.features(stage, data["movies"], frame)
    return stage, users, frame, ranker.labels(frame, stage.targets)


def trainable(frame: pd.DataFrame, y: np.ndarray, users: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows of the given users whose candidate list holds at least one liked movie."""
    mask = frame["user"].isin(users).to_numpy()
    positives = pd.Series(y[mask]).groupby(frame.loc[mask, "user"].to_numpy()).transform("sum").to_numpy()
    keep = np.flatnonzero(mask)[positives > 0]
    return frame.iloc[keep], y[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--quick", action="store_true")
    args = parse_args(parser, __file__)
    settings = QUICK_RANKER if args.quick else RankerSettings()
    data = load(args.data_dir)

    started = time.perf_counter()
    _, users, frame, y = stage_frame("validation", data, args.artifacts)
    half_a, half_b = split_users(users, SEED)
    model = ranker.fit(*trainable(frame, y, half_a), settings, SEED)
    in_b = frame["user"].isin(half_b).to_numpy()
    items, scores = ranker.rank(model, frame.loc[in_b], half_b, K)
    out = stage_dir(args.artifacts, "validation")
    save_lists(out / "lists" / "ranker.npz", half_b, items, scores)
    recall_in_candidates = float(
        pd.Series(y[in_b]).groupby(frame.loc[in_b, "user"].to_numpy()).sum().gt(0).mean()
    )
    Timer(out / "timings.json").record("ranker", time.perf_counter() - started)

    started = time.perf_counter()
    model = ranker.fit(*trainable(frame, y, users), settings, SEED)
    importance = pd.DataFrame({"feature": ranker.FEATURES, "gain": model.booster_.feature_importance("gain")})
    importance["share"] = importance["gain"] / importance["gain"].sum()
    importance.sort_values("share", ascending=False).to_csv(
        args.artifacts / "ranker_importance.csv", index=False
    )
    _, test_users, test_frame, _ = stage_frame("test", data, args.artifacts)
    items, scores = ranker.rank(model, test_frame, test_users, K)
    out = stage_dir(args.artifacts, "test")
    save_lists(out / "lists" / "ranker.npz", test_users, items, scores)
    Timer(out / "timings.json").record("ranker", time.perf_counter() - started)

    info = {
        "settings": asdict(settings),
        "validation_rows": int(len(frame)),
        "candidates_per_user": float(len(frame) / len(users)),
        "share_of_comparison_users_with_a_liked_candidate": recall_in_candidates,
    }
    (args.artifacts / "ranker.json").write_text(json.dumps(info, indent=2))
    print(f"ranker: {json.dumps(info)}")


if __name__ == "__main__":
    main()
