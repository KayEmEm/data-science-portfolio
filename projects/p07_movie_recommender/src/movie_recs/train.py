"""Fit the retrieval models for the validation and the test stage and store their candidate lists.

Validation stage: the ALS settings and the number of two tower epochs are
chosen by recall at 10 on the comparison half of the validation users.
Test stage: the chosen settings are refit on the history up to the test
window. Every finished step is stored under ``artifacts/<stage>/`` and is
skipped when the command runs again, so a stopped run resumes.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from movie_recs import als, baselines, two_tower
from movie_recs.config import (
    ALS_GRID,
    ARTIFACTS_DIR,
    DATA_DIR,
    N_CANDIDATES,
    QUICK_ALS_GRID,
    QUICK_TWO_TOWER,
    RECENT_DAYS,
    SEED,
    ALSSettings,
    K,
    TwoTowerSettings,
)
from movie_recs.data import load
from movie_recs.interactions import STAGES, Stage, make_stage, split_users
from movie_recs.metrics import per_user

SELECTION_SAMPLE = 5_000  # comparison users scored after each two tower epoch


def stage_dir(artifacts: Path, stage: str) -> Path:
    return artifacts / stage


def save_lists(path: Path, users: np.ndarray, items: np.ndarray, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, users=users, items=items, scores=scores)


def load_lists(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as stored:
        return stored["users"], stored["items"], stored["scores"]


def recall_at_k(stage: Stage, users: np.ndarray, items: np.ndarray) -> float:
    return float(per_user(users, items, stage.targets, K)["recall"].mean())


class Timer:
    def __init__(self, path: Path):
        self.path = path
        self.times = json.loads(path.read_text()) if path.exists() else {}

    def record(self, name: str, seconds: float) -> None:
        self.times[name] = round(seconds, 1)
        self.path.write_text(json.dumps(self.times, indent=2))


def run_stage(name: str, data: dict, artifacts: Path, quick: bool, force: bool) -> None:
    out = stage_dir(artifacts, name)
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer(out / "timings.json")
    stage = make_stage(data["ratings"], data["split"], name)
    users = stage.eval_users()
    np.save(out / "users.npy", users)
    compare = split_users(users, SEED)[1] if name == "validation" else users
    compare_rows = np.searchsorted(users, compare)
    print(
        f"{name}: {len(stage.history):,} history ratings, {len(stage.items):,} catalog movies, "
        f"{len(users):,} users with a liked movie in the window "
        f"({len(stage.new_users()):,} more without history)"
    )
    selection_path = artifacts / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}

    for model, days in (("popularity", None), ("popularity_recent", RECENT_DAYS)):
        path = out / "lists" / f"{model}.npz"
        if force or not path.exists():
            started = time.perf_counter()
            save_lists(path, users, *baselines.recommend(stage, users, N_CANDIDATES, days))
            timer.record(model, time.perf_counter() - started)

    path = out / "lists" / "als.npz"
    if force or not path.exists():
        if name == "validation":
            grid, rows = (QUICK_ALS_GRID if quick else ALS_GRID), []
            best = None
            for settings in grid:
                started = time.perf_counter()
                model = als.fit(stage, settings, SEED)
                items, scores = als.recommend(model, stage, users, N_CANDIDATES)
                seconds = time.perf_counter() - started
                recall = recall_at_k(stage, compare, items[compare_rows])
                rows.append({**asdict(settings), f"recall@{K}": recall, "seconds": round(seconds, 1)})
                print(f"ALS {settings}: recall@{K} {recall:.4f} in {seconds:.0f} s")
                if best is None or recall > best[0]:
                    best = (recall, settings, items, scores, seconds)
            pd.DataFrame(rows).to_csv(out / "als_grid.csv", index=False)
            selection["als"] = asdict(best[1])
            selection_path.write_text(json.dumps(selection, indent=2))
            _, _, items, scores, seconds = best
        else:
            started = time.perf_counter()
            model = als.fit(stage, ALSSettings(**selection["als"]), SEED)
            items, scores = als.recommend(model, stage, users, N_CANDIDATES)
            seconds = time.perf_counter() - started
        save_lists(path, users, items, scores)
        timer.record("als", seconds)

    path = out / "lists" / "two_tower.npz"
    if force or not path.exists():
        settings = QUICK_TWO_TOWER if quick else TwoTowerSettings()
        new_items = np.setdiff1d(stage.targets["item"].unique(), stage.items)
        features = two_tower.ItemFeatures(
            stage, data["movies"], data["tags"], settings.n_tags, extra=new_items
        )
        directory = out / "two_tower"
        started = time.perf_counter()
        if name == "validation":
            sample = compare[np.random.default_rng(SEED).permutation(len(compare))[:SELECTION_SAMPLE]]

            def on_epoch(model, epoch):
                items, _ = two_tower.recommend(model, stage, sample, K, settings)
                return {f"recall@{K}": recall_at_k(stage, sample, items)}

            model = two_tower.train(stage, features, settings, SEED, settings.epochs, directory, on_epoch)
            records = two_tower.epoch_records(directory)
            best = max(records, key=lambda r: r[f"recall@{K}"])
            pd.DataFrame(records).to_csv(out / "two_tower_epochs.csv", index=False)
            selection["two_tower"] = {**asdict(settings), "epochs": best["epoch"]}
            selection_path.write_text(json.dumps(selection, indent=2))
            state = torch.load(two_tower.checkpoint_path(directory, best["epoch"]), weights_only=False)
            model.load_state_dict(state["model"])
        else:
            chosen = dict(selection["two_tower"])
            epochs = chosen.pop("epochs")
            settings = TwoTowerSettings(**chosen)
            model = two_tower.train(stage, features, settings, SEED, epochs, directory)
        items, scores = two_tower.recommend(model, stage, users, N_CANDIDATES, settings)
        save_lists(path, users, items, scores)
        timer.record("two_tower", time.perf_counter() - started)
        save_new_item_lists(stage, model, features, settings, out)


def save_new_item_lists(stage: Stage, model, features, settings: TwoTowerSettings, out: Path) -> None:
    """Cold start items: rank only the movies first rated in the window, for users who liked one of them.

    The two tower model places these movies by their features alone (unknown id). The
    comparison is a random order of the same movies.
    """
    new = features.extra
    if len(new) == 0:
        return
    liked_new = stage.targets.loc[stage.targets["item"].isin(new)]
    users = np.unique(liked_new["user"].to_numpy())
    users = users[np.isin(users, stage.users)]
    if len(users) == 0:
        return
    user_vectors, item_vectors = two_tower.embed(model, stage, users, settings, extra=True)
    scores = user_vectors @ item_vectors.T
    n = min(K, len(new))
    best = np.argsort(-scores, axis=1, kind="stable")[:, :n]
    save_lists(
        out / "lists" / "two_tower_new_items.npz", users, new[best], np.take_along_axis(scores, best, 1)
    )
    rng = np.random.default_rng(SEED)
    shuffled = np.stack([rng.permutation(len(new))[:n] for _ in users])
    save_lists(
        out / "lists" / "random_new_items.npz", users, new[shuffled], np.zeros(shuffled.shape, np.float32)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--quick", action="store_true", help="small settings for the smoke run")
    parser.add_argument("--force", action="store_true", help="refit steps that are already stored")
    args = parser.parse_args()
    data = load(args.data_dir)
    for name in args.stages:
        run_stage(name, data, args.artifacts, args.quick, args.force)


if __name__ == "__main__":
    main()
