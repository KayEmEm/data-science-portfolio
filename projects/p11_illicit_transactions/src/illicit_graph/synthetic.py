"""Synthetic transaction graphs in the Elliptic layout, for the tests and ``make smoke``.

Their numbers say nothing about the Elliptic dataset. Each time step is a separate graph. Licit
transactions pay earlier transactions of the same step, mostly the ones that already receive many
payments, so a few hubs form. Illicit transactions form chains (each one passes funds to the next)
that touch the licit graph at a few points. Local features are noise; illicit transactions are
shifted on a few of them, and on other ones from ``SHUTDOWN_STEP`` on, so that a model trained on
early steps misses part of the later illicit pattern. The aggregated features are means of the
neighbors' local features, as in the release. About 23 percent of the transactions carry a label.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from illicit_graph.config import (
    CLASSES_FILE,
    EDGES_FILE,
    FEATURES_FILE,
    N_AGGREGATED,
    N_LOCAL,
    SEED,
    SHUTDOWN_STEP,
    TIME_STEPS,
)

ILLICIT_SHARE = 0.025  # of all transactions
LABELLED_SHARE = 0.23
EARLY_SHIFT = (slice(0, 8), 0.9)  # features shifted for illicit transactions before the shutdown
LATE_SHIFT = (slice(20, 28), 0.9)  # and from the shutdown on, with half of the early shift


def step_graph(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Edges of one time step and a 0/1 illicit flag per transaction."""
    n_illicit = max(3, int(round(ILLICIT_SHARE * n)))
    illicit = np.zeros(n, dtype=np.int8)
    illicit[rng.choice(n, n_illicit, replace=False)] = 1
    licit_nodes = np.flatnonzero(illicit == 0)
    edges = []
    received = np.ones(n)
    for i, node in enumerate(licit_nodes[1:], start=1):
        earlier = licit_nodes[:i]
        k = min(len(earlier), rng.choice([1, 1, 2, 3]))
        weights = received[earlier] / received[earlier].sum()
        for target in rng.choice(earlier, k, replace=False, p=weights):
            edges.append((node, target))
            received[target] += 1
    chain_nodes = rng.permutation(np.flatnonzero(illicit == 1))
    start = 0
    while start < len(chain_nodes):
        length = int(rng.integers(3, 9))
        chain = chain_nodes[start : start + length]
        start += length
        edges.extend(zip(chain[:-1], chain[1:], strict=True))
        # the chain is funded by one licit transaction and pays out to one or two others
        edges.append((rng.choice(licit_nodes), chain[0]))
        for target in rng.choice(licit_nodes, int(rng.integers(1, 3)), replace=False):
            edges.append((chain[-1], target))
    return np.array(edges, dtype=np.int64).reshape(-1, 2), illicit


def make_graph(n_nodes: int, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Feature, class and edge tables in the release layout, with ``n_nodes`` transactions in all."""
    rng = np.random.default_rng(seed)
    per_step = rng.multinomial(n_nodes - 20 * TIME_STEPS, np.ones(TIME_STEPS) / TIME_STEPS) + 20
    ids = rng.choice(np.arange(1_000_000, 400_000_000), n_nodes, replace=False)
    feature_rows, class_rows, edge_rows = [], [], []
    offset = 0
    for step, n in enumerate(per_step, start=1):
        edges, illicit = step_graph(rng, int(n))
        local = rng.normal(0.0, 1.0, (n, N_LOCAL))
        early, late = EARLY_SHIFT, LATE_SHIFT
        if step < SHUTDOWN_STEP:
            local[illicit == 1, early[0]] += early[1]
        else:
            local[illicit == 1, early[0]] += early[1] / 2
            local[illicit == 1, late[0]] += late[1]
        degree = np.bincount(edges.ravel(), minlength=n)
        local[:, 0] = np.log1p(degree) + rng.normal(0.0, 0.1, n)
        neighbor_sum = np.zeros((n, N_AGGREGATED))
        both = np.concatenate([edges, edges[:, ::-1]])
        np.add.at(neighbor_sum, both[:, 0], local[both[:, 1], :N_AGGREGATED])
        aggregated = neighbor_sum / np.maximum(degree, 1)[:, None]
        node_ids = ids[offset : offset + n]
        feature_rows.append(pd.DataFrame(np.column_stack([node_ids, np.full(n, step), local, aggregated])))
        labelled = rng.random(n) < LABELLED_SHARE
        labelled |= (illicit == 1) & (rng.random(n) < 0.4)
        codes = np.where(~labelled, "unknown", np.where(illicit == 1, "1", "2"))
        class_rows.append(pd.DataFrame({"txId": node_ids, "class": codes}))
        edge_rows.append(pd.DataFrame({"txId1": node_ids[edges[:, 0]], "txId2": node_ids[edges[:, 1]]}))
        offset += n
    features = pd.concat(feature_rows, ignore_index=True)
    features[0] = features[0].astype("int64")
    features[1] = features[1].astype("int64")
    return features, pd.concat(class_rows, ignore_index=True), pd.concat(edge_rows, ignore_index=True)


def write_release(folder: Path, n_nodes: int, seed: int = SEED) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    features, classes, edges = make_graph(n_nodes, seed)
    features.to_csv(folder / FEATURES_FILE, header=False, index=False, float_format="%.6g")
    classes.to_csv(folder / CLASSES_FILE, index=False)
    edges.to_csv(folder / EDGES_FILE, index=False)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--nodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parse_args(parser, __file__)
    write_release(args.folder, args.nodes, args.seed)
    print(f"wrote {args.nodes:,} synthetic transactions to {args.folder}")


if __name__ == "__main__":
    main()
