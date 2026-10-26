"""Network statistics, centrality and communities of the transaction graphs, with NetworkX.

Each time step is its own graph (no edge joins two steps), so every measure is computed per step
and uses no labels. Outputs under ``artifacts/network/``:

* ``node_metrics.parquet``: per transaction, in and out degree, PageRank, betweenness (estimated
  from ``BETWEENNESS_SAMPLES`` sources), clustering coefficient, core number, size of its weakly
  connected component and its Louvain community;
* ``step_stats.csv``: per time step, nodes, edges, density, components, the share of the largest
  one, degree assortativity, mean clustering and the number of communities with their modularity;
* ``centrality.csv``: per measure, median for illicit and licit transactions and the ROC AUC of
  the measure alone as an illicit score (labelled transactions of all steps);
* ``communities.csv``: per community, size, labelled and illicit transactions;
* ``concentration.json``: how strongly illicit transactions cluster: label homophily of edges and
  the share of illicit transactions in mostly illicit communities, each against labels shuffled
  within the time step.

The structural measures (``STRUCTURAL``) are also model features, in ``train.py``.
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from portfolio_common.config import parse_args
from sklearn.metrics import roc_auc_score

from illicit_graph.config import (
    ARTIFACTS_DIR,
    BETWEENNESS_SAMPLES,
    DATA_DIR,
    ILLICIT,
    LICIT,
    LOUVAIN_RESOLUTION,
    MIN_COMMUNITY,
    SEED,
    UNKNOWN,
)
from illicit_graph.data import load

STRUCTURAL = [
    "in_degree",
    "out_degree",
    "pagerank",
    "betweenness",
    "clustering",
    "core_number",
    "component_size",
]
SHUFFLES = 20


def step_graph(nodes: pd.DataFrame, edges: pd.DataFrame, step: int) -> nx.DiGraph:
    """Directed graph of one time step; nodes are row positions of the node table."""
    members = np.flatnonzero(nodes["time_step"].to_numpy() == step)
    graph = nx.DiGraph()
    graph.add_nodes_from(members.tolist())
    inside = edges["src"].isin(members).to_numpy()
    graph.add_edges_from(
        zip(edges["src"].to_numpy()[inside].tolist(), edges["dst"].to_numpy()[inside].tolist(), strict=True)
    )
    return graph


def node_metrics(graph: nx.DiGraph, samples: int, seed: int) -> tuple[pd.DataFrame, dict]:
    undirected = graph.to_undirected(as_view=False)
    order = list(graph.nodes)
    k = min(samples, len(order))
    betweenness = nx.betweenness_centrality(undirected, k=k, seed=seed, normalized=True)
    pagerank = nx.pagerank(graph, alpha=0.85)
    clustering = nx.clustering(undirected)
    core = nx.core_number(undirected)
    component = {}
    components = list(nx.weakly_connected_components(graph))
    for members in components:
        for node in members:
            component[node] = len(members)
    communities = nx.community.louvain_communities(undirected, resolution=LOUVAIN_RESOLUTION, seed=seed)
    community = {}
    for index, members in enumerate(communities):
        for node in members:
            community[node] = index
    frame = pd.DataFrame(
        {
            "row": order,
            "in_degree": [graph.in_degree(n) for n in order],
            "out_degree": [graph.out_degree(n) for n in order],
            "pagerank": [pagerank[n] * len(order) for n in order],  # 1 is the mean in every step
            "betweenness": [betweenness[n] for n in order],
            "clustering": [clustering[n] for n in order],
            "core_number": [core[n] for n in order],
            "component_size": [component[n] for n in order],
            "community_local": [community[n] for n in order],
        }
    )
    largest = max((len(c) for c in components), default=0)
    n_edges = graph.number_of_edges()
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": n_edges,
        "density": nx.density(graph),
        "components": len(components),
        "largest_component_share": largest / max(graph.number_of_nodes(), 1),
        "degree_assortativity": nx.degree_assortativity_coefficient(undirected) if n_edges > 1 else np.nan,
        "mean_clustering": float(np.mean(list(clustering.values()))) if clustering else np.nan,
        "communities": len(communities),
        "modularity": nx.community.modularity(undirected, communities) if n_edges else np.nan,
    }
    return frame, stats


def all_metrics(
    nodes: pd.DataFrame, edges: pd.DataFrame, samples: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, stats = [], []
    offset = 0
    for step in sorted(nodes["time_step"].unique()):
        frame, row = node_metrics(step_graph(nodes, edges, int(step)), samples, seed)
        frame["community"] = frame["community_local"] + offset
        offset += int(frame["community_local"].max()) + 1
        frames.append(frame.drop(columns="community_local"))
        stats.append({"time_step": int(step), **row})
    metrics = pd.concat(frames, ignore_index=True).sort_values("row").reset_index(drop=True)
    return metrics, pd.DataFrame(stats)


def centrality_comparison(metrics: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    labelled = labels != UNKNOWN
    y = (labels[labelled] == ILLICIT).astype(int)
    rows = []
    for column in STRUCTURAL:
        values = metrics[column].to_numpy()
        rows.append(
            {
                "measure": column,
                "median_illicit": float(np.median(values[labels == ILLICIT])),
                "median_licit": float(np.median(values[labels == LICIT])),
                "mean_illicit": float(np.mean(values[labels == ILLICIT])),
                "mean_licit": float(np.mean(values[labels == LICIT])),
                "roc_auc_alone": float(roc_auc_score(y, values[labelled])),
            }
        )
    return pd.DataFrame(rows)


def community_table(metrics: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "community": metrics["community"].to_numpy(),
            "time_step": nodes["time_step"].to_numpy(),
            "labelled": (nodes["label"] != UNKNOWN).to_numpy(),
            "illicit": (nodes["label"] == ILLICIT).to_numpy(),
        }
    )
    table = frame.groupby("community").agg(
        time_step=("time_step", "first"),
        size=("labelled", "size"),
        labelled=("labelled", "sum"),
        illicit=("illicit", "sum"),
    )
    table["illicit_share_of_labelled"] = table["illicit"] / table["labelled"].where(table["labelled"] > 0)
    return table.reset_index()


def homophily(labels: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """Share of edges between two labelled transactions whose both ends are illicit, among those with
    at least one illicit end."""
    a, b = labels[src], labels[dst]
    both = (a != UNKNOWN) & (b != UNKNOWN)
    any_illicit = both & ((a == ILLICIT) | (b == ILLICIT))
    if not any_illicit.any():
        return float("nan")
    return float(((a == ILLICIT) & (b == ILLICIT))[any_illicit].mean())


def mostly_illicit_share(labels: np.ndarray, community: np.ndarray) -> float:
    """Share of illicit transactions that sit in a community of ``MIN_COMMUNITY`` or more nodes whose
    labelled members are at least half illicit."""
    frame = pd.DataFrame(
        {"community": community, "illicit": labels == ILLICIT, "labelled": labels != UNKNOWN}
    )
    groups = frame.groupby("community").agg(
        size=("illicit", "size"), illicit=("illicit", "sum"), labelled=("labelled", "sum")
    )
    mostly = groups.index[
        (groups["size"] >= MIN_COMMUNITY)
        & (groups["illicit"] >= 0.5 * groups["labelled"])
        & (groups["illicit"] > 0)
    ]
    total = frame["illicit"].sum()
    return (
        float(frame.loc[frame["community"].isin(mostly), "illicit"].sum() / total) if total else float("nan")
    )


def shuffled_within_step(labels: np.ndarray, steps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = labels.copy()
    for step in np.unique(steps):
        rows = np.flatnonzero(steps == step)
        out[rows] = labels[rows][rng.permutation(len(rows))]
    return out


def concentration(nodes: pd.DataFrame, edges: pd.DataFrame, metrics: pd.DataFrame, seed: int) -> dict:
    labels = nodes["label"].to_numpy()
    steps = nodes["time_step"].to_numpy()
    src, dst = edges["src"].to_numpy(), edges["dst"].to_numpy()
    community = metrics["community"].to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = [shuffled_within_step(labels, steps, rng) for _ in range(SHUFFLES)]
    observed_h, observed_c = homophily(labels, src, dst), mostly_illicit_share(labels, community)
    null_h = [homophily(s, src, dst) for s in shuffled]
    null_c = [mostly_illicit_share(s, community) for s in shuffled]
    return {
        "edge_homophily_illicit": observed_h,
        "edge_homophily_illicit_shuffled": float(np.mean(null_h)),
        "edge_homophily_illicit_shuffled_max": float(np.max(null_h)),
        "illicit_in_mostly_illicit_communities": observed_c,
        "illicit_in_mostly_illicit_communities_shuffled": float(np.mean(null_c)),
        "illicit_in_mostly_illicit_communities_shuffled_max": float(np.max(null_c)),
        "shuffles": SHUFFLES,
        "min_community": MIN_COMMUNITY,
    }


def load_metrics(artifacts: Path) -> pd.DataFrame:
    return pd.read_parquet(artifacts / "network" / "node_metrics.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--samples", type=int, default=BETWEENNESS_SAMPLES)
    parser.add_argument("--force", action="store_true", help="recompute stored measures")
    args = parse_args(parser, __file__)
    nodes, edges = load(args.data_dir)
    out = args.artifacts / "network"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "node_metrics.parquet").exists() and not args.force:
        metrics = load_metrics(args.artifacts)
        print("node measures: stored")
    else:
        metrics, stats = all_metrics(nodes, edges, args.samples, SEED)
        metrics.to_parquet(out / "node_metrics.parquet", index=False)
        stats.to_csv(out / "step_stats.csv", index=False)
    labels = nodes["label"].to_numpy()
    comparison = centrality_comparison(metrics, labels)
    comparison.to_csv(out / "centrality.csv", index=False)
    community_table(metrics, nodes).to_csv(out / "communities.csv", index=False)
    result = concentration(nodes, edges, metrics, SEED)
    (out / "concentration.json").write_text(json.dumps(result, indent=2))
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(
        f"illicit edge homophily {result['edge_homophily_illicit']:.3f} "
        f"(shuffled {result['edge_homophily_illicit_shuffled']:.3f}); illicit in mostly illicit "
        f"communities {result['illicit_in_mostly_illicit_communities']:.3f} "
        f"(shuffled {result['illicit_in_mostly_illicit_communities_shuffled']:.3f})"
    )


if __name__ == "__main__":
    main()
