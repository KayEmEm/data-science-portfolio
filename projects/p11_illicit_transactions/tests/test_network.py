import networkx as nx
import numpy as np
import pandas as pd

from illicit_graph.network import (
    STRUCTURAL,
    all_metrics,
    centrality_comparison,
    homophily,
    mostly_illicit_share,
    node_metrics,
    shuffled_within_step,
)


def test_node_metrics_on_a_star_and_a_pair():
    graph = nx.DiGraph([(1, 0), (2, 0), (3, 0), (4, 5)])
    frame, stats = node_metrics(graph, samples=10, seed=0)
    frame = frame.set_index("row")
    assert frame.loc[0, "in_degree"] == 3
    assert frame.loc[0, "component_size"] == 4
    assert frame.loc[5, "component_size"] == 2
    assert np.isclose(frame["pagerank"].mean(), 1.0)
    assert frame.loc[0, "betweenness"] > frame.loc[1, "betweenness"]
    assert stats["components"] == 2
    assert np.isclose(stats["largest_component_share"], 4 / 6)


def test_all_metrics_cover_every_node_once(prepared):
    nodes, edges, _ = prepared
    metrics, stats = all_metrics(nodes, edges, samples=8, seed=0)
    assert (metrics["row"].to_numpy() == np.arange(len(nodes))).all()
    assert set(STRUCTURAL) <= set(metrics.columns)
    assert len(stats) == nodes["time_step"].nunique()
    # community ids do not repeat across time steps
    per_community = (
        pd.Series(nodes["time_step"].to_numpy()).groupby(metrics["community"].to_numpy()).nunique()
    )
    assert (per_community == 1).all()
    table = centrality_comparison(metrics, nodes["label"].to_numpy())
    assert table["roc_auc_alone"].between(0, 1).all()


def test_homophily_and_communities():
    labels = np.array([1, 1, 0, 0, -1, 1])
    src, dst = np.array([0, 1, 2, 5]), np.array([1, 2, 3, 4])
    # edges with an illicit end and both ends labelled: 0-1 (both illicit) and 1-2
    assert homophily(labels, src, dst) == 0.5
    community = np.array([0, 0, 0, 0, 0, 1])
    assert mostly_illicit_share(labels, community) == 2 / 3


def test_shuffle_keeps_counts_per_step():
    labels = np.array([1, 0, 0, 1, 1, -1, 0])
    steps = np.array([1, 1, 1, 2, 2, 2, 2])
    out = shuffled_within_step(labels, steps, np.random.default_rng(0))
    for step in (1, 2):
        assert sorted(out[steps == step]) == sorted(labels[steps == step])
