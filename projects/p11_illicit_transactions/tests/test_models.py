import numpy as np
from sklearn.metrics import average_precision_score

from illicit_graph import gnn
from illicit_graph.config import ILLICIT, QUICK_GNN
from illicit_graph.network import all_metrics
from illicit_graph.train import feature_matrix, masks


def test_feature_sets(prepared):
    nodes, edges, _ = prepared
    metrics, _ = all_metrics(nodes, edges, samples=4, seed=0)
    assert feature_matrix(nodes, None, "local")[0].shape[1] == 93
    assert feature_matrix(nodes, None, "all")[0].shape[1] == 165
    x, columns = feature_matrix(nodes, metrics, "all_graph")
    assert x.shape == (len(nodes), 172)
    assert columns[-1] == "component_size"


def test_masks_hold_only_labelled_transactions_of_their_steps(prepared):
    nodes = prepared[0]
    m = masks(nodes)
    labels = nodes["label"].to_numpy()
    for part in ("train", "validation", "test"):
        assert (labels[m[part]] != -1).all()
    assert not (m["train"] & m["test"]).any()
    assert m["train_steps"].sum() > m["train"].sum()


def test_graph_networks_learn_the_synthetic_signal(prepared):
    nodes, edges, _ = prepared
    m = masks(nodes)
    x, _ = feature_matrix(nodes, None, "all")
    y = (nodes["label"].to_numpy() == ILLICIT).astype("int64")
    for kind in gnn.KINDS:
        prob, history = gnn.fit_predict(
            kind,
            x,
            edges[["src", "dst"]].to_numpy(),
            y,
            m["train"],
            m["validation"],
            m["train_steps"],
            m["validation_steps"],
            QUICK_GNN,
            0,
        )
        assert prob.shape == (len(nodes),)
        assert np.all((prob >= 0) & (prob <= 1))
        assert 1 <= len(history) <= QUICK_GNN.epochs
        assert average_precision_score(y[m["test"]], prob[m["test"]]) > 3 * y[m["test"]].mean()
