"""Graph neural networks for node classification: GCN and GraphSAGE in PyTorch Geometric.

Edges never join two time steps, so the graph of the training steps, the graph of the validation
steps and the whole graph can be used as separate full batches: training runs on the graph of the
training steps (the loss uses its labelled nodes), early stopping on the graph of the validation
steps, and the final probabilities come from the whole graph. No message passes between periods.
Edges are used in both directions (a payment links both transactions). Features are standardized
with the mean and spread of the training steps. Training stops when validation PR AUC has not
improved for ``patience`` epochs, and the best epoch is kept.
"""

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph, to_undirected

from illicit_graph.config import GNNSettings

KINDS = {"gcn": GCNConv, "sage": SAGEConv}


class NodeClassifier(nn.Module):
    def __init__(self, kind: str, in_dim: int, settings: GNNSettings):
        super().__init__()
        conv = KINDS[kind]
        dims = [in_dim] + [settings.hidden] * settings.layers
        self.convs = nn.ModuleList(conv(a, b) for a, b in zip(dims[:-1], dims[1:], strict=True))
        self.dropout = settings.dropout
        self.head = nn.Linear(settings.hidden, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = nn.functional.dropout(x, self.dropout, self.training)
            x = torch.relu(conv(x, edge_index))
        return self.head(x)


def standardize(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mean = x[reference].mean(axis=0)
    std = x[reference].std(axis=0)
    return ((x - mean) / np.where(std > 0, std, 1.0)).astype("float32")


def fit_predict(
    kind: str,
    x: np.ndarray,
    edges: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    reference: np.ndarray,
    val_steps: np.ndarray,
    settings: GNNSettings,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Illicit probability for every node and the training history.

    ``y`` holds 1 for illicit and 0 for licit on the masked nodes; ``reference`` marks all nodes of the
    training steps (they set the standardization and form the training graph) and ``val_steps`` all
    nodes of the validation steps.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(settings.threads)
    features = torch.from_numpy(standardize(x, reference))
    edge_index = to_undirected(torch.from_numpy(edges.T.astype("int64")), num_nodes=len(x))
    labels = torch.from_numpy(np.where(train_mask | val_mask, y, 0).astype("int64"))
    steps_train = torch.from_numpy(np.flatnonzero(reference))
    steps_val = torch.from_numpy(np.flatnonzero(val_steps))
    train_edges, _ = subgraph(steps_train, edge_index, relabel_nodes=True, num_nodes=len(x))
    val_edges, _ = subgraph(steps_val, edge_index, relabel_nodes=True, num_nodes=len(x))
    x_train, x_val = features[steps_train], features[steps_val]
    # positions of the labelled nodes inside the two subgraphs
    train_rows = torch.from_numpy(np.flatnonzero(train_mask[reference]))
    y_train = labels[steps_train][train_rows]
    val_rows = np.flatnonzero(val_mask[val_steps])
    y_val = y[val_steps][val_rows]
    positive = float(y[train_mask].mean())
    weight = settings.illicit_weight or (1 - positive) / positive
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, weight]))
    model = NodeClassifier(kind, x.shape[1], settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    best, best_state, since, history = -1.0, None, 0, []
    for epoch in range(1, settings.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train, train_edges)
        loss = loss_fn(logits[train_rows], y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(x_val, val_edges), dim=1)[:, 1].numpy()
        val_ap = float(average_precision_score(y_val, prob[val_rows]))
        history.append({"epoch": epoch, "loss": float(loss), "validation_pr_auc": val_ap})
        if val_ap > best:
            best, best_state, since = val_ap, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= settings.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(features, edge_index), dim=1)[:, 1].numpy()
    return prob.astype("float64"), history
