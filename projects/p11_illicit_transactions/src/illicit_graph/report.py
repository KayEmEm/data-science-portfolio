"""Tables, figures and results.json under reports/ from the network and evaluation outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import networkx as nx
import numpy as np
import pandas as pd
from portfolio_common.config import parse_args
from portfolio_common.plotting import apply_style

from illicit_graph.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    ILLICIT,
    LICIT,
    REPORTS_DIR,
    SHUTDOWN_STEP,
    SPLIT_STEPS,
    SUBGRAPH_HOPS,
    UNKNOWN,
)
from illicit_graph.data import load
from illicit_graph.network import STRUCTURAL, load_metrics
from illicit_graph.train import BASELINE, MODELS

SELECTED = "#2a78d6"
OTHER = "#9dbbe0"
BASELINE_COLOR = "#898781"
REFERENCE_COLOR = "#eb6834"
GNN_COLOR = "#1baf7a"
INK = "#0b0b0b"
CLASS_COLORS = {ILLICIT: "#d1495b", LICIT: "#2a78d6", UNKNOWN: "#c9c7c1"}
SUBGRAPHS = 4


def color(model: str, selected: str) -> str:
    if model == selected:
        return SELECTED
    if model == BASELINE:
        return BASELINE_COLOR
    if model == "lightgbm_all":
        return REFERENCE_COLOR
    if MODELS[model]["kind"] in {"gcn", "sage"}:
        return GNN_COLOR
    return OTHER


def models_figure(summary: pd.DataFrame, selected: str, path: Path) -> None:
    order = summary.sort_values("test_f1")
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(order) + 1.4))
    value = order["test_f1"].to_numpy()
    ax.barh(y, value, color=[color(m, selected) for m in order["model"]], height=0.6)
    right = value
    if "test_f1_low" in order:
        right = order["test_f1_high"].to_numpy()
        low = value - order["test_f1_low"].to_numpy()
        # a percentile interval need not contain the estimate on all of the test steps
        xerr = [np.maximum(low, 0), np.maximum(right - value, 0)]
        ax.errorbar(value, y, xerr=xerr, fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        right = np.maximum(right, value)
    for yi, v, r in zip(y, value, right, strict=True):
        ax.text(r + 0.015, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, order["label"])
    ax.set_xlim(0, 1.1)
    ax.set_title("Illicit F1 on the test steps, threshold from validation", loc="left")
    fig.savefig(path)
    plt.close(fig)


def over_time_figure(by_step: pd.DataFrame, models: list[str], selected: str, path: Path) -> None:
    fig, (ax, bottom) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, height_ratios=[3, 1])
    for model in models:
        part = by_step.loc[by_step["model"] == model]
        width = 2.4 if model == selected else 1.4
        ax.plot(
            part["time_step"],
            part["f1"],
            marker="o",
            ms=3,
            lw=width,
            color=color(model, selected),
            label=MODELS[model]["label"],
        )
    for axis in (ax, bottom):
        axis.axvline(SHUTDOWN_STEP - 0.5, color=INK, lw=1, ls="--")
    ax.text(SHUTDOWN_STEP - 0.4, 1.02, "dark market shutdown", fontsize=8, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Illicit F1")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Illicit F1 per test time step", loc="left")
    counts = by_step.loc[by_step["model"] == selected]
    bottom.bar(counts["time_step"], counts["illicit"], color=CLASS_COLORS[ILLICIT])
    bottom.set_ylabel("Illicit")
    bottom.set_xlabel("Time step")
    fig.savefig(path)
    plt.close(fig)


def steps_figure(nodes: pd.DataFrame, path: Path) -> None:
    table = nodes.groupby(["time_step", "label"]).size().unstack("label", fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bottom = np.zeros(len(table))
    for code, name in ((ILLICIT, "illicit"), (LICIT, "licit"), (UNKNOWN, "unknown")):
        values = table.get(code, pd.Series(0, index=table.index)).to_numpy()
        ax.bar(table.index, values, bottom=bottom, color=CLASS_COLORS[code], label=name, width=0.8)
        bottom += values
    for name, steps in SPLIT_STEPS.items():
        ax.text((steps.start + steps.stop - 1) / 2, bottom.max() * 1.02, name, ha="center", fontsize=8)
        ax.axvline(steps.start - 0.5, color=INK, lw=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Transactions")
    ax.legend(fontsize=8, ncols=3, loc="upper right", bbox_to_anchor=(1, 0.95))
    ax.set_title("Transactions per time step and split", loc="left")
    fig.savefig(path)
    plt.close(fig)


def centrality_figure(
    metrics: pd.DataFrame, labels: np.ndarray, comparison: pd.DataFrame, path: Path
) -> None:
    shown = ["in_degree", "out_degree", "pagerank", "betweenness"]
    fig, axes = plt.subplots(1, len(shown), figsize=(11, 3.4))
    auc = comparison.set_index("measure")["roc_auc_alone"]
    for ax, column in zip(axes, shown, strict=True):
        for code, name in ((LICIT, "licit"), (ILLICIT, "illicit")):
            values = np.sort(metrics[column].to_numpy()[labels == code])
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=CLASS_COLORS[code], label=name)
        ax.set_xscale("symlog", linthresh=1e-4 if column == "betweenness" else 1)
        ax.set_title(f"{column}\nROC AUC alone {auc[column]:.2f}", loc="left", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Share of transactions at or below")
    axes[0].legend(fontsize=8)
    fig.suptitle("Centrality of labelled transactions (cumulative distribution)", x=0.02, y=1.06, ha="left")
    fig.savefig(path)
    plt.close(fig)


def communities_figure(communities: pd.DataFrame, concentration: dict, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), width_ratios=[3, 2])
    part = communities.loc[communities["labelled"] > 0]
    left.scatter(
        part["size"],
        part["illicit_share_of_labelled"],
        s=8 + 2 * part["illicit"],
        color=SELECTED,
        alpha=0.45,
        edgecolors="none",
    )
    left.set_xscale("log")
    left.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    left.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    left.set_xlabel("Community size (transactions)")
    left.set_ylabel("Illicit share of labelled")
    left.set_title("Louvain communities with labelled transactions", loc="left", fontsize=9)
    keys = [
        ("edge_homophily_illicit", "Edges with an illicit end:\nboth ends illicit"),
        ("illicit_in_mostly_illicit_communities", "Illicit transactions in\nmostly illicit communities"),
    ]
    y = np.arange(len(keys))
    right.barh(
        y + 0.18,
        [concentration[k] for k, _ in keys],
        height=0.34,
        color=CLASS_COLORS[ILLICIT],
        label="observed",
    )
    right.barh(
        y - 0.18,
        [concentration[f"{k}_shuffled"] for k, _ in keys],
        height=0.34,
        color=BASELINE_COLOR,
        label="labels shuffled within step",
    )
    for yi, (k, _) in zip(y, keys, strict=True):
        right.text(concentration[k] + 0.02, yi + 0.18, f"{concentration[k]:.2f}", va="center", fontsize=8)
        right.text(
            concentration[f"{k}_shuffled"] + 0.02,
            yi - 0.18,
            f"{concentration[f'{k}_shuffled']:.2f}",
            va="center",
            fontsize=8,
        )
    right.set_yticks(y, [name for _, name in keys], fontsize=8)
    right.set_xlim(0, 1.15)
    right.legend(fontsize=8, loc="lower right")
    right.set_title("Do illicit transactions cluster?", loc="left", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def subgraph_centers(nodes: pd.DataFrame, graph: nx.Graph, count: int) -> list[int]:
    """Illicit test transactions whose neighborhood holds the most illicit transactions, one per step."""
    labels = nodes["label"].to_numpy()
    steps = nodes["time_step"].to_numpy()
    candidates = np.flatnonzero((labels == ILLICIT) & (nodes["split"].astype(str).to_numpy() == "test"))
    scored = []
    for node in candidates:
        around = nx.single_source_shortest_path_length(graph, int(node), cutoff=SUBGRAPH_HOPS)
        if len(around) > 120:
            continue
        # neighborhoods of 8 or more transactions first, then by illicit transactions in them
        scored.append((len(around) >= 8, sum(labels[n] == ILLICIT for n in around), len(around), int(node)))
    scored.sort(reverse=True)
    chosen, used_steps = [], set()
    for *_, node in scored:
        if steps[node] not in used_steps:
            chosen.append(node)
            used_steps.add(steps[node])
        if len(chosen) == count:
            break
    return chosen


def subgraph_figure(
    nodes: pd.DataFrame, edges: pd.DataFrame, scores: np.ndarray, threshold: float, path: Path
) -> list[dict]:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(zip(edges["src"].tolist(), edges["dst"].tolist(), strict=True))
    labels = nodes["label"].to_numpy()
    centers = subgraph_centers(nodes, graph, SUBGRAPHS)
    fig, axes = plt.subplots(1, max(len(centers), 1), figsize=(3.6 * max(len(centers), 1), 3.8))
    axes = np.atleast_1d(axes)
    described = []
    for ax, center in zip(axes, centers, strict=False):
        members = nx.single_source_shortest_path_length(graph, center, cutoff=SUBGRAPH_HOPS)
        sub = graph.subgraph(members)
        layout = nx.spring_layout(sub, seed=7)
        flagged = [n for n in sub if scores[n] >= threshold]
        nx.draw_networkx_edges(sub, layout, ax=ax, edge_color="#b8b6b0", width=0.8)
        nx.draw_networkx_nodes(
            sub, layout, ax=ax, node_size=28, node_color=[CLASS_COLORS[labels[n]] for n in sub]
        )
        nx.draw_networkx_nodes(
            sub,
            layout,
            ax=ax,
            nodelist=flagged,
            node_size=70,
            node_color="none",
            edgecolors=INK,
            linewidths=1.0,
        )
        nx.draw_networkx_nodes(
            sub, layout, ax=ax, nodelist=[center], node_size=120, node_shape="*", node_color=INK
        )
        step = int(nodes["time_step"].iloc[center])
        ax.set_title(f"step {step}, {len(sub)} transactions", fontsize=9, loc="left")
        ax.set_axis_off()
        described.append(
            {
                "tx_id": int(nodes["tx_id"].iloc[center]),
                "time_step": step,
                "transactions": len(sub),
                "illicit": int((labels[list(sub)] == ILLICIT).sum()),
                "licit": int((labels[list(sub)] == LICIT).sum()),
                "flagged": len(flagged),
                "flagged_illicit": int(sum(labels[n] == ILLICIT for n in flagged)),
                "flagged_unknown": int(sum(labels[n] == UNKNOWN for n in flagged)),
            }
        )
    handles = [
        plt.Line2D([], [], ls="", marker="o", color=CLASS_COLORS[c], label=n)
        for c, n in ((ILLICIT, "illicit"), (LICIT, "licit"), (UNKNOWN, "unknown"))
    ]
    handles.append(plt.Line2D([], [], ls="", marker="o", mfc="none", mec=INK, label="flagged by the model"))
    fig.legend(handles=handles, loc="lower center", ncols=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"{SUBGRAPH_HOPS} hop neighborhoods of illicit test transactions (star: center)", x=0.02, ha="left"
    )
    fig.savefig(path)
    plt.close(fig)
    return described


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}f}"


def results_markdown(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    selection: dict,
    audit: dict,
    stats: pd.DataFrame,
    comparison: pd.DataFrame,
    concentration: dict,
    subgraphs: list[dict],
) -> str:
    lines = [
        "# Results",
        "",
        f"Data: {selection['data']}. Test: {selection['test_labelled']:,} labelled transactions of the test "
        f"steps, {selection['test_illicit']:,} of them illicit. Selected on validation F1: "
        f"**{selection['label']}** (threshold {selection['threshold']:.3f}).",
        "",
        "## Models",
        "",
        "| Model | Validation F1 | Test F1 (95% interval) | Precision | Recall | F1 at 0.5 | PR AUC | "
        f"F1 before step {SHUTDOWN_STEP} | F1 from step {SHUTDOWN_STEP} | F1 range over seeds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in summary.sort_values("test_f1", ascending=False).iterrows():
        interval = f" ({fmt(r['test_f1_low'])} to {fmt(r['test_f1_high'])})" if "test_f1_low" in r else ""
        seeds = (
            f"{fmt(r['test_f1_seed_min'])} to {fmt(r['test_f1_seed_max'])}"
            if "test_f1_seed_min" in r and not pd.isna(r["test_f1_seed_min"])
            else ""
        )
        lines.append(
            f"| {r['label']} | {fmt(r['validation_f1'])} | {fmt(r['test_f1'])}{interval} | "
            f"{fmt(r['test_precision'])} | {fmt(r['test_recall'])} | {fmt(r['test_f1_at_half'])} | "
            f"{fmt(r['test_pr_auc'])} | {fmt(r['test_f1_before_shutdown'])} | "
            f"{fmt(r['test_f1_from_shutdown'])} | {seeds} |"
        )
    lines += [
        "",
        "Intervals: 95 percent, from resampling the test time steps with replacement. Graph networks: mean "
        "probability over seeds; the last column gives the test F1 of each seed alone.",
        "",
        "## Paired differences to LightGBM on all features",
        "",
        "| Model | Test F1 difference | 95% interval |",
        "| --- | ---: | ---: |",
    ]
    for _, r in paired.iterrows():
        lines.append(
            f"| {MODELS[r['model']]['label']} | {r['f1_difference']:+.3f} | "
            f"{fmt(r['low'])} to {fmt(r['high'])} |"
        )
    lines += ["", "## Network", "", "| Measure | Value |", "| --- | ---: |"]
    lines += [
        f"| Transactions | {audit['counts_found']['nodes']:,} |",
        f"| Edges | {audit['counts_found']['edges']:,} |",
        f"| Counts match the documented ones | {audit['counts_match_documented']} |",
        f"| Edges across time steps | {audit['edges_across_time_steps']} |",
        f"| Labelled share | {audit['labelled_share']:.1%} |",
        f"| Illicit share of labelled | {audit['illicit_share_of_labelled']:.1%} |",
        f"| Isolated transactions | {audit['isolated_nodes']:,} |",
        f"| Mean density per step | {stats['density'].mean():.2e} |",
        f"| Mean share of the largest component per step | {stats['largest_component_share'].mean():.3f} |",
        f"| Mean degree assortativity per step | {stats['degree_assortativity'].mean():.3f} |",
        f"| Louvain communities (all steps) | {int(stats['communities'].sum()):,} |",
        f"| Mean modularity per step | {stats['modularity'].mean():.3f} |",
        "",
        "### Centrality, illicit against licit",
        "",
        "| Measure | Median illicit | Median licit | Mean illicit | Mean licit | ROC AUC alone |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in comparison.iterrows():
        lines.append(
            f"| {r['measure']} | {r['median_illicit']:.4g} | {r['median_licit']:.4g} | "
            f"{r['mean_illicit']:.4g} | {r['mean_licit']:.4g} | {r['roc_auc_alone']:.3f} |"
        )
    c = concentration
    lines += [
        "",
        "ROC AUC alone: the measure used as the only illicit score, over labelled transactions of all steps; "
        "0.5 is no signal, values below 0.5 mean that illicit transactions have lower values.",
        "",
        "### Clustering of illicit transactions",
        "",
        f"| Measure | Observed | Labels shuffled within step (mean, max of {c['shuffles']}) |",
        "| --- | ---: | ---: |",
        f"| Edges with an illicit end whose other end is illicit too | {c['edge_homophily_illicit']:.3f} | "
        f"{c['edge_homophily_illicit_shuffled']:.3f}, {c['edge_homophily_illicit_shuffled_max']:.3f} |",
        f"| Illicit transactions in communities (at least {c['min_community']} transactions) whose labelled "
        f"members are at least half illicit | {c['illicit_in_mostly_illicit_communities']:.3f} | "
        f"{c['illicit_in_mostly_illicit_communities_shuffled']:.3f}, "
        f"{c['illicit_in_mostly_illicit_communities_shuffled_max']:.3f} |",
        "",
        "### Subgraphs drawn",
        "",
        "| Center tx_id | Step | Transactions | Illicit | Licit | Flagged | Flagged illicit "
        "| Flagged unlabelled |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in subgraphs:
        lines.append(
            f"| {s['tx_id']} | {s['time_step']} | {s['transactions']} | {s['illicit']} | {s['licit']} | "
            f"{s['flagged']} | {s['flagged_illicit']} | {s['flagged_unknown']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parse_args(parser, __file__)
    apply_style()
    nodes, edges = load(args.data_dir)
    metrics = load_metrics(args.artifacts)
    network = args.artifacts / "network"
    evaluation = args.artifacts / "evaluation"
    stats = pd.read_csv(network / "step_stats.csv")
    comparison = pd.read_csv(network / "centrality.csv")
    communities = pd.read_csv(network / "communities.csv")
    concentration = json.loads((network / "concentration.json").read_text())
    summary = pd.read_csv(evaluation / "summary.csv")
    by_step = pd.read_csv(evaluation / "by_step.csv")
    paired = pd.read_csv(evaluation / "paired.csv")
    selection = json.loads((evaluation / "selection.json").read_text())
    audit = json.loads((args.data_dir / "audit.json").read_text())
    figures = args.reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    selected = selection["model"]
    labels = nodes["label"].to_numpy()

    models_figure(summary, selected, figures / "models.png")
    gnn = summary.loc[summary["model"].map(lambda m: MODELS[m]["kind"] in {"gcn", "sage"})]
    best_gnn = gnn.sort_values("validation_f1", ascending=False)["model"].iloc[0] if len(gnn) else None
    shown = list(
        dict.fromkeys(m for m in [selected, "lightgbm_all", best_gnn, BASELINE] if m in set(summary["model"]))
    )
    over_time_figure(by_step, shown, selected, figures / "f1_over_time.png")
    steps_figure(nodes, figures / "time_steps.png")
    centrality_figure(metrics, labels, comparison, figures / "centrality.png")
    communities_figure(communities, concentration, figures / "communities.png")
    scores = np.load(args.artifacts / "models" / selected / "scores.npy")
    subgraphs = subgraph_figure(nodes, edges, scores, selection["threshold"], figures / "subgraphs.png")

    text = results_markdown(summary, paired, selection, audit, stats, comparison, concentration, subgraphs)
    (args.reports / "results.md").write_text(text)
    columns = [c for c in summary.columns if c != "label"]
    results = {
        "data": selection["data"],
        "selection": selection,
        "models": summary[columns].replace({np.nan: None}).to_dict(orient="records"),
        "paired": paired.replace({np.nan: None}).to_dict(orient="records"),
        "audit": {
            k: audit[k] for k in ("counts_found", "counts_documented", "counts_match_documented", "by_split")
        },
        "centrality": comparison.to_dict(orient="records"),
        "concentration": concentration,
        "structural_features": STRUCTURAL,
    }
    (args.reports / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.reports / 'results.md'} and {len(list(figures.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
