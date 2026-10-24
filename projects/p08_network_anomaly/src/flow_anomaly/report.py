"""Tables, figures, an alert triage example and results.json under reports/ from the evaluation outputs."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.plotting import apply_style

from flow_anomaly import detectors
from flow_anomaly.config import (
    ALERT_BUDGETS,
    ARTIFACTS_DIR,
    CONTAMINATION_SHARES,
    DATA_DIR,
    NORMAL,
    REPORTS_DIR,
    TARGET_FPR,
    TRAIN_ATTACK_SHARE,
)
from flow_anomaly.data import load
from flow_anomaly.features import signed_log
from flow_anomaly.train import contamination_name

# The selected detector in the accent blue, the other unsupervised detectors in a lighter step of the
# same hue, the baseline in gray and the supervised reference in orange.
SELECTED = "#2a78d6"
OTHER = "#9dbbe0"
BASELINE = "#898781"
REFERENCE = "#eb6834"
INK = "#0b0b0b"
SERIES = {"iforest": "#2a78d6", "ecod": "#1baf7a", "autoencoder": "#eb6834", "lightgbm": "#898781"}
TRIAGE_ALERTS = 20


def label(detector: str) -> str:
    return detectors.SPECS[detector].label


def color(detector: str, selected: str) -> str:
    if detector == selected:
        return SELECTED
    if detector == "lightgbm":
        return REFERENCE
    if detector == "robust_z":
        return BASELINE
    return OTHER


def metrics_figure(main: pd.DataFrame, selected: str, path: Path) -> None:
    order = main.sort_values("test_pr_auc")
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(order) + 1.6), sharey=True)
    y = np.arange(len(order))
    colors = [color(d, selected) for d in order["detector"]]
    panels = (("pr_auc", "PR AUC"), ("recall", f"Recall at {TARGET_FPR:.0%} false alarm rate"))
    for ax, (metric, title) in zip(axes, panels, strict=True):
        value = order[f"test_{metric}"].to_numpy()
        ax.barh(y, value, color=colors, height=0.6)
        right = value
        if f"test_{metric}_low" in order:
            low = value - order[f"test_{metric}_low"].to_numpy()
            right = order[f"test_{metric}_high"].to_numpy()
            ax.errorbar(value, y, xerr=[low, right - value], fmt="none", ecolor=INK, elinewidth=1, capsize=2)
        for yi, v, r in zip(y, value, right, strict=True):
            ax.text(r + 0.02, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 1.15)
        ax.set_title(title, loc="left")
    axes[0].set_yticks(y, [label(d) for d in order["detector"]])
    fig.suptitle("Detection of attack flows, test split", x=0.02, y=1.02, ha="left")
    fig.savefig(path)
    plt.close(fig)


def by_type_figure(by_type: pd.DataFrame, order: list[str], path: Path) -> None:
    table = by_type.pivot(index="detector", columns="attack_cat", values="recall").reindex(order)
    flows = by_type.drop_duplicates("attack_cat").set_index("attack_cat")["flows"]
    columns = flows.sort_values(ascending=False).index.tolist()
    table = table[columns]
    fig, ax = plt.subplots(figsize=(1.0 * len(columns) + 3, 0.45 * len(table) + 1.8))
    image = ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(table.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(columns)), [f"{c}\n({int(flows[c]):,})" for c in columns], fontsize=8)
    ax.set_yticks(range(len(table)), [label(d) for d in table.index])
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.03, label="Recall")
    ax.set_title(f"Recall per attack type at {TARGET_FPR:.0%} false alarm rate (test flows)", loc="left")
    fig.savefig(path)
    plt.close(fig)


def contamination_figure(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for detector in ("iforest", "autoencoder"):
        points = []
        for share in CONTAMINATION_SHARES:
            row = summary.loc[
                (summary["experiment"] == contamination_name(share)) & (summary["detector"] == detector)
            ]
            if len(row):
                points.append((share * 100, float(row["test_pr_auc"].iloc[0])))
        if points:
            x, v = zip(*points, strict=True)
            ax.plot(x, v, marker="o", color=SERIES[detector], linewidth=2, label=label(detector))
    ax.set_xticks([s * 100 for s in CONTAMINATION_SHARES])
    ax.set_xlabel("Attacks in the training sample (percent)")
    ax.set_ylabel("Test PR AUC")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Effect of attacks hidden in the training sample", loc="left")
    fig.savefig(path)
    plt.close(fig)


def ablation_figure(summary: pd.DataFrame, path: Path) -> None:
    names = [d for d in ("iforest", "ecod", "autoencoder", "lightgbm") if d in set(summary["detector"])]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    width = 0.38
    for j, (experiment, title, shade) in enumerate(
        (("main", "All features", SELECTED), ("no_ttl", "Without time to live features", OTHER))
    ):
        part = summary.loc[summary["experiment"] == experiment].set_index("detector").reindex(names)
        bars = ax.bar(np.arange(len(names)) + j * width, part["test_pr_auc"], width, color=shade, label=title)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(names)) + width / 2, [label(d).split(",")[0] for d in names], fontsize=8)
    ax.set_ylabel("Test PR AUC")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.set_title("Ablation: time to live features", loc="left")
    fig.savefig(path)
    plt.close(fig)


def autoencoder_figure(history: dict, path: Path) -> None:
    epochs = pd.DataFrame(history["history"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs["epoch"], epochs["train_loss"], color=SELECTED, linewidth=2, label="Training sample")
    ax.plot(epochs["epoch"], epochs["holdout_loss"], color=REFERENCE, linewidth=2, label="Held out tenth")
    ax.axvline(history["best_epoch"], color=BASELINE, linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared reconstruction error")
    ax.legend(fontsize=8)
    ax.set_title(f"Autoencoder training (kept epoch {history['best_epoch']})", loc="left")
    fig.savefig(path)
    plt.close(fig)


def triage(flows: pd.DataFrame, artifacts: Path, detector: str, n: int) -> str:
    """The top test alerts of a detector, each with the three features that deviate most from the
    training sample (robust z-score of the signed log value)."""
    prep = joblib.load(artifacts / "main" / "preprocessor.joblib")
    fit = flows.iloc[np.load(artifacts / "main" / "fit_rows.npy")]
    test = flows.loc[flows["split"].to_numpy() == "test"].reset_index(drop=True)
    scores = np.load(artifacts / "main" / "scores" / f"{detector}_test.npy")
    s_val = np.load(artifacts / "main" / "scores" / f"{detector}_validation.npy")
    y_val = flows.loc[flows["split"].to_numpy() == "validation", "label"].to_numpy()
    threshold = float(np.quantile(s_val[y_val == 0], 1 - TARGET_FPR, method="higher"))
    columns = prep.numeric
    logged_fit = signed_log(fit[columns].to_numpy(dtype="float64"))
    median = np.median(logged_fit, axis=0)
    iqr = np.subtract(*np.percentile(logged_fit, [75, 25], axis=0))
    scale = np.where(iqr > 1e-6, iqr, logged_fit.std(axis=0) + 1e-6)
    raw_median = fit[columns].median()
    top = np.argsort(-scores)[:n]
    lines = [
        "| Rank | Score | Protocol / service / state "
        "| Most unusual features (value; training median) | Label |",
        "| ---: | ---: | --- | --- | --- |",
    ]
    for rank, row in enumerate(top, start=1):
        flow = test.iloc[row]
        z = np.abs((signed_log(flow[columns].to_numpy(dtype="float64")) - median) / scale)
        reasons = "; ".join(
            f"{columns[j]} {flow[columns[j]]:,.4g} ({raw_median[columns[j]]:,.4g})"
            for j in np.argsort(-z)[:3]
        )
        truth = "normal" if flow["attack_cat"] == NORMAL else flow["attack_cat"]
        lines.append(
            f"| {rank} | {scores[row]:.4g} | {flow['proto']} / {flow['service']} / {flow['state']} | "
            f"{reasons} | {truth} |"
        )
    alerts = scores > threshold
    attack_types = test.loc[alerts & (test["label"].to_numpy() == 1), "attack_cat"].value_counts()
    top_labels = test.iloc[top]["attack_cat"]
    head = [
        f"# Alert triage example: {label(detector)}",
        "",
        f"At the threshold set on validation flows ({TARGET_FPR:.0%} of normal flows flagged), the detector "
        f"raises {int(alerts.sum()):,} alerts on {len(test):,} test flows; "
        f"{int((alerts & (test['label'].to_numpy() == 1)).sum()):,} of them are attacks "
        f"(precision {test['label'].to_numpy()[alerts].mean():.3f}).",
        f"Of the {n} highest scores below, {int((top_labels != NORMAL).sum())} are attacks. An analyst "
        "starts with these. Each row names the three features whose signed log value lies furthest from "
        "the training sample median, in interquartile ranges, with the raw value and the raw training "
        "median. The label column is shown for this example only; the detector does not see it.",
        "",
    ]
    tail = [
        "",
        "Attack types among all alerts: "
        + ", ".join(f"{name} {count:,}" for name, count in attack_types.items())
        + ".",
    ]
    return "\n".join(head + lines + tail) + "\n"


def fmt_interval(row: pd.Series, metric: str) -> str:
    value = f"{row[f'test_{metric}']:.3f}"
    if f"test_{metric}_low" in row and pd.notna(row.get(f"test_{metric}_low")):
        value += f" ({row[f'test_{metric}_low']:.3f} to {row[f'test_{metric}_high']:.3f})"
    return value


def main_table(main: pd.DataFrame, selected: str) -> str:
    rows = []
    for _, row in main.sort_values("test_pr_auc", ascending=False).iterrows():
        name = label(row["detector"]) + (" (selected)" if row["detector"] == selected else "")
        rows.append(
            {
                "Detector": name,
                "Validation PR AUC": f"{row['validation_pr_auc']:.3f}",
                "Test PR AUC (95% interval)": fmt_interval(row, "pr_auc"),
                "Test ROC AUC": f"{row['test_roc_auc']:.3f}",
                f"Recall at {TARGET_FPR:.0%} FPR (95% interval)": fmt_interval(row, "recall"),
                "Test FPR": f"{row['test_fpr']:.4f}",
                "Precision": f"{row['test_precision']:.3f}",
                **{f"Precision of top {k}": f"{row[f'test_precision@{k}']:.3f}" for k in ALERT_BUDGETS},
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    apply_style()
    figures = args.reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    evaluation = args.artifacts / "evaluation"
    summary = pd.read_csv(evaluation / "summary.csv")
    by_type = pd.read_csv(evaluation / "by_type.csv")
    selection = json.loads((evaluation / "selection.json").read_text())
    audit = json.loads((args.data_dir / "audit.json").read_text())
    selected = selection["detector"]
    flows = load(args.data_dir)

    main = summary.loc[summary["experiment"] == "main"]
    order = main.sort_values("test_pr_auc", ascending=False)["detector"].tolist()
    metrics_figure(main, selected, figures / "detectors.png")
    by_type_figure(by_type, order, figures / "recall_by_type.png")
    contamination_figure(summary, figures / "contamination.png")
    ablation_figure(summary, figures / "ttl_ablation.png")
    history = json.loads((args.artifacts / "main" / "models" / "autoencoder" / "history.json").read_text())
    autoencoder_figure(history, figures / "autoencoder_training.png")
    (args.reports / "triage_example.md").write_text(triage(flows, args.artifacts, selected, TRIAGE_ALERTS))

    type_table = (
        by_type.pivot(index="detector", columns="attack_cat", values="recall").reindex(order).round(3)
    )
    type_table.index = [label(d) for d in type_table.index]
    other = summary.loc[summary["experiment"] != "main"].copy()
    other["detector"] = other["detector"].map(label)
    other_cols = ["experiment", "detector", "attack_share", "validation_pr_auc", "test_pr_auc", "test_recall"]
    label_free = main[["detector", "label_free_alert_share", "label_free_fpr", "label_free_recall"]].copy()
    label_free["detector"] = label_free["detector"].map(label)
    timings = {
        name: json.loads(p.read_text())
        for name in ("main", "no_ttl")
        if (p := args.artifacts / name / "timings.json").exists()
    }
    lines = [
        "# Project 8 results",
        "",
        f"Data: {selection['data']}, {audit['layout']} layout, {audit['rows']:,} flows "
        f"({audit['attack_share']:.1%} attacks). Test split: {selection['test_flows']:,} flows "
        f"({selection['test_attack_share']:.1%} attacks). Unsupervised detectors are fitted on a training "
        f"sample with {TRAIN_ATTACK_SHARE:.0%} attacks and no labels. Thresholds: {TARGET_FPR:.0%} of normal "
        "validation flows flagged. Intervals: bootstrap over test flows.",
        "",
        f"Selected on validation PR AUC: **{selection['label']}**.",
        "",
        "## Detectors (main experiment)",
        "",
        main_table(main, selected),
        "",
        "## Recall per attack type at the threshold",
        "",
        type_table.to_markdown(),
        "",
        "## Threshold without labels",
        "",
        f"Threshold at the score above which {TARGET_FPR:.0%} of the training sample lies.",
        "",
        label_free.round(4).to_markdown(index=False),
        "",
        "## Contamination study and time to live ablation",
        "",
        other[other_cols].round(4).to_markdown(index=False),
        "",
        "## Data audit",
        "",
        "```json",
        json.dumps({k: v for k, v in audit.items() if k != "by_type_and_split"}, indent=2),
        "```",
        "",
        "## Run times (seconds)",
        "",
        "```json",
        json.dumps(timings, indent=2),
        "```",
        "",
    ]
    (args.reports / "results.md").write_text("\n".join(lines))
    results = {
        "selection": selection,
        "main": main.round(4).to_dict(orient="records"),
        "other_experiments": summary.loc[summary["experiment"] != "main"].round(4).to_dict(orient="records"),
        "recall_by_type": by_type.round(4).to_dict(orient="records"),
        "audit": audit,
    }
    (args.reports / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.reports / 'results.md'} and {len(list(figures.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
