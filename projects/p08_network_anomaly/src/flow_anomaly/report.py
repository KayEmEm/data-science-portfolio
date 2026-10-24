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
    CONTAMINATION_SEEDS,
    CONTAMINATION_SHARES,
    DATA_DIR,
    REPORTS_DIR,
    RESCALED_PREVALENCE,
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
TRIAGE_GROUPS = 15


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
    order = main.sort_values("test_pr_auc_rescaled")
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(order) + 1.6), sharey=True)
    y = np.arange(len(order))
    colors = [color(d, selected) for d in order["detector"]]
    panels = (
        ("pr_auc_rescaled", f"PR AUC at {RESCALED_PREVALENCE:.0%} attacks"),
        ("recall", f"Recall at {TARGET_FPR:.0%} false alarm rate"),
    )
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


def contamination_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest and autoencoder scores per training attack share and seed."""
    rows = []
    for share in CONTAMINATION_SHARES:
        for seed in CONTAMINATION_SEEDS:
            part = summary.loc[
                (summary["experiment"] == contamination_name(share, seed))
                & summary["detector"].isin(["iforest", "autoencoder"])
            ]
            rows.append(part.assign(attack_share=share, seed=seed))
    return pd.concat(rows, ignore_index=True)


def contamination_figure(study: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = (("test_roc_auc", "Test ROC AUC"), ("test_recall", f"Test recall at {TARGET_FPR:.0%} FPR"))
    for ax, (metric, title) in zip(axes, panels, strict=True):
        for detector in ("iforest", "autoencoder"):
            stats = (
                study.loc[study["detector"] == detector]
                .groupby("attack_share")[metric]
                .agg(["mean", "min", "max"])
            )
            x = stats.index.to_numpy() * 100
            ax.errorbar(
                x,
                stats["mean"],
                yerr=[stats["mean"] - stats["min"], stats["max"] - stats["mean"]],
                marker="o",
                color=SERIES[detector],
                linewidth=2,
                capsize=3,
                label=label(detector),
            )
        ax.set_xticks([s * 100 for s in CONTAMINATION_SHARES])
        ax.set_xlabel("Attacks in the training sample (percent)")
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Attacks hidden in the training sample (mean of {len(CONTAMINATION_SEEDS)} seeds, range as bars)",
        x=0.02,
        y=1.03,
        ha="left",
    )
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
        bars = ax.bar(
            np.arange(len(names)) + j * width, part["test_pr_auc_rescaled"], width, color=shade, label=title
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(names)) + width / 2, [label(d).split(",")[0] for d in names], fontsize=8)
    ax.set_ylabel(f"Test PR AUC at {RESCALED_PREVALENCE:.0%} attacks")
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


def triage(flows: pd.DataFrame, artifacts: Path, detector: str, n_groups: int) -> str:
    """The test alerts of a detector, grouped as an analyst would work them: by protocol, service, state
    and the three features that deviate most from the training sample (robust z-score of the signed log
    value). Groups are listed by their highest score."""
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

    alerts = test.loc[scores > threshold].copy()
    alerts["score"] = scores[scores > threshold]
    z = np.abs((signed_log(alerts[columns].to_numpy(dtype="float64")) - median) / scale)
    top3 = np.argsort(-z, axis=1)[:, :3]
    alerts["features"] = [", ".join(sorted(columns[j] for j in row)) for row in top3]
    alerts["traffic"] = alerts["proto"] + " / " + alerts["service"] + " / " + alerts["state"]
    groups = (
        alerts.groupby(["traffic", "features"])
        .agg(
            alerts=("score", "size"),
            top_score=("score", "max"),
            attacks=("label", "mean"),
            types=("attack_cat", lambda c: ", ".join(f"{k} {v:,}" for k, v in c.value_counts().items())),
        )
        .reset_index()
        .sort_values("top_score", ascending=False)
    )

    def values(row) -> str:
        part = alerts.loc[(alerts["traffic"] == row["traffic"]) & (alerts["features"] == row["features"])]
        names = row["features"].split(", ")
        return "; ".join(f"{c} {part[c].median():,.4g} ({raw_median[c]:,.4g})" for c in names)

    lines = [
        "| Group | Alerts | Top score | Protocol / service / state "
        "| Most unusual features: group median (training median) | Share of attacks | Labels |",
        "| ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    shown = groups.head(n_groups)
    for rank, (_, row) in enumerate(shown.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['alerts']:,} | {row['top_score']:.4g} | {row['traffic']} | {values(row)} | "
            f"{row['attacks']:.2f} | {row['types']} |"
        )
    n_attacks = int(alerts["label"].sum())
    head = [
        f"# Alert triage example: {label(detector)}",
        "",
        f"At the threshold set on validation flows ({TARGET_FPR:.0%} of normal flows flagged), the detector "
        f"raises {len(alerts):,} alerts on {len(test):,} test flows; {n_attacks:,} of them are attacks "
        f"(precision {n_attacks / max(len(alerts), 1):.3f}).",
        "",
        f"An analyst does not read {len(alerts):,} alerts one by one. They fall into {len(groups):,} groups "
        "of the same protocol, service and state with the same three most unusual features (the features "
        "whose signed log value lies furthest from the training sample median, in interquartile ranges). "
        f"The {len(shown)} groups with the highest scores are below. The share of attacks and the labels are "
        "shown for this example only; the detector does not see them.",
        "",
    ]
    by_type = alerts.loc[alerts["label"] == 1, "attack_cat"].value_counts()
    tail = [
        "",
        "Attack types among all alerts: " + ", ".join(f"{k} {v:,}" for k, v in by_type.items()) + ".",
    ]
    return "\n".join(head + lines + tail) + "\n"


def fmt_interval(row: pd.Series, metric: str) -> str:
    value = f"{row[f'test_{metric}']:.3f}"
    if f"test_{metric}_low" in row and pd.notna(row.get(f"test_{metric}_low")):
        value += f" ({row[f'test_{metric}_low']:.3f} to {row[f'test_{metric}_high']:.3f})"
    return value


def main_table(main: pd.DataFrame, selected: str) -> str:
    rows = []
    share = f"{RESCALED_PREVALENCE:.0%} attacks"
    for _, row in main.sort_values("test_pr_auc_rescaled", ascending=False).iterrows():
        name = label(row["detector"]) + (" (selected)" if row["detector"] == selected else "")
        rows.append(
            {
                "Detector": name,
                f"Validation PR AUC at {share}": f"{row['validation_pr_auc_rescaled']:.3f}",
                f"Test PR AUC at {share} (95% interval)": fmt_interval(row, "pr_auc_rescaled"),
                "Test PR AUC as observed (95% interval)": fmt_interval(row, "pr_auc"),
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
    order = main.sort_values("test_pr_auc_rescaled", ascending=False)["detector"].tolist()
    metrics_figure(main, selected, figures / "detectors.png")
    by_type_figure(by_type, order, figures / "recall_by_type.png")
    study = contamination_rows(summary)
    contamination_figure(study, figures / "contamination.png")
    ablation_figure(summary, figures / "ttl_ablation.png")
    history = json.loads((args.artifacts / "main" / "models" / "autoencoder" / "history.json").read_text())
    autoencoder_figure(history, figures / "autoencoder_training.png")
    (args.reports / "triage_example.md").write_text(triage(flows, args.artifacts, selected, TRIAGE_GROUPS))

    type_table = (
        by_type.pivot(index="detector", columns="attack_cat", values="recall").reindex(order).round(3)
    )
    type_table.index = [label(d) for d in type_table.index]
    metrics = ["test_roc_auc", "test_recall", "test_pr_auc_rescaled"]
    study_table = study.groupby(["detector", "attack_share"])[metrics].agg(["mean", "min", "max"]).round(3)
    study_table.columns = [f"{m} {stat}" for m, stat in study_table.columns]
    study_table = study_table.reset_index()
    study_table["detector"] = study_table["detector"].map(label)
    ablation = summary.loc[summary["experiment"].isin(["main", "no_ttl"])]
    ablation = ablation.loc[ablation["detector"].isin(["iforest", "ecod", "autoencoder", "lightgbm"])]
    ablation = ablation.pivot(index="detector", columns="experiment", values=metrics).round(3)
    ablation.columns = [f"{m} {e}" for m, e in ablation.columns]
    ablation.index = [label(d) for d in ablation.index]
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
        f"validation flows flagged. PR AUC is shown as observed and rescaled to {RESCALED_PREVALENCE:.0%} "
        "attacks (precision recomputed from recall and false alarm rate). Intervals: bootstrap over test "
        "flows.",
        "",
        f"Selected on validation PR AUC at {RESCALED_PREVALENCE:.0%} attacks: **{selection['label']}**.",
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
        "## Contamination study",
        "",
        "Isolation Forest and the autoencoder; each share drawn and fitted with seeds "
        f"{CONTAMINATION_SEEDS}.",
        "",
        study_table.to_markdown(index=False),
        "",
        "## Time to live ablation",
        "",
        "`main` uses all features, `no_ttl` drops sttl, dttl and ct_state_ttl.",
        "",
        ablation.to_markdown(),
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
        "contamination_study": study.round(4).to_dict(orient="records"),
        "ttl_ablation": summary.loc[summary["experiment"] == "no_ttl"].round(4).to_dict(orient="records"),
        "recall_by_type": by_type.round(4).to_dict(orient="records"),
        "audit": audit,
    }
    (args.reports / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.reports / 'results.md'} and {len(list(figures.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
