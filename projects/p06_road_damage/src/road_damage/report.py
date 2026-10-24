"""Data audit, result tables, figures and ``results.json`` under ``reports/``."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.plotting import PALETTE, apply_style

from road_damage.config import (
    ANNOTATIONS_PATH,
    ARTIFACTS_DIR,
    CLASSES,
    DATA_DIR,
    IMAGES_PATH,
    REPORTS_DIR,
)
from road_damage.train import MAIN_RUN, RUNS

RUN_LABELS = {
    "frozen_backbone": "Frozen backbone (baseline)",
    "full": "Full fine tune",
    "light_augment": "Full fine tune, light augmentation (main)",
}


def audit(images: pd.DataFrame, boxes: pd.DataFrame) -> tuple[str, dict]:
    """The data audit as Markdown, and its numbers."""
    boxes = boxes.assign(
        width=boxes["xmax"] - boxes["xmin"],
        height=boxes["ymax"] - boxes["ymin"],
        scored=boxes["label"].isin(CLASSES),
    )
    sizes = images.set_index("image_id")[["width", "height"]]
    boxes = boxes.merge(sizes.add_suffix("_image"), left_on="image_id", right_index=True, how="left")
    boxes["area_share"] = boxes["width"] * boxes["height"] / (boxes["width_image"] * boxes["height_image"])

    per_split = images.pivot_table(index="country", columns="split", values="image_id", aggfunc="count")
    per_split = per_split.reindex(columns=["train", "val", "test"]).fillna(0).astype(int)
    per_split["total"] = per_split.sum(axis=1)
    damaged = boxes.loc[boxes["scored"], "image_id"].unique()
    no_damage = (
        images.assign(no_damage=~images["image_id"].isin(damaged)).groupby("country")["no_damage"].mean()
    )

    per_class = boxes.pivot_table(
        index="label", columns="country", values="image_id", aggfunc="count"
    ).fillna(0)
    per_class = per_class.astype(int)
    per_class["total"] = per_class.sum(axis=1)
    per_class.insert(0, "damage type", [CLASSES.get(code, "not scored, dropped") for code in per_class.index])
    per_class = per_class.sort_values(["damage type", "total"], key=lambda s: s.eq("not scored, dropped"))

    scored = boxes[boxes["scored"]]
    size_rows = scored.groupby("label")["area_share"].describe(percentiles=[0.5])[["count", "50%", "max"]]
    size_rows.columns = ["boxes", "median share of image area", "largest share"]
    resolutions = images.groupby("country")[["width", "height"]].agg(
        lambda s: ", ".join(map(str, s.unique()))
    )
    tertiles = images.groupby("lighting")["brightness"].agg(["min", "max", "count"]).round(1)
    mismatched = int(
        ((images["xml_width"] != images["width"]) | (images["xml_height"] != images["height"])).sum()
    )

    numbers = {
        "images": int(len(images)),
        "images_per_country": per_split["total"].to_dict(),
        "boxes_scored": int(scored.shape[0]),
        "boxes_dropped_other_labels": int((~boxes["scored"]).sum()),
        "share_images_without_damage": no_damage.round(3).to_dict(),
        "xml_size_mismatches": mismatched,
    }
    lines = [
        "# Data audit: RDD2022, Czech and India",
        "",
        "Labelled images only (the `train` folders of RDD2022; its `test` folders have no labels). "
        "Split by blocks of 25 consecutive file names within each country, 70/15/15, seed 42.",
        "",
        "## Images per country and split",
        "",
        per_split.to_markdown(),
        "",
        "Share of images without any of the four scored damage types: "
        + ", ".join(f"{c} {v:.1%}" for c, v in no_damage.items())
        + ".",
        "",
        "Image sizes in pixels (width, height): "
        + "; ".join(f"{c}: {r.width} x {r.height}" for c, r in resolutions.iterrows())
        + f". Annotation files whose stated size differs from the image: {mismatched}.",
        "",
        "## Boxes per label",
        "",
        per_class.to_markdown(),
        "",
        "## Box size, scored labels",
        "",
        size_rows.round(3).to_markdown(),
        "",
        "## Lighting bins (tertiles of mean luminance, 0 to 255)",
        "",
        tertiles.to_markdown(),
        "",
    ]
    return "\n".join(lines), numbers


def plot_class_counts(boxes: pd.DataFrame, path: Path) -> None:
    scored = boxes[boxes["label"].isin(CLASSES)]
    counts = scored.pivot_table(index="label", columns="country", values="image_id", aggfunc="count")
    counts = counts.reindex(list(CLASSES)).fillna(0)
    fig, ax = plt.subplots()
    x = np.arange(len(counts))
    width = 0.8 / len(counts.columns)
    for i, country in enumerate(counts.columns):
        ax.bar(x + i * width, counts[country], width, label=country, color=PALETTE[i])
    ax.set_xticks(x + width * (len(counts.columns) - 1) / 2, [f"{c}\n{CLASSES[c]}" for c in counts.index])
    ax.set_ylabel("boxes")
    ax.set_title("Boxes per damage type and country")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_ap_by_class(results: dict, path: Path) -> None:
    runs = [r for r in RUNS if r in results["runs"]]
    fig, ax = plt.subplots()
    x = np.arange(len(CLASSES))
    width = 0.8 / len(runs)
    for i, run in enumerate(runs):
        values = [results["runs"][run]["ap50_per_class"][c] for c in CLASSES]
        ax.bar(x + i * width, values, width, label=RUN_LABELS[run], color=PALETTE[i])
    ax.set_xticks(x + width * (len(runs) - 1) / 2, [f"{c}\n{CLASSES[c]}" for c in CLASSES])
    ax.set_ylabel("AP at IoU 0.5, test split")
    ax.set_ylim(0, 1)
    ax.set_title("Average precision per damage type")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_breakdown(results: dict, path: Path) -> None:
    keys = list(results["breakdown"])
    values = [results["breakdown"][k]["map50"] for k in keys]
    fig, ax = plt.subplots()
    colors = [PALETTE[0] if k[0].isupper() else PALETTE[1] for k in keys]  # countries are capitalized
    ax.barh(keys, values, color=colors)
    ax.axvline(results["runs"][MAIN_RUN]["map50"], color=PALETTE[2], linestyle="--", label="all test images")
    ax.set_xlabel("mAP at IoU 0.5, test split")
    ax.set_title("Main model by country and by lighting")
    ax.invert_yaxis()
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(results: dict, path: Path) -> None:
    matrix = np.asarray(results["confusion_matrix"], dtype=float)  # rows predicted, columns true
    names = [*CLASSES, "background"]
    share = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for (i, j), value in np.ndenumerate(share):
        if matrix[i, j]:
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.5 else "black")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    conf = results.get("confusion_conf")
    ax.set_title(f"Main model, confidence {conf} or more: share of each true class" if conf else "Main model")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.savefig(path)
    plt.close(fig)


def plot_curves(artifacts_dir: Path, path: Path) -> None:
    fig, ax = plt.subplots()
    for i, run in enumerate(RUNS):
        csv = artifacts_dir / "runs" / run / "results.csv"
        if csv.exists():
            history = pd.read_csv(csv)
            history.columns = history.columns.str.strip()
            ax.plot(history["epoch"], history["metrics/mAP50(B)"], label=RUN_LABELS[run], color=PALETTE[i])
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP at IoU 0.5, validation split")
    ax.set_title("Learning curves")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


# One format per column of the run table: scores to 3 decimals, minutes to 1, epochs as integers.
RUN_TABLE_FORMAT = ("", *[".3f"] * 8, ".1f", ".0f", "")


def results_markdown(results: dict, train: dict, export: dict | None) -> str:
    rows = []
    for run in RUNS:
        if run not in results["runs"]:
            continue
        row = results["runs"][run]
        rows.append(
            {
                "run": RUN_LABELS[run],
                "mAP50": row["map50"],
                "mAP50-95": row["map50_95"],
                "precision": row["precision"],
                "recall": row["recall"],
                **{f"AP50 {c}": row["ap50_per_class"][c] for c in CLASSES},
                "train minutes": train.get(run, {}).get("train_minutes", float("nan")),
                "epochs": train.get(run, {}).get("epochs", float("nan")),
                "trained on": train.get(run, {}).get("device", ""),
            }
        )
    breakdown = pd.DataFrame(
        [
            {
                "subset": key,
                "images": results["test_counts"][key]["images"],
                "boxes": sum(results["test_counts"][key]["boxes"].values()),
                "mAP50": row["map50"],
                "mAP50-95": row["map50_95"],
                **{f"AP50 {c}": row["ap50_per_class"][c] for c in CLASSES},
            }
            for key, row in results["breakdown"].items()
        ]
    )
    lines = [
        "# Results on the test split",
        "",
        "## Runs",
        "",
        pd.DataFrame(rows).to_markdown(index=False, floatfmt=RUN_TABLE_FORMAT),
        "",
        "## Main model by country and lighting",
        "",
        breakdown.to_markdown(index=False, floatfmt=".3f") if not breakdown.empty else "No breakdown.",
        "",
    ]
    if export:
        speed = pd.DataFrame(
            [
                {"runtime": "PyTorch", **export["pytorch"]},
                {"runtime": "ONNX Runtime", **export["onnxruntime"]},
            ]
        )
        lines += [
            f"## CPU inference speed ({export['cpu']['cores']} cores, {export['cpu']['threads']} threads, "
            f"{export['imgsz']} pixels, batch 1)",
            "",
            speed.to_markdown(index=False, floatfmt=".1f"),
            "",
            f"ONNX model: {export['onnx_megabytes']} MB; test mAP50 {export['onnx_test']['map50']:.3f}, "
            f"mAP50-95 {export['onnx_test']['map50_95']:.3f}.",
            "",
        ]
    return "\n".join(lines)


def build(data_dir: Path, artifacts_dir: Path, reports_dir: Path) -> dict:
    apply_style()
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    images = pd.read_parquet(data_dir / IMAGES_PATH.name)
    boxes = pd.read_parquet(data_dir / ANNOTATIONS_PATH.name)
    text, numbers = audit(images, boxes)
    (reports_dir / "data_audit.md").write_text(text)
    plot_class_counts(boxes, figures / "class_counts.png")

    summary = {"data": numbers}
    eval_path = artifacts_dir / "eval_results.json"
    if eval_path.exists():
        results = json.loads(eval_path.read_text())
        train_path, export_path = artifacts_dir / "train_summary.json", artifacts_dir / "export_results.json"
        train = json.loads(train_path.read_text()) if train_path.exists() else {}
        export = json.loads(export_path.read_text()) if export_path.exists() else None
        (reports_dir / "results.md").write_text(results_markdown(results, train, export))
        plot_ap_by_class(results, figures / "ap50_by_class.png")
        plot_curves(artifacts_dir, figures / "learning_curves.png")
        if results["breakdown"]:
            plot_breakdown(results, figures / "breakdown.png")
        if "confusion_matrix" in results:
            plot_confusion(results, figures / "confusion_matrix.png")
        summary |= {"test": results, "train": train, "export": export}
    (reports_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    build(args.data_dir, args.artifacts, args.reports)


if __name__ == "__main__":
    main()
