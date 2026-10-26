"""Robustness scorecard, defense table and figures from the results of each part.

A part without results (for example images without detector weights) is left out of the report.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from portfolio_common.config import parse_args
from portfolio_common.plotting import PALETTE, apply_style

from adv_robustness.config import ARTIFACTS_DIR, POISON_TYPE, REPORTS_DIR

CREDIT_LABELS = {
    "baseline": "Project 1 model",
    "verified_fields": "Without self reported fields",
    "adversarial_training": "Adversarial training",
    "verified_and_adversarial": "Both defenses",
}
FLOW_LABELS = {
    "lightgbm": "LightGBM (project 8 reference)",
    "lightgbm_no_ttl": "LightGBM without TTL features",
    "lightgbm_adversarial": "LightGBM, adversarial training",
    "iforest": "Isolation Forest (project 8 selection)",
}
IMAGE_LABELS = {
    "clean": "No attack",
    "noise_patch": "Random noise patch (control)",
    "adversarial_patch": "Adversarial patch",
    "fgsm_2": "FGSM, epsilon 2/255",
    "fgsm_8": "FGSM, epsilon 8/255",
    "pgd_2": "PGD, epsilon 2/255",
    "pgd_8": "PGD, epsilon 8/255",
}
DEFENSE_LABELS = {"none": "none", "jpeg": "JPEG 75", "median": "median 3x3", "noise": "Gaussian noise"}


def load(artifacts: Path) -> dict:
    found = {}
    for part in ("credit", "flows", "images"):
        path = artifacts / part / "results.json"
        if path.exists():
            found[part] = json.loads(path.read_text())
    return found


def table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".3f")


def credit_tables(result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    score, defense = [], []
    base = result["variants"]["baseline"]
    for name, v in result["variants"].items():
        both = v["attacks"]["both"]
        score.append(
            {
                "Model": CREDIT_LABELS[name],
                "ROC AUC": v["clean"]["roc_auc"],
                "Approval rate": v["clean"]["approval_rate"],
                "Rejections turned (payment)": v["attacks"]["payment"]["flip_rate"],
                "Rejections turned (self reported)": v["attacks"]["self_reported"]["flip_rate"],
                "Rejections turned (both)": both["flip_rate"],
                "Defaulters among them turned": both["flip_rate_defaulters"],
                "Profit clean (M NT$)": v["clean"]["profit"] / 1e6,
                "Profit attacked (M NT$)": both["after"]["profit"] / 1e6,
            }
        )
        if name != "baseline":
            defense.append(
                {
                    "Target": "Credit model (project 1)",
                    "Defense": CREDIT_LABELS[name],
                    "Clean metric": "ROC AUC",
                    "Clean change": v["clean"]["roc_auc"] - base["clean"]["roc_auc"],
                    "Attack metric": "rejections turned (both)",
                    "Attacked before": base["attacks"]["both"]["flip_rate"],
                    "Attacked after": both["flip_rate"],
                }
            )
    return pd.DataFrame(score), pd.DataFrame(defense)


def flow_tables(result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    score, defense = [], []
    base = result["models"]["lightgbm"]
    for name, m in result["models"].items():
        groups = m["evasion"]["groups"]
        score.append(
            {
                "Detector": FLOW_LABELS[name],
                "Recall at 1% FPR": m["clean"]["recall"],
                "Test FPR": m["clean"]["fpr"],
                "Detected flows attacked": m["evasion"]["attacked"],
                "Evaded (size and timing)": groups["size_timing"]["evasion_rate"],
                "Evaded (TTL)": groups["ttl"]["evasion_rate"],
                "Evaded (all)": groups["all"]["evasion_rate"],
                "Recall after attack": groups["all"]["recall_after"],
            }
        )
        if name in ("lightgbm_no_ttl", "lightgbm_adversarial"):
            defense.append(
                {
                    "Target": "Flow detector (project 8)",
                    "Defense": FLOW_LABELS[name],
                    "Clean metric": "recall at 1% FPR",
                    "Clean change": m["clean"]["recall"] - base["clean"]["recall"],
                    "Attack metric": "recall after evasion",
                    "Attacked before": base["evasion"]["groups"]["all"]["recall_after"],
                    "Attacked after": groups["all"]["recall_after"],
                }
            )
    return pd.DataFrame(score), pd.DataFrame(defense)


def poisoning_table(result: dict) -> pd.DataFrame:
    rows = []
    for share, entry in result["poisoning"].items():
        row = {"Poison share": float(share), "Poison flows": entry["poison_flows"]}
        for defense in ("none", "trim", "dedup"):
            row[f"{POISON_TYPE} recall ({defense})"] = entry[defense]["recall_target"]
            row[f"Other recall ({defense})"] = entry[defense]["recall_other"]
        rows.append(row)
    return pd.DataFrame(rows)


def smoothing_table(result: dict) -> pd.DataFrame:
    rows = []
    for sigma, s in result["smoothing"].items():
        rows.append(
            {
                "Sigma": float(sigma),
                "Accuracy": s["accuracy"],
                "Recall": s["recall"],
                "FPR": s["fpr"],
                "Abstain": s["abstain"],
            }
            | {f"Certified at radius {r}": v for r, v in s["certified_accuracy"].items()}
        )
    return pd.DataFrame(rows)


def image_tables(result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = pd.DataFrame(
        [
            {"Attack": IMAGE_LABELS.get(name, name)}
            | {DEFENSE_LABELS[d]: v[d]["map50"] for d in DEFENSE_LABELS if d in v}
            for name, v in result["results"].items()
        ]
    )
    clean = result["results"]["clean"]
    # The patch is the attack that works in the physical world; PGD at 2/255 is the weakest digital one.
    shown = [n for n in ("adversarial_patch", "pgd_2") if n in result["results"]]
    defense = [
        {
            "Target": "Road damage detector (project 6)",
            "Defense": DEFENSE_LABELS[d],
            "Clean metric": "mAP50",
            "Clean change": clean[d]["map50"] - clean["none"]["map50"],
            "Attack metric": f"mAP50 under {IMAGE_LABELS.get(name, name)}",
            "Attacked before": result["results"][name]["none"]["map50"],
            "Attacked after": result["results"][name][d]["map50"],
        }
        for name in shown
        for d in DEFENSE_LABELS
        if d != "none"
    ]
    return grid, pd.DataFrame(defense)


def credit_figure(result: dict, path: Path) -> None:
    fig, ax = plt.subplots()
    names = list(result["variants"])
    width = 0.25
    for i, attack in enumerate(("payment", "self_reported", "both")):
        values = [result["variants"][n]["attacks"][attack]["flip_rate"] for n in names]
        ax.bar(
            np.arange(len(names)) + (i - 1) * width,
            values,
            width,
            label=attack.replace("_", " "),
            color=PALETTE[i],
        )
    ax.set_xticks(np.arange(len(names)), [CREDIT_LABELS[n] for n in names], rotation=12)
    ax.set_ylabel("Share of rejected applicants approved after the change")
    ax.set_title("Credit model: rejections turned into approvals, by what the applicant changes")
    ax.legend(title="Change")
    fig.savefig(path)
    plt.close(fig)


def flows_figure(result: dict, path: Path) -> None:
    fig, ax = plt.subplots()
    names = list(result["models"])
    clean = [result["models"][n]["clean"]["recall"] for n in names]
    after = [result["models"][n]["evasion"]["groups"]["all"]["recall_after"] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, clean, 0.4, label="no attack", color=PALETTE[0])
    ax.bar(x + 0.2, after, 0.4, label="after evasion", color=PALETTE[1])
    ax.set_xticks(x, [FLOW_LABELS[n] for n in names], rotation=12)
    ax.set_ylabel("Recall at 1% false alarm rate")
    ax.set_title("Flow detectors: attack flows found, before and after the attacker's changes")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def poisoning_figure(result: dict, path: Path) -> None:
    frame = poisoning_table(result)
    fig, ax = plt.subplots()
    for i, defense in enumerate(("none", "trim", "dedup")):
        ax.plot(
            frame["Poison share"],
            frame[f"{POISON_TYPE} recall ({defense})"],
            marker="o",
            color=PALETTE[i],
            label=f"{POISON_TYPE}, defense: {defense}",
        )
    ax.plot(
        frame["Poison share"],
        frame["Other recall (none)"],
        marker="s",
        linestyle="--",
        color=PALETTE[3],
        label="other attack types, no defense",
    )
    ax.set_xlabel(f"Share of {POISON_TYPE} flows slipped into the training traffic")
    ax.set_ylabel("Recall at 1% false alarm rate")
    ax.set_title("Isolation Forest: poisoning the unlabeled training traffic")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def images_figure(result: dict, path: Path) -> None:
    grid, _ = image_tables(result)
    values = grid.drop(columns="Attack").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(values.max(), 1e-6), aspect="auto")
    ax.set_xticks(range(values.shape[1]), grid.columns[1:])
    ax.set_yticks(range(len(grid)), grid["Attack"])
    ax.grid(False)
    for (i, j), v in np.ndenumerate(values):
        ax.text(
            j, i, f"{v:.3f}", ha="center", va="center", color="white" if v > values.max() * 0.6 else "black"
        )
    ax.set_xlabel("Input transformation defense")
    ax.set_title("Road damage detector: test mAP50 by attack and defense")
    fig.colorbar(im, ax=ax, label="mAP50")
    fig.savefig(path)
    plt.close(fig)


def examples_figure(artifacts: Path, path: Path) -> bool:
    names = ["clean", "pgd_8", "noise_patch", "adversarial_patch"]
    files = [artifacts / "images" / f"example_{n}.png" for n in names]
    if not all(f.exists() for f in files):
        return False
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, name, file in zip(axes, names, files, strict=True):
        ax.imshow(Image.open(file))
        ax.set_title(IMAGE_LABELS[name], fontsize=9)
        ax.axis("off")
    fig.savefig(path)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--data-label", default="full data")
    args = parse_args(parser, __file__)
    apply_style()
    figures = args.reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    results = load(args.artifacts)
    sections = [f"# Project 13 results ({args.data_label})\n"]
    defenses = []
    if "credit" in results:
        score, defense = credit_tables(results["credit"])
        defenses.append(defense)
        credit_figure(results["credit"], figures / "credit_attacks.png")
        r = results["credit"]
        sections += [
            "## Credit model (project 1)\n",
            f"Test clients: {r['n_test']:,}. Budget: extra payment up to {r['budget']['extra_payment']:.0%} "
            "of the credit limit in the latest month; any education and marital status.\n",
            table(score),
        ]
    if "flows" in results:
        score, defense = flow_tables(results["flows"])
        defenses.append(defense)
        flows_figure(results["flows"], figures / "flow_evasion.png")
        poisoning_figure(results["flows"], figures / "poisoning.png")
        hsj = results["flows"]["hop_skip_jump"]["lightgbm"]
        sections += [
            "## Flow detectors (project 8)\n",
            f"Initial TTL values the attacker tries: {results['flows']['ttl_values']}.\n",
            table(score),
            f"Unconstrained reference (HopSkipJump, any feature): {hsj['evasion_rate']:.1%} of "
            f"{hsj['attacked']} detected flows evade LightGBM, median L2 change {hsj['median_l2']:.2f} "
            "in the standardized space.",
            "### Poisoning of Isolation Forest\n",
            table(poisoning_table(results["flows"])),
            "### Randomized smoothing of LightGBM (numeric features, standardized log scale)",
            "Largest feasible change of size and timing, median L2: "
            f"{results['flows']['largest_change_l2']:.2f}.",
            table(smoothing_table(results["flows"])),
        ]
    if "images" in results:
        grid, defense = image_tables(results["images"])
        defenses.append(defense)
        images_figure(results["images"], figures / "image_attacks.png")
        examples_figure(args.artifacts, figures / "image_examples.png")
        r = results["images"]
        sections += [
            "## Road damage detector (project 6)\n",
            f"Test images: {r['test_images']} ({r['test_images_with_damage']} with damage), "
            f"weights {r['weights']}, input {r['image_size']} pixels.",
            table(grid),
        ]
    if defenses:
        sections += ["## Defense trade offs\n", table(pd.concat(defenses, ignore_index=True))]
    (args.reports / "results.md").write_text("\n\n".join(s.strip() for s in sections) + "\n")
    (args.reports / "results.json").write_text(json.dumps({"data": args.data_label, **results}, indent=2))
    print(f"report written to {args.reports}")


if __name__ == "__main__":
    main()
