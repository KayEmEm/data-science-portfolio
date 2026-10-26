"""Tables, figures and results.json under reports/, from the outputs of evaluate.py."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.config import parse_args
from portfolio_common.plotting import PALETTE, apply_style
from scipy import stats

from inventory_rl import demand as demand_model
from inventory_rl.baselines import load_baselines
from inventory_rl.config import ARTIFACTS_DIR, DATA_DIR, REPORTS_DIR, SHAPING, SHAPING_PRODUCT, TEST_START
from inventory_rl.data import load_demand, load_products
from inventory_rl.dynamics import Problem, base_stock, s_s, seeded_scenarios, simulate

LABELS = {"base_stock": "Base stock (R, S)", "s_S": "(s, S)", "ppo": "PPO", "sac": "SAC"}
COLORS = {"base_stock": PALETTE[2], "s_S": PALETTE[3], "ppo": PALETTE[0], "sac": PALETTE[1]}
SHAPING_COLORS = {"cost": PALETTE[0], "potential": PALETTE[4], "service": PALETTE[2]}


def load(artifacts: Path) -> tuple[pd.DataFrame, dict]:
    out = artifacts / "evaluation"
    return pd.read_csv(out / "summary.csv"), json.loads((out / "selection.json").read_text())


def main_rows(summary: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """Per product: the two baselines and the selected seed of each RL algorithm."""
    rows = []
    for product, chosen in selection["products"].items():
        mine = summary[summary["product"] == product].set_index("policy")
        for key in ("base_stock", "s_S"):
            rows.append(mine.loc[key].to_dict() | {"product": product, "policy": key, "kind": key})
        for algorithm, policy in chosen["selected"].items():
            rows.append(
                mine.loc[policy].to_dict() | {"product": product, "policy": policy, "kind": algorithm}
            )
    return pd.DataFrame(rows)


def seed_spread(summary: pd.DataFrame) -> pd.DataFrame:
    rl = summary[summary["shaping"] == "cost"]
    rl = rl[rl["algorithm"].isin(["ppo", "sac"])]
    return (
        rl.groupby(["product", "algorithm"])
        .agg(
            seeds=("policy", "size"),
            test_mean=("test_cost", "mean"),
            test_sd=("test_cost", "std"),
            test_min=("test_cost", "min"),
            test_max=("test_cost", "max"),
            pct_against_s_S=("test_diff_share", lambda d: 100 * d.mean()),
            beats_s_S=("test_diff_high", lambda d: int((d < 0).sum())),
        )
        .reset_index()
    )


def figure_demand(products: list[dict], demand: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(products), figsize=(4.2 * len(products), 3.6))
    for ax, product in zip(np.atleast_1d(axes), products, strict=True):
        part = demand[(demand["product"] == product["product"]) & ~demand["closed"].astype(bool)]
        test = part[part["d"] >= TEST_START]
        model = product["demand"]
        top = int(np.quantile(test["units"], 0.995)) + 1
        values = np.arange(0, top + 1)
        real = np.bincount(np.clip(test["units"], 0, top), minlength=top + 1) / len(test)
        n, p = demand_model.nb_params(
            np.asarray(model["weekday_mean"])[test["wday"].to_numpy() - 1], model["dispersion"]
        )
        fitted = np.array([stats.nbinom.pmf(v, n, p).mean() for v in values])
        width = max(1, top // 40)
        ax.bar(values, real, width=0.8 * width, color=PALETTE[0], alpha=0.55, label="real test year")
        ax.plot(values, fitted, color=PALETTE[1], lw=2, label="model fitted on the year before")
        ax.set_title(f"{product['product']}: {product['series_id']}", fontsize=9)
        ax.set_xlabel("units per day")
    np.atleast_1d(axes)[0].set_ylabel("share of days")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Daily demand: negative binomial with weekday means against the real test year")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def baseline_validation_costs(product: dict, artifacts: Path, episodes: int) -> dict:
    from inventory_rl.config import VALIDATION_SEED

    problem = Problem.from_product(product)
    tuned = load_baselines(artifacts, problem.name)
    scenarios = seeded_scenarios(problem, VALIDATION_SEED, episodes)
    policies = {
        "base_stock": base_stock(tuned["base_stock"]["S"], tuned["base_stock"]["R"]),
        "s_S": s_s(tuned["s_S"]["s"], tuned["s_S"]["S"]),
    }
    return {k: float(simulate(problem, scenarios, p)["totals"]["cost"].mean()) for k, p in policies.items()}


def curves(artifacts: Path, summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in summary[summary["algorithm"].isin(["ppo", "sac"])].itertuples():
        path = artifacts / "runs" / row.policy / "curve.csv"
        if path.exists():
            frames.append(
                pd.read_csv(path).assign(
                    product=row.product, algorithm=row.algorithm, shaping=row.shaping, policy=row.policy
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def band(ax, frame: pd.DataFrame, color: str, label: str) -> None:
    pivot = frame.pivot_table(index="step", columns="policy", values="cost").sort_index().ffill()
    steps = pivot.index.to_numpy()
    ax.plot(
        steps,
        pivot.median(axis=1),
        color=color,
        lw=2,
        label=f"{label} (median of {pivot.shape[1]} seeds)" if pivot.shape[1] > 1 else f"{label}, one seed",
    )
    ax.fill_between(steps, pivot.min(axis=1), pivot.max(axis=1), color=color, alpha=0.18, lw=0)


def figure_learning(products, artifacts, summary, curve, episodes, path: Path) -> None:
    fig, axes = plt.subplots(len(products), 2, figsize=(10, 3.2 * len(products)), squeeze=False)
    for i, product in enumerate(products):
        name = product["product"]
        refs = baseline_validation_costs(product, artifacts, episodes)
        for j, algorithm in enumerate(("ppo", "sac")):
            ax = axes[i, j]
            mine = curve[
                (curve["product"] == name) & (curve["algorithm"] == algorithm) & (curve["shaping"] == "cost")
            ]
            if len(mine):
                band(ax, mine, COLORS[algorithm], LABELS[algorithm])
            for key, style in (("s_S", "-"), ("base_stock", "--")):
                ax.axhline(refs[key], color=COLORS[key], ls=style, lw=1.5, label=f"{LABELS[key]}, tuned")
            costs = mine["cost"] if len(mine) else pd.Series([refs["base_stock"]])
            ax.set_ylim(
                0, max(2.2 * refs["base_stock"], 1.0) if costs.max() > 2.2 * refs["base_stock"] else None
            )
            ax.set_title(f"{name}: {LABELS[algorithm]}", fontsize=10)
            ax.set_xlabel("training steps (days simulated), thousands")
            ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}")
            ax.set_ylabel("validation cost per year ($)")
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Learning curves: mean yearly cost on the validation scenarios (band: best to worst seed)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_costs(rows: pd.DataFrame, path: Path, prefix: str = "test", title: str = "") -> None:
    products = list(dict.fromkeys(rows["product"]))
    fig, axes = plt.subplots(1, len(products), figsize=(4.2 * len(products), 3.8), squeeze=False)
    for ax, product in zip(axes[0], products, strict=True):
        mine = rows[rows["product"] == product]
        x = np.arange(len(mine))
        cost = mine[f"{prefix}_cost"].to_numpy()
        err = np.vstack([cost - mine[f"{prefix}_cost_low"], mine[f"{prefix}_cost_high"] - cost])
        ax.bar(x, cost, color=[COLORS[k] for k in mine["kind"]], yerr=err, capsize=3, width=0.65)
        for xi, value, top in zip(x, cost, mine[f"{prefix}_cost_high"], strict=True):
            ax.text(xi, top, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [LABELS[k] for k in mine["kind"]], fontsize=8)
        ax.set_title(product, fontsize=10)
        ax.set_ylabel("cost per year ($)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_seeds(summary: pd.DataFrame, path: Path) -> None:
    rl = summary[(summary["shaping"] == "cost") & summary["algorithm"].isin(["ppo", "sac"])]
    products = list(dict.fromkeys(summary["product"]))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, algorithm in enumerate(("ppo", "sac")):
        for i, product in enumerate(products):
            mine = rl[(rl["product"] == product) & (rl["algorithm"] == algorithm)]
            x = i + (k - 0.5) * 0.3 + np.linspace(-0.05, 0.05, max(len(mine), 1))[: len(mine)]
            ax.scatter(
                x,
                100 * mine["test_diff_share"],
                color=COLORS[algorithm],
                s=36,
                label=LABELS[algorithm] if i == 0 else None,
                zorder=3,
            )
    ax.axhline(0, color=COLORS["s_S"], lw=1.5, label="(s, S), tuned")
    ax.set_xticks(range(len(products)), products)
    ax.set_ylabel("test cost against (s, S), percent")
    ax.set_title("Every seed: test cost relative to the tuned (s, S) policy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_shaping(summary: pd.DataFrame, path: Path) -> None:
    mine = summary[(summary["product"] == SHAPING_PRODUCT) & (summary["algorithm"] == "ppo")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for i, shaping in enumerate(SHAPING):
        part = mine[mine["shaping"] == shaping]
        x = i + np.linspace(-0.1, 0.1, max(len(part), 1))[: len(part)]
        axes[0].scatter(x, part["test_cost"], color=SHAPING_COLORS[shaping], s=40, zorder=3)
        axes[1].scatter(x, 100 * part["test_fill_rate"], color=SHAPING_COLORS[shaping], s=40, zorder=3)
    ref = summary[(summary["product"] == SHAPING_PRODUCT) & (summary["policy"] == "s_S")].iloc[0]
    axes[0].axhline(ref["test_cost"], color=COLORS["s_S"], lw=1.5, label="(s, S), tuned")
    axes[1].axhline(100 * ref["test_fill_rate"], color=COLORS["s_S"], lw=1.5, label="(s, S), tuned")
    for ax, label in zip(axes, ("true test cost per year ($)", "fill rate (percent)"), strict=True):
        ax.set_xticks(range(len(SHAPING)), list(SHAPING))
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle(f"Reward shaping, PPO on the {SHAPING_PRODUCT} product: each dot is one seed")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_policy_map(artifacts: Path, products: list[dict], selection: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, len(products), figsize=(4.2 * len(products), 3.6), squeeze=False)
    for ax, product in zip(axes[0], products, strict=True):
        name = product["product"]
        frame = pd.read_csv(artifacts / "evaluation" / f"policy_map_{name}.csv")
        ax.plot(frame["position"], frame["s_S"], color=COLORS["s_S"], lw=2.5, label=LABELS["s_S"])
        for algorithm, policy in selection["products"][name]["selected"].items():
            ax.plot(
                frame["position"], frame[policy], color=COLORS[algorithm], lw=1.5, label=LABELS[algorithm]
            )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("inventory position, all on hand (units)")
        ax.set_ylabel("order placed (units)")
        ax.legend(fontsize=8)
    fig.suptitle("Order rule of each policy on a Monday with nothing on order")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".1f")


def dp_lines(artifacts: Path) -> tuple[list[str], dict | None]:
    path = artifacts / "dp" / "result.json"
    if not path.exists():
        return [], None
    dp = json.loads(path.read_text())
    table = pd.DataFrame(
        [
            {
                "Policy": name,
                "Simulated cost per year ($)": f"{dp[key]['mean']:.2f} +- {dp[key]['half_width']:.2f}",
                "Above the optimum": "" if key == "optimal" else f"{100 * dp[key]['gap_share']:+.1f}%",
                "Parameters": ""
                if key == "optimal"
                else dp[key]["params"].get("run", str(dp[key]["params"])),
            }
            for key, name in (
                ("optimal", "Optimal (dynamic programming)"),
                ("s_S", "(s, S)"),
                ("base_stock", "Base stock (R, S)"),
                ("ppo", "PPO, seed with the lowest validation cost"),
            )
            if key in dp
        ]
    )
    lines = [
        "## Validation against the exact optimum "
        f"({dp['product']} product, lead time fixed at {dp['lead_time']} days)",
        "",
        "Expected cost of the optimal policy by backward induction: "
        f"{dp['expected_optimal']:.2f} $ per year. "
        f"Simulated over {dp['episodes']} years:",
        "",
        table.to_markdown(index=False),
        "",
    ]
    return lines, dp


def write_results(rows, spread, summary, products, selection, reports: Path, artifacts: Path) -> dict:
    main = rows.assign(
        policy=[LABELS[k] for k in rows["kind"]],
        test=[
            f"{c:.0f} ({lo:.0f} to {hi:.0f})"
            for c, lo, hi in zip(
                rows["test_cost"], rows["test_cost_low"], rows["test_cost_high"], strict=True
            )
        ],
        vs_s_S=[("" if np.isnan(d) else f"{100 * d:+.1f}%") for d in rows["test_diff_share"].fillna(np.nan)],
        fill=[f"{100 * f:.1f}%" for f in rows["test_fill_rate"]],
        replay=[f"{c:.0f}" for c in rows["replay_cost"]],
        replay_fill=[f"{100 * f:.1f}%" for f in rows["replay_fill_rate"]],
    )[["product", "policy", "test", "vs_s_S", "fill", "test_orders", "test_on_hand", "replay", "replay_fill"]]
    main.columns = [
        "Product",
        "Policy",
        "Test cost per year, 95% CI ($)",
        "Against (s, S)",
        "Fill rate",
        "Orders per year",
        "Mean on hand (units)",
        "Replay cost ($)",
        "Replay fill rate",
    ]
    shaping = summary[(summary["product"] == SHAPING_PRODUCT) & (summary["algorithm"] == "ppo")]
    shaping = (
        shaping.groupby("shaping")
        .agg(
            seeds=("policy", "size"),
            test_cost=("test_cost", "mean"),
            sd=("test_cost", "std"),
            fill=("test_fill_rate", "mean"),
            orders=("test_orders", "mean"),
        )
        .reindex(list(SHAPING))
        .reset_index()
    )
    product_table = pd.DataFrame(
        [
            {
                "Product": p["product"],
                "Series": p["series_id"],
                "Mean units per day": np.mean(p["demand"]["weekday_mean"]),
                "Dispersion": p["demand"]["dispersion"],
                "Price ($)": p["mean_price"],
                "Lost margin per unit ($)": p["lost_sale"],
                "Holding per unit and day ($)": p["holding"],
                "Capacity (units)": p["capacity"],
                "Test year mean / model": p["fit_check"]["test"]["mean_ratio"],
                "Test days in 80% interval": f"{100 * p['fit_check']['test']['coverage_80']:.1f}%",
            }
            for p in products
        ]
    )
    text = [
        "# Results: inventory control with reinforcement learning",
        "",
        f"Data: {selection['data']}. Test: {selection['episodes']['test']} simulated years per product; "
        f"replay: the real test year with {selection['episodes']['replay']} lead time draws.",
        "",
        "## Products and demand model",
        "",
        product_table.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Policies",
        "",
        "RL rows are the seed with the lowest validation cost. "
        "Against (s, S): mean paired difference of the test cost.",
        "",
        main.to_markdown(index=False, floatfmt=".1f"),
        "",
        "## Spread over seeds",
        "",
        markdown_table(spread),
        "",
        "`beats_s_S`: seeds whose paired 95% interval of the test cost difference lies below zero.",
        "",
        f"## Reward shaping ({SHAPING_PRODUCT} product, PPO)",
        "",
        markdown_table(shaping),
        "",
        *[f"* `{k}`: {v}" for k, v in SHAPING.items()],
        "",
    ]
    validation, dp = dp_lines(artifacts)
    text += validation
    (reports / "results.md").write_text("\n".join(text))
    results = {
        "data": selection["data"],
        "products": {
            p["product"]: {
                "series": p["series_id"],
                "policies": {
                    r.kind: {
                        "policy": r.policy,
                        "test_cost": round(r.test_cost, 2),
                        "test_cost_interval": [round(r.test_cost_low, 2), round(r.test_cost_high, 2)],
                        "test_against_s_S": None
                        if np.isnan(r.test_diff_share)
                        else round(r.test_diff_share, 4),
                        "test_fill_rate": round(r.test_fill_rate, 4),
                        "replay_cost": round(r.replay_cost, 2),
                    }
                    for r in rows[rows["product"] == p["product"]].itertuples()
                },
            }
            for p in products
        },
        "shaping": {
            r.shaping: {"seeds": int(r.seeds), "test_cost": round(r.test_cost, 2)}
            for r in shaping.dropna(subset=["seeds"]).itertuples()
        },
        "dp_validation": dp,
    }
    (reports / "results.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--validation-episodes", type=int, default=50)
    args = parse_args(parser, __file__)
    apply_style()
    figures = args.reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    products = load_products(args.data_dir)
    demand = load_demand(args.data_dir)
    summary, selection = load(args.artifacts)
    rows = main_rows(summary, selection)
    spread = seed_spread(summary)
    figure_demand(products, demand, figures / "demand_fit.png")
    figure_learning(
        products,
        args.artifacts,
        summary,
        curves(args.artifacts, summary),
        args.validation_episodes,
        figures / "learning_curves.png",
    )
    figure_costs(
        rows,
        figures / "test_costs.png",
        "test",
        "Test cost per year on 'simulated' years of the fitted demand (95% CI)".replace("'", ""),
    )
    figure_costs(
        rows,
        figures / "replay_costs.png",
        "replay",
        "Replay of the real test year: cost per year (95% CI over lead time draws)",
    )
    figure_seeds(summary, figures / "seeds.png")
    figure_shaping(summary, figures / "shaping.png")
    figure_policy_map(args.artifacts, products, selection, figures / "policy_map.png")
    write_results(rows, spread, summary, products, selection, args.reports, args.artifacts)
    print(f"wrote {args.reports / 'results.md'} and {len(list(figures.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
