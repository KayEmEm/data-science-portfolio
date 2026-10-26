"""Animation of one simulated test year: stock, orders and cumulative cost of (s, S) and the best RL policy.

Both policies face the same demand and lead times (the first test scenario). One frame per week.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from portfolio_common.config import parse_args
from portfolio_common.plotting import apply_style

from inventory_rl.config import ARTIFACTS_DIR, REPORTS_DIR, SHAPING_PRODUCT
from inventory_rl.report import COLORS, LABELS


def load_trace(artifacts: Path, product: str) -> tuple[dict, str]:
    selection = json.loads((artifacts / "evaluation" / "selection.json").read_text())
    best = selection["products"][product]["best_rl"]
    raw = np.load(artifacts / "evaluation" / f"trace_{product}.npz")
    traces = {}
    for key in raw.files:
        policy, field = key.split("__")
        traces.setdefault(policy, {})[field] = raw[key][0]
    return {"s_S": traces["s_S"], best: traces[best]}, best


def animate(artifacts: Path, product: str, path: Path, fps: int = 8) -> Path:
    apply_style()
    traces, best = load_trace(artifacts, product)
    kinds = {"s_S": "s_S", best: best.split("_")[1]}
    days = len(traces["s_S"]["demand"])
    t = np.arange(1, days + 1)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    peak = max(float(tr["on_hand_end"].max() + tr["arrived"].max()) for tr in traces.values())
    total = max(float(np.cumsum(tr["cost"])[-1]) for tr in traces.values())

    def draw(frame: int) -> None:
        end = min(days, 7 * (frame + 1))
        top.clear()
        bottom.clear()
        for policy, trace in traces.items():
            color = COLORS[kinds[policy]]
            label = LABELS[kinds[policy]]
            top.step(
                t[:end],
                trace["on_hand_end"][:end],
                where="post",
                color=color,
                lw=1.5,
                label=f"{label}: on hand",
            )
            ordered = np.flatnonzero(trace["order"][:end] > 0)
            top.scatter(t[ordered], np.full(len(ordered), peak * 1.04), marker="v", s=22, color=color)
            lost = np.flatnonzero(trace["lost"][:end] > 0)
            top.scatter(t[lost], np.zeros(len(lost)), marker="x", s=18, color=color)
            cumulative = np.cumsum(trace["cost"][:end])
            bottom.plot(t[:end], cumulative, color=color, lw=2, label=f"{label}: {cumulative[-1]:.0f} $")
        top.set_xlim(1, days)
        top.set_ylim(-0.05 * peak, peak * 1.1)
        top.set_ylabel("units on hand")
        top.set_title(
            f"One simulated year, {product} product: day {end} (triangles: orders, crosses: lost sales)",
            fontsize=9,
        )
        top.legend(fontsize=8, loc="upper right")
        bottom.set_ylim(0, total * 1.05)
        bottom.set_xlabel("day of the year")
        bottom.set_ylabel("cumulative cost ($)")
        bottom.legend(fontsize=8, loc="upper left")

    frames = int(np.ceil(days / 7))
    animation = FuncAnimation(fig, draw, frames=frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--product", default=SHAPING_PRODUCT)
    args = parse_args(parser, __file__)
    print(animate(args.artifacts, args.product, args.reports / "figures" / "simulated_year.gif"))


if __name__ == "__main__":
    main()
