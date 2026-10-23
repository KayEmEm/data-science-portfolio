"""One figure style for every report in the portfolio."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f"]


def apply_style() -> None:
    """Set consistent fonts, grid and colors for report figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (7, 4.5),
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
            "font.size": 10,
        }
    )
