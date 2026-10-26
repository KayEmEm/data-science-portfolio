"""Collect every answer run into one results table, a cost and latency table, and a figure."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.config import parse_args
from portfolio_common.plotting import apply_style

from filings_qa.config import QUESTIONS_PATH, REPORTS_DIR, RUNS_DIR
from filings_qa.scoring import free_scores


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """Read each run, preferring its graded copy when one exists and is complete."""
    frames = []
    for path in sorted(runs_dir.glob("*.jsonl")):
        if path.name.endswith(".judged.jsonl"):
            continue
        judged = path.with_suffix(".judged.jsonl")
        source = path
        if judged.exists():
            n_run = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            n_judged = sum(1 for line in judged.read_text(encoding="utf-8").splitlines() if line.strip())
            source = judged if n_judged == n_run else path
        lines = source.read_text(encoding="utf-8").splitlines()
        frames.append(pd.DataFrame([json.loads(line) for line in lines if line.strip()]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _share(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.astype(bool).mean()) if len(values) else float("nan")


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, g in scored.groupby("run", sort=True):
        judged = "judge_correctness" in g and g["judge_correctness"].notna().all()
        row = {
            "run": run,
            "contexts": g["contexts"].iloc[0],
            "generator": g["generator"].iloc[0],
            "model": g["model"].iloc[0],
            "questions": len(g),
            "accuracy (judge)": float((g["judge_correctness"] == "correct").mean()) if judged else np.nan,
            "declined (judge)": float((g["judge_correctness"] == "declined").mean()) if judged else np.nan,
            "supported (judge)": _share(g["judge_supported"]) if judged else np.nan,
            "figure match (metrics questions)": _share(g["numeric_match"]),
            "cites an evidence page": float(g["cites_evidence"].mean()),
            "input tokens per question": float(g["input_tokens"].mean()),
            "output tokens per question": float(g["output_tokens"].mean()),
            "USD per question": float(g["cost_usd"].mean()),
            "median latency s": float(g["latency_s"].median()),
            "p90 latency s": float(g["latency_s"].quantile(0.9)),
            "USD total (answers)": float(g["cost_usd"].sum()),
            "USD total (grading)": float(g["judge_cost_usd"].sum()) if "judge_cost_usd" in g else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_accuracy(summary: pd.DataFrame, path: Path) -> None:
    data = summary.dropna(subset=["accuracy (judge)"])
    if data.empty:
        return
    apply_style()
    labels = [f"{r.model}\n{r.contexts}" for r in data.itertuples(index=False)]
    order = np.argsort(data["accuracy (judge)"].to_numpy())
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 1.5))
    ax.barh(np.array(labels)[order], data["accuracy (judge)"].to_numpy()[order], label="correct")
    ax.barh(
        np.array(labels)[order],
        data["supported (judge)"].to_numpy()[order],
        height=0.3,
        label="supported by cited passages",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the 150 FinanceBench questions")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    args = parse_args(parser, __file__)

    runs = load_runs(args.runs_dir)
    if runs.empty:
        raise SystemExit(f"no runs in {args.runs_dir}")
    summary = summarize(free_scores(runs, pd.read_parquet(args.questions)))
    (args.reports / "figures").mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(args.reports / "qa_results.csv", index=False)
    (args.reports / "qa_results.md").write_text(
        summary.drop(columns=["run"]).round(3).to_markdown(index=False) + "\n", encoding="utf-8"
    )
    plot_accuracy(summary, args.reports / "figures" / "qa_accuracy.png")
    print(summary.drop(columns=["run"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
