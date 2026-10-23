# Data science portfolio

Thirteen projects, one per task type. Each solves one concrete problem on a public dataset and shows the full path from raw data to a reproducible result.

| # | Project | Task type | Status | Headline result |
| --- | --- | --- | --- | --- |
| 1 | [Credit default risk](projects/p01_loan_default) | Predictive modeling on tabular data | complete | test ROC AUC 0.782 (baseline 0.747); 34.2 million NT dollars profit at the chosen threshold |

## Shared conventions

* **README first:** problem, dataset and license, approach, headline result, and how to reproduce in three commands or fewer.
* **Reproducibility:** one `pyproject.toml` with pinned dependencies per project, a fixed seed, and a `Makefile` with `data`, `train` and `evaluate` targets.
* **Data:** downloaded by a script and never committed. Only synthetic fixtures enter the repository.
* **Quality gates:** ruff and pytest in GitHub Actions; each project job runs only when its folder or `common/` changes.
* **Tracking:** MLflow, stored locally.
* **Honest evaluation:** a simple baseline next to the main model, with the metric chosen for the business question.

## Layout

```
common/                 shared helpers: seed, metrics, plotting style
projects/
  p01_loan_default/     own README, pyproject.toml, Makefile, src/, tests/, notebooks/, reports/
.github/workflows/      one CI workflow
```

## License

Code: MIT (see `LICENSE`). Each dataset keeps its own license, stated in its project README.
