# portfolio-common

Small shared helpers used by the projects in this repository:

| Module | Purpose |
| --- | --- |
| `portfolio_common.seed` | One function that seeds Python, NumPy and the environment hash seed |
| `portfolio_common.evaluation` | Binary classification metrics in one dictionary (ROC AUC, PR AUC, Brier score) |
| `portfolio_common.plotting` | One plotting style for all report figures |
| `portfolio_common.sample` | Seeded samples of whole units (rows, users, series, time steps) behind the `--sample` option of each data step |
| `portfolio_common.config` | The YAML control interface: option values of each command and field values of each settings class, from the project `config.yaml`, the files in `PORTFOLIO_CONFIG` and `--config` |

Each project installs this package in editable mode from its own `Makefile`.
