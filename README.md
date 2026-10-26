# Data science portfolio

Thirteen projects, one per task type. Each solves one concrete problem on a public dataset and shows the full path from raw data to a reproducible result.

| # | Project | Task type | Status | Headline result |
| --- | --- | --- | --- | --- |
| 1 | [Credit default risk](projects/p01_loan_default) | Predictive modeling on tabular data | complete | test ROC AUC 0.782 (baseline 0.747); 34.2 million NT dollars profit at the chosen threshold |
| 2 | [Question answering over filings](projects/p02_filings_qa) | NLP and LLM applications | in progress | dense retrieval recall@8 0.553 (BM25 0.360); answer accuracy pending |
| 3 | [New York taxi demand](projects/p03_taxi_demand) | Exploratory analysis and reporting | complete | 38.7 million trips in a DuckDB warehouse; dashboard mock up and five findings (airports: 9.8 percent of trips, 27.1 percent of revenue) |
| 4 | [Email campaign uplift](projects/p04_email_uplift) | Experimentation and causal inference | complete | mens email: +$0.77 spend per customer (95% CI 0.49 to 1.05); mailing everyone beats four uplift model policies; observational estimate without past spend is biased by +2.0 pp |
| 5 | [Retail demand forecasting](projects/p05_retail_forecasting) | Time series forecasting | complete | M5 test WRMSSE 0.562 with a global LightGBM model (seasonal naive 0.847); 95% intervals cover 94.1% of item x store days |
| 6 | [Road damage detection](projects/p06_road_damage) | Computer vision | complete | RDD2022 Czech and India test mAP50 0.288 with YOLO11n (frozen backbone baseline 0.214); ONNX 26.6 ms per image on CPU |
| 7 | [Movie recommendations](projects/p07_movie_recommender) | Recommendation and ranking | complete | MovieLens 32M test NDCG at 10 0.134 with ALS and two tower candidates and a LightGBM ranker (90 day popularity baseline 0.095) |
| 8 | [Network intrusion as anomaly detection](projects/p08_network_anomaly) | Anomaly detection | complete | UNSW-NB15 test recall 0.22 at 1% false alarm rate without labels (Isolation Forest; supervised LightGBM 0.93); the autoencoder finds other attack types and loses most recall when 1% of its training data are attacks |
| 9 | [Customer segments](projects/p09_customer_segments) | Clustering and segmentation | complete | Online Retail II: three stable k means segments (bootstrap adjusted Rand 0.95); active repeat buyers are 29% of customers and 82% of spend; the ten rule RFM grid separates next quarter's spend better (adjusted eta squared 0.262 against 0.214) |
| 10 | [Delivery routes and driver shifts](projects/p10_delivery_planning) | Mathematical optimization | complete | Solomon VRPTW: OR-Tools 0.9% above the best known distance in 300 s per instance (first solution heuristics 53 to 102%); driver shift MIP optimal in about 1 s for 4 days; cost aware routing lowers the week's cost by 5.6% |
| 11 | [Illicit transactions in a payment network](projects/p11_illicit_transactions) | Graph analytics | complete | Elliptic test illicit F1 0.799 with LightGBM on transaction features (GraphSAGE 0.697, GCN 0.458, baseline 0.428); every model drops to F1 0.03 or less after the dark market shutdown; illicit transactions cluster in Louvain communities |
| 12 | [Inventory control](projects/p12_inventory_control) | Reinforcement learning | complete | M5 store CA_1, three products (sample run): tuned (s, S) costs least for all three; PPO costs 12 to 19% more and SAC 12 to 50% more; (s, S) is 3.6% above the exact optimum on the slow product with a fixed lead time |
| 13 | [Adversarial robustness](projects/p13_adversarial_robustness) | Adversarial ML (cross cutting) | complete | sample run: a borrowed payment of up to 10% of the limit plus free self reported fields turns 24% of credit rejections into approvals (6% with two defenses); a changed packet TTL, padding and delays evade 92% of LightGBM flow detections (3% after adversarial training); PGD at 2/255 brings detector mAP50 from 0.50 to 0.00 |

## Shared conventions

* **README first:** problem, dataset and license, approach, headline result, and how to reproduce in three commands or fewer.
* **Reproducibility:** one `pyproject.toml` with pinned dependencies per project, a fixed seed, and a `Makefile` with `data`, `train` and `evaluate` targets.
* **Data:** downloaded by a script and never committed. Only synthetic fixtures enter the repository.
* **Sample runs:** `make sample SAMPLE=<share>` runs a project's real pipeline on a seeded share of whole units (clients, filings, series, users, time step graphs), through `portfolio_common.sample`. Outputs stay under `artifacts/sample/`; the committed results come from the full data.
* **Settings in YAML:** each project keeps every option of its commands and every field of its settings classes in `config.yaml`, one key per command (`train`, `evaluate`, ...) and `settings` for the classes. A run takes changes from `--config <file>` or from the files listed in `PORTFOLIO_CONFIG`; options on the command line win over all files. `--print-config` shows the resolved values of a command. Relative paths in a file are relative to its folder. Options whose default is computed in code (from the environment, the processor count or a settings class) are not listed, but a file may still set them.
* **Quality gates:** ruff and pytest in GitHub Actions; each project job runs only when its folder or `common/` changes.
* **Tracking:** MLflow, stored locally.
* **Honest evaluation:** a simple baseline next to the main model, with the metric chosen for the business question.

## Layout

```
common/                 shared helpers: seed, metrics, plotting style, samples, YAML settings
projects/
  p01_loan_default/     own README, config.yaml, pyproject.toml, Makefile, src/, tests/, notebooks/, reports/
  p02_filings_qa/
  p03_taxi_demand/
  p04_email_uplift/
  p05_retail_forecasting/
  p06_road_damage/      AGPL-3.0, because it uses Ultralytics YOLO
  p07_movie_recommender/
  p08_network_anomaly/
  p09_customer_segments/
  p10_delivery_planning/
  p11_illicit_transactions/
  p12_inventory_control/
  p13_adversarial_robustness/  AGPL-3.0, because it uses Ultralytics YOLO
.github/workflows/      one CI workflow
```

## License

Code: MIT (see `LICENSE`), except `projects/p06_road_damage` and `projects/p13_adversarial_robustness`, which are AGPL-3.0 (see their `LICENSE` files). Each dataset keeps its own license, stated in its project README.
