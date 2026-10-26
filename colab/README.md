# Run the projects in Google Colab

[`run_project.ipynb`](run_project.ipynb) runs any of the 13 projects in Colab, with Google Drive as the only storage:
[open it in Colab](https://colab.research.google.com/github/KayEmEm/data-science-portfolio/blob/main/colab/run_project.ipynb).
Each run clones the repository onto the local disk of the Colab machine, installs the project, copies its raw files from Drive, runs it, and copies the outputs back to Drive. The same notebook and code serve the free tier and Colab Pro; only the profile changes.

## Pinned version

The notebook runs the code at the tag `colab-v1`, not the latest `main`, so a change reaches Colab only after it is reviewed and the pin moves. To move the pin: create a new tag (`colab-v2`) on the reviewed commit of `main`, and change the default of `REF` in the notebook in the same step. `REF` accepts any branch, tag or commit for a single run.

## Drive layout

```
MyDrive/portfolio/
  raw/<project folder>/   input files, uploaded once (table below)
  out/<project folder>/   artifacts/ and reports/ of the runs, copied every 10 minutes and at the end
```

Every file is downloaded by hand and uploaded once. At the start of a run the files are copied to `data/raw/` of the project, and the data step reads them there instead of downloading them; p04, p10 and p12, and the full run of p05, check the SHA256 checksum of each file. A file that is missing is downloaded from its source, where the project has one.

| Project | Files in `raw/<project folder>/` |
| --- | --- |
| p01_loan_default | `default_of_credit_card_clients.zip` (UCI) |
| p02_filings_qa | `financebench_open_source.jsonl` and `financebench_document_information.jsonl` (from `data/`) and the folder `pdfs/`, from the [FinanceBench zip at the pinned commit](https://github.com/patronus-ai/financebench/archive/cc39aeb4afdf33909ee1412188bf89035950c2eb.zip); only the 84 filings that the questions cite are read |
| p03_taxi_demand | `trips/yellow_tripdata_2024-01.parquet` to `-12.parquet` ([TLC trip data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)), [`taxi_zone_lookup.csv`](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv), and `weather_central_park.csv`: the result of [this NOAA query](https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00094728&startDate=2024-01-01&endDate=2024-12-31&dataTypes=PRCP,SNOW,TMAX,TMIN&units=metric&format=csv) |
| p04_email_uplift | `hillstrom.csv.gz`: [this file](https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz), renamed |
| p05_retail_forecasting | [`m5.zip`](https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip) |
| p06_road_damage | `RDD2022_Czech.zip`, `RDD2022_India.zip`; GPU runtime |
| p07_movie_recommender | `ml-32m.zip` (GroupLens) |
| p08_network_anomaly | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |
| p09_customer_segments | `online_retail_II.xlsx` (UCI) |
| p10_delivery_planning | the 112 `.vrp` and `.sol` files of the folder `VRPTW/Solomon` in the [PyVRP Instances zip at the pinned commit](https://github.com/PyVRP/Instances/archive/7474b068a8effa7f39448b241314b615c9271525.zip), without subfolders |
| p11_illicit_transactions | `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` |
| p12_inventory_control | [`m5.zip`](https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip) (the same file as p05) |
| p13_adversarial_robustness | `credit/` (the p01 zip), `flows/` (the two p08 files), `images/RDD2022_Czech.zip`, `detector/best.pt` (p06 weights); runs on CPU |

## Profiles

[`profiles/free.yaml`](profiles/free.yaml) runs the `sample` pipeline of each project on the share set in its Makefile. [`profiles/pro.yaml`](profiles/pro.yaml) runs the same pipelines on the full data (`SAMPLE=1`), for the larger memory and longer sessions of Colab Pro. A profile sets the Makefile targets, the share, and how often outputs are copied to Drive; an entry under `projects` changes these for one project. A project may also hold a `colab.yaml`, which [`run_project.py`](run_project.py) adds to `PORTFOLIO_CONFIG` (p06: train on GPU 0 with 2 workers).

## Limits

* Colab keeps its own Python, CUDA and PyTorch builds; the notebook installs the exact pins of the project except `torch` and `torchvision`. A pin that conflicts with a package Colab provides shows on the first run.
* A session that ends during a run keeps what was last copied to Drive. With `RESUME`, those outputs are copied back first: steps that resume (p02 generation, p06 training from its last epoch, p10 and p12 training) continue; other steps run again from their start.
* The dashboards (p03, p10), the scoring API and Docker image (p01), the Airflow DAG (p03) and the Neo4j export (p11) run outside Colab.
* `drive.mount` gives the notebook access to the whole Drive, not only `MyDrive/portfolio`.
