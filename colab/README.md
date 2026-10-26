# Run the projects in Google Colab

[`run_project.ipynb`](run_project.ipynb) runs any of the 13 projects in Colab, with Google Drive as the only storage:
[open it in Colab](https://colab.research.google.com/github/KayEmEm/data-science-portfolio/blob/main/colab/run_project.ipynb).
Each run clones the repository onto the local disk of the Colab machine, installs the project, copies its raw files from Drive, runs it, and copies the outputs back to Drive. The same notebook and code serve the free tier and Colab Pro; only the profile changes.

## Pinned version

The notebook runs the code at the tag `colab-v4`, not the latest `main`, so a change reaches Colab only after it is reviewed and the pin moves. To move the pin: create a new tag (`colab-v5`) on the reviewed commit of `main`, and change the default of `REF` in the notebook in the same step. `REF` accepts any branch, tag or commit for a single run.

## Drive layout

```
MyDrive/portfolio/
  raw/<project>/   input files, uploaded once: one subfolder per kind of data (table below)
  out/<project>/   artifacts/ and reports/ of the runs, copied every 10 minutes and at the end
```

`<project>` is the project number, for example `raw/p01/clients/`. A folder with the full project name (`raw/p01_loan_default/`) is also accepted when no folder with the number exists; `out/` then uses the full name as well.

Every file is downloaded by hand and uploaded once, into one subfolder per kind of data of the project (table below). The file names do not matter. At the start of a run the files are copied to `data/raw/` of the project and placed under the names the code reads, following [`raw_files.yaml`](raw_files.yaml):

1. A subfolder for one file must hold exactly one file of its kind (for example one `.csv` in `p11/edges/`), or one zip that holds it; with two or more, the run stops and names them. A subfolder for many files (`trips/`, `pdfs/`, `instances/`, `records/`) gives all its files of the kind, under their names from the source, because the code finds months, filings and instances by those names.
2. A zip is extracted when the project does not read the zip itself; the member with the expected name is taken first, wherever it sits in the zip.
3. Files placed outside the subfolders are still found: by the names of the original downloads, or inside any zip in the project folder (for example the Kaggle `archive.zip` of p11).
4. When data is still missing, the run stops before the install. The notebook shows the missing kinds with their subfolders and sources, and the same list is written to `out/<project>/missing_files.txt` on Drive; it is removed by the next complete run.

The data steps then read the files in `data/raw/` and download nothing. p04, p10 and p12, and the full run of p05, check the SHA256 checksum of each file, and p07 its MD5 checksum, so a wrong file stops the run. The targets `smoke`, `test`, `lint`, `install` and `clean` skip the check.

| Project | Subfolders of `raw/<project>/` and what each holds |
| --- | --- |
| p01 | `clients/`: the [UCI zip](https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip) |
| p02 | `questions/`: `financebench_open_source.jsonl`; `documents/`: `financebench_document_information.jsonl`; `pdfs/`: the filings (at least the 84 that the questions cite); all from the [FinanceBench zip at the pinned commit](https://github.com/patronus-ai/financebench/archive/cc39aeb4afdf33909ee1412188bf89035950c2eb.zip), which may also be placed in a subfolder as downloaded |
| p03 | `trips/`: the 12 monthly files of 2024 ([TLC trip data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)), with their names; `zones/`: [the zone table](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv); `weather/`: the result of [this NOAA query](https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00094728&startDate=2024-01-01&endDate=2024-12-31&dataTypes=PRCP,SNOW,TMAX,TMIN&units=metric&format=csv) |
| p04 | `customers/`: [the Hillstrom `.gz` file](https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz), not extracted |
| p05 | `m5/`: [`m5.zip`](https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip) |
| p06 | `czech/`, `india/`: the RDD2022 zip of the country; GPU runtime |
| p07 | `ratings/`: [`ml-32m.zip`](https://files.grouplens.org/datasets/movielens/ml-32m.zip) |
| p08 | `records/`: the full records `UNSW-NB15_1.csv` to `UNSW-NB15_4.csv`, with their names; `features/`: `NUSW-NB15_features.csv` (optional); from the [UNSW-NB15 page](https://research.unsw.edu.au/projects/unsw-nb15-dataset). The partition files `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`, placed in the project folder, are used when the full records are absent |
| p09 | `transactions/`: the [UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii) workbook, or its zip |
| p10 | `instances/`: the [PyVRP Instances zip at the pinned commit](https://github.com/PyVRP/Instances/archive/7474b068a8effa7f39448b241314b615c9271525.zip) as downloaded (only `VRPTW/Solomon` is used), or the 112 `.vrp` and `.sol` files of that folder |
| p11 | `features/`, `classes/`, `edges/`: one CSV each, from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |
| p12 | `m5/`: [`m5.zip`](https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip), the same file as p05 |
| p13 | `credit/`: the p01 zip; `flows_train/` and `flows_test/`: the two p08 partition files, as for the committed results, or else `flows_records/`: the four p08 full record files, with their names, and `flows_features/`: `NUSW-NB15_features.csv` (optional); `images/`: the Czech zip of p06; `detector/`: `best.pt` from the p06 run; runs on CPU |

## Profiles

[`profiles/free.yaml`](profiles/free.yaml) runs the `sample` pipeline of each project on the share set in its Makefile. [`profiles/pro.yaml`](profiles/pro.yaml) runs the same pipelines on the full data (`SAMPLE=1`), for the larger memory and longer sessions of Colab Pro. A profile sets the Makefile targets, the share, and how often outputs are copied to Drive; an entry under `projects` changes these for one project. A project may also hold a `colab.yaml`, which [`run_project.py`](run_project.py) adds to `PORTFOLIO_CONFIG` (p06: train on GPU 0 with 2 workers; p08: go on when the row counts differ from the documented ones, as for the full records).

Most projects write the reports of a run to `artifacts/sample/reports/`, so `out/<project>/reports/` keeps the committed reports. p12 and p13 write their reports to `reports/`, so for these two projects `out/<project>/reports/` holds the reports of the last run.

## Limits

* Colab keeps its own Python, CUDA and PyTorch builds; the notebook installs the exact pins of the project except `torch` and `torchvision`. A pin that conflicts with a package Colab provides shows on the first run.
* A session that ends during a run keeps what was last copied to Drive. With `RESUME`, those outputs are copied back first: steps that resume (p02 generation, p06 training from its last epoch, p10 and p12 training) continue; other steps run again from their start.
* The dashboards (p03, p10), the scoring API and Docker image (p01), the Airflow DAG (p03) and the Neo4j export (p11) run outside Colab.
* `drive.mount` gives the notebook access to the whole Drive, not only `MyDrive/portfolio`.
