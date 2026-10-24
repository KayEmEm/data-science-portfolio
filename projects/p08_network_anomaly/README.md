# Project 8: Network intrusion as anomaly detection

Flag unusual network flows in UNSW-NB15 without labels for known attacks, then measure how well each detector finds the attacks, per attack type, at a fixed false alarm rate.

## Headline result

Pending: the run on UNSW-NB15 follows. The pipeline, tests and CI run on synthetic flows.

## Dataset and license

[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) (Cyber Range Lab of UNSW Canberra, 2015): network flows from a test bed that mixes normal traffic with nine attack types (Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms). Each flow has 49 columns: addresses and ports, protocol, service and state, sizes, packet counts, timing, TCP features, connection counts over recent flows, the attack type and a 0/1 label.

Cite: N. Moustafa and J. Slay, 2015, UNSW-NB15: a comprehensive data set for network intrusion detection systems, Military Communications and Information Systems Conference (MilCIS).

**License:** to my knowledge the dataset page allows free use for academic research and asks for the citation above, and it does not grant redistribution. The page could not be opened from the build environment, so these terms were not checked again there. This repository does not include the data.

The release is a shared folder without stable direct file links, so `make data` reads the files from `data/raw/` (or `SOURCE=<folder>`):

* **full records** (used here): `UNSW-NB15_1.csv` to `UNSW-NB15_4.csv` and `NUSW-NB15_features.csv` from the folder "CSV Files";
* **official partition** (also read): `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` from the folder "Training and Testing Sets".

## Approach

1. **Data** (`data.py`): the files are read and checked (49 columns, labels 0 or 1, one spelling per attack type, cells that are not numbers counted and set to 0). Counts are compared with the documented ones and recorded in `data/audit.json`.
2. **Split:** the full records are split 60 / 20 / 20 into train, validation and test, stratified by attack type. The official partition keeps its test file and splits its training file 80 / 20.
3. **Features** (`features.py`): 36 numeric flow features (signed log, then standard scaling) and one hot columns for the 12 most frequent protocol, service and state values. Addresses, ports, times and TCP sequence numbers are left out: they identify test bed hosts and sessions, not the behavior of a flow.
4. **Training sample without labels:** the unsupervised detectors are fitted on 200,000 training flows with 1 percent attacks, as a log from a network that is attacked now and then. The label is used only to build this sample and to score the results.
5. **Detectors** (`detectors.py`): a robust z-score baseline (mean distance from the median in interquartile ranges), Isolation Forest, Local Outlier Factor (on 20,000 flows), HBOS, ECOD and COPOD from PyOD, a dense autoencoder in PyTorch (`autoencoder.py`: 64, 32, 8 units, early stopping on a held out tenth of the training sample), and an ensemble of Isolation Forest, ECOD and the autoencoder (mean of score quantiles). A supervised LightGBM model trained with labels is the upper reference.
6. **Threshold and metrics** (`metrics.py`, `evaluate.py`): each detector's threshold flags 1 percent of the normal validation flows. On the test flows: PR AUC, ROC AUC, recall, false alarm rate and precision at that threshold, precision of the top 100 and 1,000 alerts, recall per attack type, and 95 percent bootstrap intervals. A second threshold uses no labels at all: the top 1 percent of training sample scores. The detector is selected on validation PR AUC.
7. **Contamination study:** Isolation Forest and the autoencoder are refitted on samples with 0, 5 and 20 percent attacks.
8. **Ablation:** four detectors without the time to live features (`sttl`, `dttl`, `ct_state_ttl`), which can separate test bed networks rather than behavior.
9. **Triage example** (`report.py`): the top 20 test alerts of the selected detector, each with the three features that deviate most from the training sample.
10. **Tracking:** scores are logged to a local MLflow store under `artifacts/` (`mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db`).

## Reproduce

From this folder, with Python 3.11 or later:

```bash
make install                      # this project and the shared common package
make all SOURCE=path/to/csv/files # or place the files in data/raw/
make smoke                        # the same pipeline on synthetic flows, no data needed
```

`make train` stores every finished detector under `artifacts/` and skips it when run again, and the autoencoder saves each epoch, so a stopped run resumes. `make smoke` writes everything under `artifacts/smoke/`; its numbers come from synthetic data and say nothing about UNSW-NB15.

## Layout

```
src/flow_anomaly/  config, data, features, detectors, autoencoder, metrics,
                   train, evaluate, report, synthetic
tests/             unit tests (data checks, split, metrics, detectors, resume) and an end to end test
reports/           results.md, results.json, triage_example.md, figures
```

## Cost

No paid service used: 0 CAD.
