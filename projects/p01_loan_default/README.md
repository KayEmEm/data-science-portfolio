# Project 1: Loan default risk

**Status: code complete and tested on synthetic tables. Results on the real data are pending, because the dataset download is not yet confirmed.** No number in this README comes from the real data yet.

## Problem

A lender must decide which applicants to approve. Approving a borrower who later defaults loses part of the loan; rejecting a borrower who would repay loses the interest margin. This project predicts the probability of default for each applicant and sets the approval threshold that gives the highest profit under a stated cost matrix.

## Dataset and license

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) on Kaggle: seven related tables on about 300,000 applicants.

| Table | Grain | Used for |
| --- | --- | --- |
| `application_train` | one row per applicant | target, application features |
| `bureau`, `bureau_balance` | credits at other lenders, monthly status | bureau history |
| `previous_application` | earlier applications to the same lender | approval history |
| `POS_CASH_balance` | monthly point of sale and cash loan balance | days past due |
| `installments_payments` | each instalment on earlier loans | payment delay and partial payment |
| `credit_card_balance` | monthly card balance | utilization |

**License: to be confirmed** from the competition rules before the first real run. The raw data is never committed to this repository; `make data` downloads it with your own Kaggle account.

## Approach

1. **Schema check** (`data.py`): each table must contain the columns listed in `config.REQUIRED_COLUMNS`, or the run stops with a clear message.
2. **Features in SQL** (`sql/*.sql`, run by DuckDB in `features.py`): one query per source table aggregates it to one row per applicant; the results join to the application table. Examples: debt to credit ratio at the bureau, share of late instalments, card utilization, share of refused earlier applications.
3. **Models** (`modeling.py`): logistic regression as the baseline; LightGBM and XGBoost with stratified five fold cross validation and early stopping. The final tree count is the mean best iteration across folds.
4. **Calibration**: isotonic regression fitted on the out of fold predictions, so the calibrator never sees a prediction on a row the model trained on.
5. **Threshold** (`threshold.py`): approve when the calibrated probability of default is below the threshold. Approving a good borrower earns `margin × credit`; approving a defaulter loses `loss_given_default × credit`. The threshold is chosen on out of fold predictions of the training split only, and compared with the break even rule `margin / (margin + loss_given_default)`.
6. **Model selection**: the model with the highest out of fold profit. The held out test split (20 percent) is scored once, for the report.
7. **Explanations** (`evaluate.py`): exact TreeSHAP values from the boosting library, shown as a SHAP summary plot.
8. **Fairness check** (`fairness.py`): approval rate, share of good borrowers rejected, share of defaulters approved and ROC AUC per age band.
9. **Tracking**: every model run is logged to a local MLflow store under `artifacts/`.
10. **Serving** (`api.py`, `Dockerfile`): a FastAPI service that returns the probability of default and the decision.

### Cost assumptions

The cost matrix is an assumption, not a value from the data:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `loss_given_default` | 0.45 | share of the credit lost when a borrower defaults |
| `margin` | 0.10 | share of the credit earned when a borrower repays |

Both can be changed: `python -m loan_default.train --lgd 0.6 --margin 0.08`.

## Results

Pending the first run on the real data. The run will fill:

* `reports/model_comparison.md`: ROC AUC, PR AUC and Brier score (raw and calibrated) per model, with threshold, approval rate and test profit
* `reports/figures/threshold_analysis.png`: profit and approval rate against the threshold
* `reports/figures/calibration.png`: reliability curve before and after calibration
* `reports/figures/explanations.png`: SHAP summary
* `reports/fairness_age.md`: metrics per age band

## Reproduce

From this folder, with Python 3.11 or later:

```bash
make install            # this project and the shared common package
make all                # data, features, train, evaluate (needs Kaggle credentials)
make smoke              # the same pipeline on synthetic tables, no download needed
```

`make data` needs the Kaggle CLI configured (`KAGGLE_USERNAME` and `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`) and the competition rules accepted with that account.

`make smoke` writes everything under `artifacts/smoke/`. Its numbers come from random synthetic tables and say nothing about credit risk.

### Scoring API

```bash
make train              # writes artifacts/model.joblib
make api                # http://localhost:8000/docs
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d @artifacts/example_request.json
```

Docker: `make docker`, then `docker run -p 8000:8000 loan-default-api`.

## Layout

```
src/loan_default/   config, data, features, sql/, modeling, threshold, fairness, train, evaluate, api, synthetic
tests/              unit tests and an end to end test on synthetic tables
notebooks/          narrative exploration (after the first real run)
reports/            comparison tables and figures
```

## Model card

To be written after the first real run: intended use, training data, metrics, calibration, fairness results and limits.

## Cost

No paid service used so far: 0 CAD.
