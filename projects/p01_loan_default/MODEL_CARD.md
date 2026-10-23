# Model card: credit default risk (project 1)

## Model

LightGBM gradient boosted trees (109 trees, learning rate 0.03, 31 leaves), followed by isotonic calibration. Output: probability that the client defaults on the payment of the following month. Decision rule: approve when the probability is below 0.170.

## Intended use

A portfolio demonstration of credit risk modeling: feature engineering, calibrated probabilities and a cost based decision threshold. **Not intended for real credit decisions.** The data covers credit card clients in Taiwan in 2005, and the cost matrix is illustrative.

## Training data

UCI Default of Credit Card Clients (CC BY 4.0, Yeh 2009, DOI 10.24432/C55S3H). 30,000 clients; 24,000 for training and cross validation, 6,000 held out for testing, split with stratification on the target and seed 42.

## Features

Credit limit, sex, education, marital status, age, six months of repayment status, bill amounts and payment amounts, and the derived features in `src/loan_default/sql/features.sql`.

## Metrics on the held out test split

| Metric | Value |
| --- | ---: |
| ROC AUC | 0.7821 |
| PR AUC | 0.5460 |
| Brier score (calibrated) | 0.1346 |
| Approval rate at threshold | 0.6687 |
| Default rate among approved | 0.1142 |
| Default rate overall | 0.2212 |

Baseline (logistic regression): ROC AUC 0.7468, PR AUC 0.4913, Brier score 0.1425.

## Fairness

Metrics per age band at the chosen threshold, on the test split:

| Age band | Clients | Default rate | Approval rate | Good clients rejected | Defaulters approved | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| under 30 | 1,912 | 0.2233 | 0.6485 | 0.2552 | 0.3138 | 0.7839 |
| 30 to 39 | 2,198 | 0.2061 | 0.6997 | 0.2160 | 0.3753 | 0.7886 |
| 40 to 49 | 1,341 | 0.2327 | 0.6659 | 0.2410 | 0.3590 | 0.7710 |
| 50 to 59 | 473 | 0.2368 | 0.6237 | 0.2770 | 0.3036 | 0.7822 |
| 60 and over | 76 | 0.3026 | 0.6053 | 0.2830 | 0.3478 | 0.7445 |

The lowest approval rate relative to the overall rate is 0.905 (60 and over). That group has only 76 clients, so its rates carry wide uncertainty. Groups with a higher observed default rate have lower approval rates.

## Limits

* Sex, age and marital status are model inputs. In many jurisdictions such attributes may not be used in credit decisions; a real deployment would remove them and measure the effect.
* The data covers clients in Taiwan over six months in 2005. Performance on other populations or periods is unknown.
* Education and marital status contain codes that the dataset documentation does not describe (education 0, 5 and 6; marital status 0). They are kept as separate categories. Repayment status values −2 and 0 are also undocumented (the scale defines −1 as paid duly and 1 to 9 as months of delay); they are kept as numbers.
* Profit depends on the assumed loss given default (0.45) and margin (0.10), with the credit limit as the exposure.
