# A/B test readout tables

Data: `hillstrom-2008`, 64,000 customers.

## Randomization checks

Sample ratio: arm counts {'control': 21306, 'mens': 21307, 'womens': 21387}; chi square 0.203, p = 0.904 against the planned equal split.

Largest absolute standardized mean difference of a pre treatment feature: 0.0137 (adequate balance is below 0.1).

| feature              |   control_mean |   smd_mens |   smd_womens |
|:---------------------|---------------:|-----------:|-------------:|
| recency              |         5.7497 |     0.0068 |       0.0052 |
| history              |       240.883  |     0.0076 |       0.0065 |
| mens                 |         0.5532 |    -0.0046 |      -0.0086 |
| womens               |         0.5476 |     0.0076 |       0.0049 |
| newbie               |         0.502  |    -0.0009 |       0.0026 |
| log_history          |         5.0048 |    -0.0012 |       0.0065 |
| zip_surburban        |         0.4518 |    -0.0117 |      -0.0011 |
| zip_rural            |         0.1473 |     0.0137 |       0.004  |
| channel_web          |         0.4399 |     0.011  |      -0.0051 |
| channel_multichannel |         0.1223 |    -0.0042 |      -0.0053 |

## Effects against control

Visit and conversion in percentage points (pp), spend in dollars per customer, over the two weeks after the send. Intervals use Welch standard errors; p values are Holm adjusted over the six tests.

| arm    | outcome    |   control mean |   arm mean | effect (95% CI)              | relative lift   |   p (Holm) |
|:-------|:-----------|---------------:|-----------:|:-----------------------------|:----------------|-----------:|
| mens   | visit      |         10.617 |     18.276 | 7.66 pp (7.00 pp to 8.32 pp) | 72.1%           |   8.2e-112 |
| mens   | conversion |          0.573 |      1.253 | 0.68 pp (0.50 pp to 0.86 pp) | 118.8%          |   6e-13    |
| mens   | spend      |          0.653 |      1.423 | $0.77 ($0.49 to $1.05)       | 117.9%          |   3.5e-07  |
| womens | visit      |         10.617 |     15.14  | 4.52 pp (3.89 pp to 5.16 pp) | 42.6%           |   1.2e-43  |
| womens | conversion |          0.573 |      0.884 | 0.31 pp (0.15 pp to 0.47 pp) | 54.3%           |   0.00031  |
| womens | spend      |          0.653 |      1.077 | $0.42 ($0.17 to $0.68)       | 65.0%           |   0.0011   |

## Power

Minimum detectable effect at 80% power and a two sided level of 0.05, with the arm sizes of this test. The last column is the number of customers per arm a new test would need to detect the observed effect.

| arm    | outcome    |   customers in arm |   minimum detectable effect |   observed effect |   n per arm needed |
|:-------|:-----------|-------------------:|----------------------------:|------------------:|-------------------:|
| mens   | visit      |              21307 |                       0.836 |             7.659 |                254 |
| mens   | conversion |              21307 |                       0.205 |             0.681 |               1931 |
| mens   | spend      |              21307 |                       0.315 |             0.77  |               3557 |
| womens | visit      |              21387 |                       0.835 |             4.523 |                729 |
| womens | conversion |              21387 |                       0.205 |             0.311 |               9235 |
| womens | spend      |              21387 |                       0.314 |             0.424 |              11703 |

## Variance reduction

CUPED uses past year spend (`history`) as the pre period covariate. Regression adjustment uses all pre treatment features, interacted with the arm (Lin, 2013), with HC2 standard errors.

| arm    | outcome    |   corr. with past spend |    SE raw |   SE CUPED |   SE regression | variance cut CUPED   | variance cut regression   |
|:-------|:-----------|------------------------:|----------:|-----------:|----------------:|:---------------------|:--------------------------|
| mens   | visit      |                   0.065 | 0.003386  |  0.003378  |       0.003336  | 0.44%                | 2.93%                     |
| womens | visit      |                   0.065 | 0.003234  |  0.003228  |       0.003189  | 0.40%                | 2.81%                     |
| mens   | conversion |                   0.029 | 0.0009209 |  0.0009205 |       0.0009199 | 0.09%                | 0.22%                     |
| womens | conversion |                   0.029 | 0.0008227 |  0.0008224 |       0.0008221 | 0.06%                | 0.14%                     |
| mens   | spend      |                   0.022 | 0.1452    |  0.1452    |       0.1451    | 0.04%                | 0.22%                     |
| womens | spend      |                   0.022 | 0.1303    |  0.1303    |       0.1304    | 0.04%                | -0.14%                    |
