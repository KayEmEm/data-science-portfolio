# Observational analysis with DoWhy

Each of 20 draws keeps about 21,307 customers of the mens email and control arms, as if a marketer had mailed recent, high value web customers more often (selection probability 0.03 to 0.98; 50% of kept customers mailed). Effects are means over draws; the spread is the standard deviation over draws (for the experiment, its standard error).

Full adjustment set: log_history, recency, channel_web, newbie. Incomplete set: recency, channel_web, newbie (past spend left out).

## Visit (percentage points)

| estimate                                         |   effect |   spread (sd) |   bias against experiment |
|:-------------------------------------------------|---------:|--------------:|--------------------------:|
| experiment (benchmark)                           |    7.659 |         0.339 |                     0     |
| naive difference                                 |   12.225 |         0.26  |                     4.566 |
| linear regression, incomplete adjustment         |    9.701 |         0.343 |                     2.042 |
| linear regression, full adjustment               |    7.564 |         0.44  |                    -0.095 |
| propensity weighting, incomplete adjustment      |    9.721 |         0.369 |                     2.062 |
| propensity weighting, full adjustment            |    7.646 |         0.57  |                    -0.013 |
| propensity stratification, incomplete adjustment |    9.734 |         0.373 |                     2.075 |
| propensity stratification, full adjustment       |    7.635 |         0.586 |                    -0.024 |

## Spend (dollars)

| estimate                                         |   effect |   spread (sd) |   bias against experiment |
|:-------------------------------------------------|---------:|--------------:|--------------------------:|
| experiment (benchmark)                           |    0.77  |         0.145 |                     0     |
| naive difference                                 |    1.155 |         0.116 |                     0.385 |
| linear regression, incomplete adjustment         |    0.951 |         0.122 |                     0.181 |
| linear regression, full adjustment               |    0.683 |         0.132 |                    -0.087 |
| propensity weighting, incomplete adjustment      |    0.987 |         0.12  |                     0.217 |
| propensity weighting, full adjustment            |    0.727 |         0.128 |                    -0.043 |
| propensity stratification, incomplete adjustment |    0.984 |         0.125 |                     0.214 |
| propensity stratification, full adjustment       |    0.726 |         0.132 |                    -0.044 |

## Refutation tests (visit, linear regression, first draw)

| adjustment   | refuter                   |   estimated_effect |   new_effect |   p_value |
|:-------------|:--------------------------|-------------------:|-------------:|----------:|
| full         | placebo_treatment_refuter |             0.0795 |      -0.0005 |    0.4558 |
| full         | random_common_cause       |             0.0795 |       0.0795 |    0.4569 |
| full         | data_subset_refuter       |             0.0795 |       0.0792 |    0.4506 |
| incomplete   | placebo_treatment_refuter |             0.0998 |      -0.0005 |    0.4551 |
| incomplete   | random_common_cause       |             0.0998 |       0.0998 |    0.4573 |
| incomplete   | data_subset_refuter       |             0.0998 |       0.0996 |    0.4585 |
