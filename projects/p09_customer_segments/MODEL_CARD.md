# Model card: customer segments (project 9)

## Model

k means (scikit-learn, k = 3, 10 restarts, seed 42) on 18 features per customer: six behavior features (days since last order, number of orders, net spend, mean basket, tenure, distinct products; log transformed where skewed) and twelve product category shares, all scaled to a normal distribution by quantiles. The category block is weighted to half the total variance of the behavior block. Segments are numbered by their share of net spend.

Chosen without the outcome: among k from 3 to 8, solutions with a mean bootstrap adjusted Rand index of at least 0.80 and no segment under 2 percent of customers were candidates; among scalings with a candidate, quantile scaling had the highest mean stability, and k = 3 the highest silhouette in it. The gap statistic also gives k = 3.

Training: CPU, 4 cores, scikit-learn 1.9.1, umap-learn 0.5.12; about 15 minutes for the full grid with gap statistic and bootstrap refits.

## Intended use

A portfolio demonstration of customer segmentation: feature design, scaling choices, choice of k with several criteria, stability under resampling, density based alternatives, and a check of the segments against the following quarter. **Not intended to assign real customers to marketing actions without a fresh run on current data and a controlled test of each action.**

## Training data

Online Retail II (UCI, CC BY 4.0): order lines of a UK online gift retailer, 1 December 2009 to 9 December 2011. Features use the orders before 9 September 2011; 5,240 customers with a positive net spend before that date. Rows without a customer id (22 percent of rows) are excluded, so customers who ordered as guests are not covered.

## Evaluation

The quarter after the cutoff (10 September to 9 December 2011), which the clustering did not see.

| Measure | Selected k means | RFM grid (baseline) | k means, RFM only |
| --- | ---: | ---: | ---: |
| Variance of log spend explained (adjusted eta squared) | 0.214 (0.192 to 0.238) | 0.262 (0.241 to 0.284) | 0.262 (0.240 to 0.285) |
| Variance of purchase indicator explained | 0.169 (0.149 to 0.189) | 0.227 (0.209 to 0.250) | 0.215 (0.194 to 0.235) |
| Purchase rate, lowest to highest segment | 21% to 73% | 16% to 82% | 17% to 78% |
| Segments | 3 | 10 | 3 |
| Bootstrap stability (adjusted Rand index) | 0.954 | rules, not fitted | 0.952 |

Intervals: 95 percent bootstrap over customers.

## Subgroups

| Segment | Customers | Net spend share | Bought next quarter | Jaccard stability |
| --- | ---: | ---: | ---: | ---: |
| 1 Active repeat buyers | 1,508 | 82% | 73% | 0.96 |
| 2 Occasional repeat buyers | 2,169 | 14% | 40% | 0.97 |
| 3 Single order customers | 1,563 | 4% | 21% | 1.00 |

Customers outside the United Kingdom are 7 to 10 percent of each segment; the features do not use country.

## Limitations and risks

* **The rule baseline is better on the next quarter.** The RFM grid explains more of next quarter's spend and purchase than the selected segments; the segments are a coarse summary, not a better targeting rule.
* **The scaling shapes the segments.** Quantile scaling turns shared values into gaps: segment 3 is exactly the customers with one order. In the first attempt, the return share produced a split by "any return or none".
* **Decisions after seeing results:** the return share was removed from the clustering features and the scaling rule was tightened after the first and second runs (see the README). The first attempt is reported next to the selected solution.
* **Keyword categories:** product categories come from keyword rules on descriptions; 7.1 percent of purchase value matches no rule, and some items fall in the wrong category.
* **Period and season:** the outcome quarter is the Christmas season. Results need not hold for other quarters or years.

## Files

Written by `make train` under `artifacts/` and not committed: the customer table, cluster labels for every solution, grids and `mlflow.db`. The repository publishes aggregate tables, figures, and the interactive map with rounded values per customer and no customer id.
