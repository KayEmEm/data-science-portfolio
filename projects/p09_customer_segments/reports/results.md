# Project 9 results (Online Retail II)

## Data

* 1,067,371 rows read from `online_retail_II.xlsx` (documented 1,067,371), 34,335 exact duplicates dropped (22,202 of them in both yearly sheets), 235,151 rows without a customer id dropped, 797,815 kept.
* 5,939 customers, 44,870 invoices, 2009-12-01 to 2011-12-09; 18,390 return lines; spend in no category rule: 7.1%.
* Cutoff 2011-09-09: features from the orders before it, outcome from the 91 days after it. 5,240 customers with positive net spend before the cutoff; 43.7% of them bought again after it.

## Selection

* Selected: k means on RFM and product mix features, quantile scaling, k = 4 (highest silhouette among candidates): silhouette 0.349, bootstrap stability 0.814, smallest segment 3.9%.
* Candidates (stability at least 0.8, smallest segment at least 2%): standard k=3, quantile k=3, quantile k=4, quantile k=5.
* Mean stability over the k range by scaling: {'quantile': 0.8318, 'standard': 0.7678, 'raw': 0.7668, 'robust': 0.578}.
* Gap statistic choice by scaling (None: no k in the range meets the rule): {'quantile': 3, 'raw': 2, 'robust': 2, 'standard': 2}.
* RFM only: raw scaling, k = 3.
* HDBSCAN setting compared: `hdbscan_m25_quantile`.

## Next quarter check

| segmentation                            |   segments |   eta2_spend |   eta2_spend_low |   eta2_spend_high |   eta2_purchase |   eta2_purchase_low |   eta2_purchase_high |   purchase_rate_min |   purchase_rate_max |
|:----------------------------------------|-----------:|-------------:|-----------------:|------------------:|----------------:|--------------------:|---------------------:|--------------------:|--------------------:|
| k means, RFM and product mix (selected) |          4 |        0.137 |            0.120 |             0.155 |           0.117 |               0.102 |                0.136 |               0.200 |               0.624 |
| k means, RFM only                       |          3 |        0.166 |            0.149 |             0.186 |           0.146 |               0.132 |                0.164 |               0.214 |               0.952 |
| HDBSCAN, RFM and product mix            |          5 |        0.141 |            0.125 |             0.161 |           0.120 |               0.105 |                0.140 |               0.201 |               0.631 |
| RFM grid, ten rule segments (baseline)  |         10 |        0.262 |            0.241 |             0.284 |           0.227 |               0.209 |                0.250 |               0.161 |               0.820 |

Adjusted eta squared: share of the variance of the outcome explained by the segment means, adjusted for the number of segments; 95 percent bootstrap intervals over customers.

## Segment profiles

|   segment |   customers |   customer_share |   spend_share |   median_recency_days |   median_frequency |   median_monetary |   median_mean_basket |   mean_return_share | top_categories                                        |   future_purchase_rate |   future_spend_mean |   future_spend_share |   jaccard |
|----------:|------------:|-----------------:|--------------:|----------------------:|-------------------:|------------------:|---------------------:|--------------------:|:------------------------------------------------------|-----------------------:|--------------------:|---------------------:|----------:|
|         1 |        1933 |             0.37 |          0.78 |                 78.40 |               7.00 |           2096.95 |               322.60 |                0.05 | bags x1.1, storage x1.1, kitchen_dining x1.1          |                   0.62 |             1071.28 |                 0.75 |      1.00 |
|         2 |        1744 |             0.33 |          0.18 |                147.47 |               3.00 |            797.12 |               242.64 |                0.00 | other x1.1, stationery_wrap x1.0, garden_outdoor x1.0 |                   0.44 |              305.77 |                 0.19 |      1.00 |
|         3 |        1358 |             0.26 |          0.03 |                321.51 |               1.00 |            219.63 |               219.63 |                0.00 | other x1.1, stationery_wrap x1.1, garden_outdoor x1.1 |                   0.20 |               95.79 |                 0.05 |      1.00 |
|         4 |         205 |             0.04 |          0.01 |                323.58 |               1.00 |            299.75 |               314.10 |                0.11 | christmas x1.4, toys_crafts x1.2, lights_candles x1.2 |                   0.25 |              168.19 |                 0.01 |      1.00 |

Segments are numbered by their share of net spend before the cutoff. `jaccard`: mean overlap of the segment with its best match over bootstrap refits (above 0.75 is usually read as stable).

## k means grid (RFM and product mix)

| scaling   |   k |   silhouette |   davies_bouldin |   min_share |   stability_ari |   stability_ari_p05 |
|:----------|----:|-------------:|-----------------:|------------:|----------------:|--------------------:|
| raw       |   2 |        0.832 |            0.750 |       0.003 |           0.441 |               0.013 |
| raw       |   3 |        0.131 |            1.829 |       0.005 |           0.555 |               0.340 |
| raw       |   4 |        0.157 |            1.658 |       0.002 |           0.861 |               0.516 |
| raw       |   5 |        0.164 |            1.537 |       0.002 |           0.871 |               0.733 |
| raw       |   6 |        0.140 |            1.582 |       0.002 |           0.798 |               0.604 |
| raw       |   7 |        0.142 |            1.428 |       0.002 |           0.769 |               0.410 |
| raw       |   8 |        0.142 |            1.276 |       0.000 |           0.746 |               0.490 |
| raw       |   9 |        0.137 |            1.550 |       0.001 |           0.827 |               0.637 |
| raw       |  10 |        0.130 |            1.635 |       0.001 |           0.806 |               0.603 |
| standard  |   2 |        0.199 |            1.703 |       0.433 |           0.977 |               0.947 |
| standard  |   3 |        0.180 |            2.075 |       0.284 |           0.927 |               0.736 |
| standard  |   4 |        0.186 |            1.773 |       0.015 |           0.682 |               0.447 |
| standard  |   5 |        0.127 |            1.773 |       0.015 |           0.821 |               0.509 |
| standard  |   6 |        0.115 |            1.942 |       0.015 |           0.769 |               0.489 |
| standard  |   7 |        0.103 |            1.992 |       0.015 |           0.793 |               0.582 |
| standard  |   8 |        0.101 |            2.082 |       0.014 |           0.615 |               0.409 |
| standard  |   9 |        0.104 |            2.103 |       0.011 |           0.603 |               0.403 |
| standard  |  10 |        0.105 |            2.002 |       0.003 |           0.550 |               0.427 |
| robust    |   2 |        0.789 |            0.590 |       0.018 |           0.920 |               0.763 |
| robust    |   3 |        0.528 |            0.818 |       0.008 |           0.952 |               0.887 |
| robust    |   4 |        0.393 |            0.888 |       0.003 |           0.526 |               0.195 |
| robust    |   5 |        0.096 |            1.651 |       0.003 |           0.363 |               0.203 |
| robust    |   6 |        0.094 |            1.646 |       0.003 |           0.562 |               0.366 |
| robust    |   7 |        0.079 |            1.725 |       0.003 |           0.518 |               0.378 |
| robust    |   8 |        0.094 |            1.815 |       0.003 |           0.548 |               0.429 |
| robust    |   9 |        0.099 |            1.744 |       0.003 |           0.536 |               0.400 |
| robust    |  10 |        0.101 |            1.722 |       0.003 |           0.502 |               0.380 |
| quantile  |   2 |        0.334 |            1.229 |       0.408 |           1.000 |               1.000 |
| quantile  |   3 |        0.333 |            1.290 |       0.265 |           0.976 |               0.908 |
| quantile  |   4 |        0.349 |            1.246 |       0.039 |           0.814 |               0.805 |
| quantile  |   5 |        0.306 |            2.055 |       0.039 |           0.865 |               0.746 |
| quantile  |   6 |        0.200 |            2.329 |       0.039 |           0.787 |               0.595 |
| quantile  |   7 |        0.120 |            2.712 |       0.039 |           0.775 |               0.627 |
| quantile  |   8 |        0.125 |            2.503 |       0.007 |           0.774 |               0.583 |
| quantile  |   9 |        0.117 |            2.406 |       0.007 |           0.717 |               0.609 |
| quantile  |  10 |        0.103 |            2.665 |       0.008 |           0.692 |               0.578 |

## Density based clustering (RFM and product mix)

| scaling   | method       |   segments |   noise_share |   silhouette |
|:----------|:-------------|-----------:|--------------:|-------------:|
| standard  | dbscan_q0.5  |          4 |         0.298 |        0.251 |
| standard  | dbscan_q0.75 |          1 |         0.127 |      nan     |
| standard  | dbscan_q0.9  |          2 |         0.042 |        0.416 |
| standard  | dbscan_q0.95 |          1 |         0.023 |      nan     |
| standard  | hdbscan_m25  |          0 |         1.000 |      nan     |
| standard  | hdbscan_m50  |          0 |         1.000 |      nan     |
| standard  | hdbscan_m100 |          0 |         1.000 |      nan     |
| robust    | dbscan_q0.5  |          5 |         0.290 |        0.516 |
| robust    | dbscan_q0.75 |          3 |         0.131 |        0.487 |
| robust    | dbscan_q0.9  |          1 |         0.047 |      nan     |
| robust    | dbscan_q0.95 |          1 |         0.023 |      nan     |
| robust    | hdbscan_m25  |          3 |         0.687 |        0.523 |
| robust    | hdbscan_m50  |          0 |         1.000 |      nan     |
| robust    | hdbscan_m100 |          0 |         1.000 |      nan     |
| quantile  | dbscan_q0.5  |         10 |         0.273 |        0.317 |
| quantile  | dbscan_q0.75 |          8 |         0.114 |        0.375 |
| quantile  | dbscan_q0.9  |          6 |         0.040 |        0.367 |
| quantile  | dbscan_q0.95 |          4 |         0.015 |        0.357 |
| quantile  | hdbscan_m25  |          4 |         0.018 |        0.358 |
| quantile  | hdbscan_m50  |          4 |         0.018 |        0.358 |
| quantile  | hdbscan_m100 |          4 |         0.018 |        0.358 |
