# Project 9 results (Online Retail II)

## Data

* 1,067,371 rows read from `online_retail_II.xlsx` (documented 1,067,371), 34,335 exact duplicates dropped (22,202 of them in both yearly sheets), 235,151 rows without a customer id dropped, 797,815 kept.
* 5,939 customers, 44,870 invoices, 2009-12-01 to 2011-12-09; 18,390 return lines; spend in no category rule: 7.1%.
* Cutoff 2011-09-09: features from the orders before it, outcome from the 91 days after it. 5,240 customers with positive net spend before the cutoff; 43.7% of them bought again after it.

## Selection

* Selected: k means on RFM and product mix features, quantile scaling, k = 3 (highest silhouette among candidates): silhouette 0.194, bootstrap stability 0.954, smallest segment 28.8%.
* Candidates (stability at least 0.8, smallest segment at least 2%): standard k=3, standard k=4, robust k=3, robust k=6, quantile k=3, quantile k=4.
* Mean stability over the k range by scaling: {'raw': 0.7934, 'quantile': 0.7743, 'standard': 0.7696, 'robust': 0.7589}.
* Gap statistic choice by scaling (None: no k in the range meets the rule): {'quantile': 3, 'raw': 2, 'robust': None, 'standard': None}.
* RFM only: robust scaling, k = 3.
* First attempt with return share: quantile scaling, k = 4 (highest silhouette among candidates): silhouette 0.349, bootstrap stability 0.814, smallest segment 3.9%.
* HDBSCAN setting compared: `hdbscan_m25_quantile`.

## Next quarter check

| segmentation                             |   segments |   eta2_spend |   eta2_spend_low |   eta2_spend_high |   eta2_purchase |   eta2_purchase_low |   eta2_purchase_high |   purchase_rate_min |   purchase_rate_max |
|:-----------------------------------------|-----------:|-------------:|-----------------:|------------------:|----------------:|--------------------:|---------------------:|--------------------:|--------------------:|
| k means, RFM and product mix (selected)  |          3 |        0.214 |            0.192 |             0.238 |           0.169 |               0.149 |                0.189 |               0.207 |               0.733 |
| k means, first attempt with return share |          4 |        0.137 |            0.120 |             0.155 |           0.117 |               0.102 |                0.136 |               0.200 |               0.624 |
| k means, RFM only                        |          3 |        0.262 |            0.240 |             0.285 |           0.215 |               0.194 |                0.235 |               0.165 |               0.781 |
| HDBSCAN, RFM and product mix             |          3 |        0.098 |            0.086 |             0.112 |           0.091 |               0.079 |                0.106 |               0.207 |               0.536 |
| RFM grid, ten rule segments (baseline)   |         10 |        0.262 |            0.241 |             0.284 |           0.227 |               0.209 |                0.250 |               0.161 |               0.820 |

Adjusted eta squared: share of the variance of the outcome explained by the segment means, adjusted for the number of segments; 95 percent bootstrap intervals over customers.

## Segment profiles

|   segment |   customers |   customer_share |   spend_share |   median_recency_days |   median_frequency |   median_monetary |   median_mean_basket |   mean_return_share | top_categories                                       |   future_purchase_rate |   future_spend_mean |   future_spend_share |   jaccard |
|----------:|------------:|-----------------:|--------------:|----------------------:|-------------------:|------------------:|---------------------:|--------------------:|:-----------------------------------------------------|-----------------------:|--------------------:|---------------------:|----------:|
|         1 |        1508 |             0.29 |          0.82 |                 43.39 |               9.00 |           3376.76 |               384.25 |                0.02 | bags x1.3, home_decor x1.1, kitchen_dining x1.0      |                   0.73 |             1413.26 |                 0.77 |      0.96 |
|         2 |        2169 |             0.41 |          0.14 |                181.56 |               3.00 |            721.43 |               226.57 |                0.02 | lights_candles x1.1, other x1.0, garden_outdoor x1.0 |                   0.40 |              218.01 |                 0.17 |      0.97 |
|         3 |        1563 |             0.30 |          0.04 |                321.57 |               1.00 |            229.95 |               233.54 |                0.01 | christmas x1.1, other x1.1, stationery_wrap x1.1     |                   0.21 |              105.28 |                 0.06 |      1.00 |

Segments are numbered by their share of net spend before the cutoff. `jaccard`: mean overlap of the segment with its best match over bootstrap refits (above 0.75 is usually read as stable).

## Segment profiles of the first attempt with return share

|   segment |   customers |   customer_share |   spend_share |   median_recency_days |   median_frequency |   median_monetary |   median_mean_basket |   mean_return_share | top_categories                                        |   future_purchase_rate |   future_spend_mean |   future_spend_share |   jaccard |
|----------:|------------:|-----------------:|--------------:|----------------------:|-------------------:|------------------:|---------------------:|--------------------:|:------------------------------------------------------|-----------------------:|--------------------:|---------------------:|----------:|
|         1 |        1933 |             0.37 |          0.78 |                 78.40 |               7.00 |           2096.95 |               322.60 |                0.05 | bags x1.1, storage x1.1, kitchen_dining x1.1          |                   0.62 |             1071.28 |                 0.75 |      1.00 |
|         2 |        1744 |             0.33 |          0.18 |                147.47 |               3.00 |            797.12 |               242.64 |                0.00 | other x1.1, stationery_wrap x1.0, garden_outdoor x1.0 |                   0.44 |              305.77 |                 0.19 |      1.00 |
|         3 |        1358 |             0.26 |          0.03 |                321.51 |               1.00 |            219.63 |               219.63 |                0.00 | other x1.1, stationery_wrap x1.1, garden_outdoor x1.1 |                   0.20 |               95.79 |                 0.05 |      1.00 |
|         4 |         205 |             0.04 |          0.01 |                323.58 |               1.00 |            299.75 |               314.10 |                0.11 | christmas x1.4, toys_crafts x1.2, lights_candles x1.2 |                   0.25 |              168.19 |                 0.01 |      1.00 |

## k means grid

| feature_set     | scaling   |   k |   silhouette |   davies_bouldin |   min_share |   stability_ari |   stability_ari_p05 |
|:----------------|:----------|----:|-------------:|-----------------:|------------:|----------------:|--------------------:|
| rfm_mix         | raw       |   2 |        0.845 |            0.740 |       0.003 |           0.472 |               0.010 |
| rfm_mix         | raw       |   3 |        0.163 |            1.585 |       0.003 |           0.533 |               0.343 |
| rfm_mix         | raw       |   4 |        0.179 |            1.502 |       0.002 |           0.903 |               0.744 |
| rfm_mix         | raw       |   5 |        0.183 |            1.328 |       0.001 |           0.795 |               0.587 |
| rfm_mix         | raw       |   6 |        0.158 |            1.416 |       0.002 |           0.817 |               0.496 |
| rfm_mix         | raw       |   7 |        0.144 |            1.728 |       0.002 |           0.809 |               0.624 |
| rfm_mix         | raw       |   8 |        0.142 |            1.630 |       0.001 |           0.903 |               0.700 |
| rfm_mix         | raw       |   9 |        0.140 |            1.517 |       0.001 |           0.883 |               0.756 |
| rfm_mix         | raw       |  10 |        0.139 |            1.529 |       0.001 |           0.807 |               0.640 |
| rfm_mix         | standard  |   2 |        0.219 |            1.580 |       0.432 |           0.977 |               0.942 |
| rfm_mix         | standard  |   3 |        0.198 |            1.907 |       0.292 |           0.945 |               0.824 |
| rfm_mix         | standard  |   4 |        0.139 |            1.850 |       0.220 |           0.886 |               0.767 |
| rfm_mix         | standard  |   5 |        0.126 |            1.986 |       0.137 |           0.780 |               0.466 |
| rfm_mix         | standard  |   6 |        0.116 |            2.009 |       0.127 |           0.797 |               0.578 |
| rfm_mix         | standard  |   7 |        0.116 |            2.067 |       0.039 |           0.653 |               0.515 |
| rfm_mix         | standard  |   8 |        0.114 |            2.079 |       0.054 |           0.556 |               0.417 |
| rfm_mix         | standard  |   9 |        0.099 |            2.105 |       0.047 |           0.568 |               0.424 |
| rfm_mix         | standard  |  10 |        0.102 |            1.989 |       0.010 |           0.624 |               0.469 |
| rfm_mix         | robust    |   2 |        0.213 |            1.621 |       0.458 |           0.980 |               0.947 |
| rfm_mix         | robust    |   3 |        0.146 |            2.053 |       0.276 |           0.844 |               0.566 |
| rfm_mix         | robust    |   4 |        0.144 |            1.951 |       0.192 |           0.794 |               0.559 |
| rfm_mix         | robust    |   5 |        0.119 |            1.865 |       0.181 |           0.732 |               0.529 |
| rfm_mix         | robust    |   6 |        0.125 |            1.918 |       0.081 |           0.856 |               0.747 |
| rfm_mix         | robust    |   7 |        0.119 |            1.969 |       0.064 |           0.703 |               0.439 |
| rfm_mix         | robust    |   8 |        0.095 |            2.036 |       0.056 |           0.624 |               0.444 |
| rfm_mix         | robust    |   9 |        0.098 |            1.934 |       0.010 |           0.566 |               0.443 |
| rfm_mix         | robust    |  10 |        0.095 |            2.023 |       0.009 |           0.591 |               0.423 |
| rfm_mix         | quantile  |   2 |        0.432 |            1.031 |       0.298 |           1.000 |               1.000 |
| rfm_mix         | quantile  |   3 |        0.194 |            1.769 |       0.288 |           0.954 |               0.936 |
| rfm_mix         | quantile  |   4 |        0.140 |            2.421 |       0.125 |           0.851 |               0.527 |
| rfm_mix         | quantile  |   5 |        0.107 |            2.698 |       0.132 |           0.773 |               0.563 |
| rfm_mix         | quantile  |   6 |        0.101 |            2.578 |       0.021 |           0.683 |               0.519 |
| rfm_mix         | quantile  |   7 |        0.106 |            2.352 |       0.010 |           0.676 |               0.469 |
| rfm_mix         | quantile  |   8 |        0.094 |            2.442 |       0.019 |           0.710 |               0.533 |
| rfm_mix         | quantile  |   9 |        0.073 |            2.458 |       0.019 |           0.586 |               0.452 |
| rfm_mix         | quantile  |  10 |        0.075 |            2.277 |       0.009 |           0.606 |               0.466 |
| rfm_mix_returns | raw       |   2 |        0.832 |            0.750 |       0.003 |           0.441 |               0.013 |
| rfm_mix_returns | raw       |   3 |        0.131 |            1.829 |       0.005 |           0.555 |               0.340 |
| rfm_mix_returns | raw       |   4 |        0.157 |            1.658 |       0.002 |           0.861 |               0.516 |
| rfm_mix_returns | raw       |   5 |        0.164 |            1.537 |       0.002 |           0.871 |               0.733 |
| rfm_mix_returns | raw       |   6 |        0.140 |            1.582 |       0.002 |           0.798 |               0.604 |
| rfm_mix_returns | raw       |   7 |        0.142 |            1.428 |       0.002 |           0.769 |               0.410 |
| rfm_mix_returns | raw       |   8 |        0.142 |            1.276 |       0.000 |           0.746 |               0.490 |
| rfm_mix_returns | raw       |   9 |        0.137 |            1.550 |       0.001 |           0.827 |               0.637 |
| rfm_mix_returns | raw       |  10 |        0.130 |            1.635 |       0.001 |           0.806 |               0.603 |
| rfm_mix_returns | standard  |   2 |        0.199 |            1.703 |       0.433 |           0.977 |               0.947 |
| rfm_mix_returns | standard  |   3 |        0.180 |            2.075 |       0.284 |           0.927 |               0.736 |
| rfm_mix_returns | standard  |   4 |        0.186 |            1.773 |       0.015 |           0.682 |               0.447 |
| rfm_mix_returns | standard  |   5 |        0.127 |            1.773 |       0.015 |           0.821 |               0.509 |
| rfm_mix_returns | standard  |   6 |        0.115 |            1.942 |       0.015 |           0.769 |               0.489 |
| rfm_mix_returns | standard  |   7 |        0.103 |            1.992 |       0.015 |           0.793 |               0.582 |
| rfm_mix_returns | standard  |   8 |        0.101 |            2.082 |       0.014 |           0.615 |               0.409 |
| rfm_mix_returns | standard  |   9 |        0.104 |            2.103 |       0.011 |           0.603 |               0.403 |
| rfm_mix_returns | standard  |  10 |        0.105 |            2.002 |       0.003 |           0.550 |               0.427 |
| rfm_mix_returns | robust    |   2 |        0.789 |            0.590 |       0.018 |           0.920 |               0.763 |
| rfm_mix_returns | robust    |   3 |        0.528 |            0.818 |       0.008 |           0.952 |               0.887 |
| rfm_mix_returns | robust    |   4 |        0.393 |            0.888 |       0.003 |           0.526 |               0.195 |
| rfm_mix_returns | robust    |   5 |        0.096 |            1.651 |       0.003 |           0.363 |               0.203 |
| rfm_mix_returns | robust    |   6 |        0.094 |            1.646 |       0.003 |           0.562 |               0.366 |
| rfm_mix_returns | robust    |   7 |        0.079 |            1.725 |       0.003 |           0.518 |               0.378 |
| rfm_mix_returns | robust    |   8 |        0.094 |            1.815 |       0.003 |           0.548 |               0.429 |
| rfm_mix_returns | robust    |   9 |        0.099 |            1.744 |       0.003 |           0.536 |               0.400 |
| rfm_mix_returns | robust    |  10 |        0.101 |            1.722 |       0.003 |           0.502 |               0.380 |
| rfm_mix_returns | quantile  |   2 |        0.334 |            1.229 |       0.408 |           1.000 |               1.000 |
| rfm_mix_returns | quantile  |   3 |        0.333 |            1.290 |       0.265 |           0.976 |               0.908 |
| rfm_mix_returns | quantile  |   4 |        0.349 |            1.246 |       0.039 |           0.814 |               0.805 |
| rfm_mix_returns | quantile  |   5 |        0.306 |            2.055 |       0.039 |           0.865 |               0.746 |
| rfm_mix_returns | quantile  |   6 |        0.200 |            2.329 |       0.039 |           0.787 |               0.595 |
| rfm_mix_returns | quantile  |   7 |        0.120 |            2.712 |       0.039 |           0.775 |               0.627 |
| rfm_mix_returns | quantile  |   8 |        0.125 |            2.503 |       0.007 |           0.774 |               0.583 |
| rfm_mix_returns | quantile  |   9 |        0.117 |            2.406 |       0.007 |           0.717 |               0.609 |
| rfm_mix_returns | quantile  |  10 |        0.103 |            2.665 |       0.008 |           0.692 |               0.578 |

## Density based clustering (RFM and product mix)

| scaling   | method       |   segments |   noise_share |   silhouette |
|:----------|:-------------|-----------:|--------------:|-------------:|
| standard  | dbscan_q0.5  |          2 |         0.290 |        0.207 |
| standard  | dbscan_q0.75 |          2 |         0.123 |        0.307 |
| standard  | dbscan_q0.9  |          2 |         0.040 |        0.419 |
| standard  | dbscan_q0.95 |          1 |         0.017 |      nan     |
| standard  | hdbscan_m25  |          0 |         1.000 |      nan     |
| standard  | hdbscan_m50  |          0 |         1.000 |      nan     |
| standard  | hdbscan_m100 |          0 |         1.000 |      nan     |
| robust    | dbscan_q0.5  |          2 |         0.293 |        0.234 |
| robust    | dbscan_q0.75 |          1 |         0.125 |      nan     |
| robust    | dbscan_q0.9  |          3 |         0.039 |        0.396 |
| robust    | dbscan_q0.95 |          1 |         0.019 |      nan     |
| robust    | hdbscan_m25  |          0 |         1.000 |      nan     |
| robust    | hdbscan_m50  |          0 |         1.000 |      nan     |
| robust    | hdbscan_m100 |          0 |         1.000 |      nan     |
| quantile  | dbscan_q0.5  |          3 |         0.280 |        0.473 |
| quantile  | dbscan_q0.75 |          6 |         0.115 |        0.426 |
| quantile  | dbscan_q0.9  |          5 |         0.043 |        0.426 |
| quantile  | dbscan_q0.95 |          6 |         0.020 |        0.417 |
| quantile  | hdbscan_m25  |          2 |         0.006 |        0.436 |
| quantile  | hdbscan_m50  |          2 |         0.006 |        0.436 |
| quantile  | hdbscan_m100 |          2 |         0.006 |        0.436 |
