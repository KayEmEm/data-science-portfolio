# Project 8 results

Data: UNSW-NB15, partition layout, 257,673 flows (63.9% attacks). Test split: 82,332 flows (55.1% attacks). Unsupervised detectors are fitted on a training sample with 1% attacks and no labels. Thresholds: 1% of normal validation flows flagged. PR AUC is shown as observed and rescaled to 1% attacks (precision recomputed from recall and false alarm rate). Intervals: bootstrap over test flows.

Selected on validation PR AUC at 1% attacks: **Isolation Forest**.

## Detectors (main experiment)

| Detector                                      |   Validation PR AUC at 1% attacks | Test PR AUC at 1% attacks (95% interval)   | Test PR AUC as observed (95% interval)   |   Test ROC AUC | Recall at 1% FPR (95% interval)   |   Test FPR |   Precision |   Precision of top 100 |   Precision of top 1000 |
|:----------------------------------------------|----------------------------------:|:-------------------------------------------|:-----------------------------------------|---------------:|:----------------------------------|-----------:|------------:|-----------------------:|------------------------:|
| LightGBM, supervised (upper reference)        |                             0.887 | 0.859 (0.853 to 0.864)                     | 0.989 (0.988 to 0.989)                   |          0.984 | 0.930 (0.928 to 0.933)            |     0.0691 |       0.943 |                   1    |                   1     |
| Isolation Forest (selected)                   |                             0.11  | 0.150 (0.141 to 0.161)                     | 0.844 (0.840 to 0.847)                   |          0.815 | 0.220 (0.216 to 0.223)            |     0.0136 |       0.952 |                   1    |                   1     |
| Robust z-score (baseline)                     |                             0.106 | 0.049 (0.046 to 0.052)                     | 0.799 (0.794 to 0.803)                   |          0.785 | 0.216 (0.213 to 0.219)            |     0.0185 |       0.935 |                   0.61 |                   0.864 |
| Autoencoder                                   |                             0.062 | 0.045 (0.044 to 0.047)                     | 0.825 (0.821 to 0.829)                   |          0.829 | 0.153 (0.149 to 0.156)            |     0.0289 |       0.866 |                   0.95 |                   0.905 |
| Ensemble: Isolation Forest, ECOD, autoencoder |                             0.057 | 0.032 (0.031 to 0.032)                     | 0.785 (0.780 to 0.789)                   |          0.818 | 0.029 (0.027 to 0.030)            |     0.0145 |       0.707 |                   0.72 |                   0.719 |
| Local Outlier Factor                          |                             0.034 | 0.029 (0.028 to 0.030)                     | 0.762 (0.758 to 0.767)                   |          0.785 | 0.004 (0.003 to 0.004)            |     0.0061 |       0.426 |                   0.61 |                   0.64  |
| HBOS (PyOD)                                   |                             0.04  | 0.021 (0.020 to 0.021)                     | 0.697 (0.692 to 0.701)                   |          0.745 | 0.013 (0.012 to 0.014)            |     0.0259 |       0.383 |                   0.22 |                   0.344 |
| ECOD (PyOD)                                   |                             0.024 | 0.016 (0.016 to 0.017)                     | 0.645 (0.640 to 0.650)                   |          0.639 | 0.014 (0.012 to 0.015)            |     0.0085 |       0.664 |                   0.89 |                   0.674 |
| COPOD (PyOD)                                  |                             0.022 | 0.014 (0.014 to 0.015)                     | 0.618 (0.612 to 0.622)                   |          0.597 | 0.012 (0.011 to 0.013)            |     0.01   |       0.605 |                   0.76 |                   0.612 |

## Recall per attack type at the threshold

|                                               |   Analysis |   Backdoor |   DoS |   Exploits |   Fuzzers |   Generic |   Reconnaissance |   Shellcode |   Worms |
|:----------------------------------------------|-----------:|-----------:|------:|-----------:|----------:|----------:|-----------------:|------------:|--------:|
| LightGBM, supervised (upper reference)        |      0.923 |      0.997 | 0.989 |      0.977 |     0.552 |     0.999 |            0.992 |       0.915 |   0.977 |
| Isolation Forest                              |      0     |      0.007 | 0.011 |      0.006 |     0.006 |     0.519 |            0.003 |       0     |   0     |
| Robust z-score (baseline)                     |      0.021 |      0.026 | 0.034 |      0.086 |     0.02  |     0.452 |            0.003 |       0     |   0.114 |
| Autoencoder                                   |      0.464 |      0.554 | 0.226 |      0.319 |     0.163 |     0.033 |            0.051 |       0.021 |   0.159 |
| Ensemble: Isolation Forest, ECOD, autoencoder |      0     |      0.019 | 0.046 |      0.067 |     0.018 |     0.011 |            0.005 |       0     |   0.159 |
| Local Outlier Factor                          |      0.001 |      0.021 | 0.013 |      0.007 |     0.001 |     0.001 |            0.001 |       0     |   0     |
| HBOS (PyOD)                                   |      0.001 |      0.002 | 0.019 |      0.035 |     0.011 |     0.002 |            0.002 |       0     |   0.114 |
| ECOD (PyOD)                                   |      0     |      0     | 0.015 |      0.02  |     0.004 |     0.016 |            0     |       0     |   0.114 |
| COPOD (PyOD)                                  |      0     |      0     | 0.017 |      0.021 |     0.006 |     0.012 |            0.001 |       0     |   0.068 |

## Threshold without labels

Threshold at the score above which 1% of the training sample lies.

| detector                                      |   label_free_alert_share |   label_free_fpr |   label_free_recall |
|:----------------------------------------------|-------------------------:|-----------------:|--------------------:|
| Robust z-score (baseline)                     |                   0.1161 |           0.0151 |              0.1985 |
| Isolation Forest                              |                   0.1174 |           0.0089 |              0.2059 |
| Local Outlier Factor                          |                   0.0032 |           0.0054 |              0.0014 |
| HBOS (PyOD)                                   |                   0.017  |           0.0242 |              0.0111 |
| ECOD (PyOD)                                   |                   0.0021 |           0.0011 |              0.0028 |
| COPOD (PyOD)                                  |                   0.0029 |           0.0019 |              0.0037 |
| Autoencoder                                   |                   0.087  |           0.0265 |              0.1364 |
| Ensemble: Isolation Forest, ECOD, autoencoder |                   0.0124 |           0.0078 |              0.0162 |
| LightGBM, supervised (upper reference)        |                   0.5365 |           0.0606 |              0.925  |

## Contamination study

Isolation Forest and the autoencoder; each share drawn and fitted with seeds [42, 43, 44].

| detector         |   attack_share |   test_roc_auc mean |   test_roc_auc min |   test_roc_auc max |   test_recall mean |   test_recall min |   test_recall max |   test_pr_auc_rescaled mean |   test_pr_auc_rescaled min |   test_pr_auc_rescaled max |
|:-----------------|---------------:|--------------------:|-------------------:|-------------------:|-------------------:|------------------:|------------------:|----------------------------:|---------------------------:|---------------------------:|
| Autoencoder      |           0    |               0.925 |              0.916 |              0.932 |              0.441 |             0.432 |             0.452 |                       0.13  |                      0.099 |                      0.164 |
| Autoencoder      |           0.01 |               0.834 |              0.829 |              0.839 |              0.131 |             0.108 |             0.153 |                       0.045 |                      0.042 |                      0.049 |
| Autoencoder      |           0.05 |               0.737 |              0.702 |              0.766 |              0.078 |             0.067 |             0.087 |                       0.029 |                      0.023 |                      0.031 |
| Autoencoder      |           0.2  |               0.649 |              0.625 |              0.663 |              0.043 |             0.038 |             0.048 |                       0.021 |                      0.019 |                      0.022 |
| Isolation Forest |           0    |               0.814 |              0.814 |              0.815 |              0.156 |             0.114 |             0.18  |                       0.061 |                      0.046 |                      0.075 |
| Isolation Forest |           0.01 |               0.821 |              0.815 |              0.826 |              0.221 |             0.22  |             0.224 |                       0.156 |                      0.15  |                      0.164 |
| Isolation Forest |           0.05 |               0.813 |              0.809 |              0.816 |              0.032 |             0.019 |             0.05  |                       0.035 |                      0.032 |                      0.038 |
| Isolation Forest |           0.2  |               0.709 |              0.705 |              0.718 |              0.005 |             0.005 |             0.006 |                       0.017 |                      0.016 |                      0.017 |

## Time to live ablation

`main` uses all features, `no_ttl` drops sttl, dttl and ct_state_ttl.

|                                        |   test_roc_auc main |   test_roc_auc no_ttl |   test_recall main |   test_recall no_ttl |   test_pr_auc_rescaled main |   test_pr_auc_rescaled no_ttl |
|:---------------------------------------|--------------------:|----------------------:|-------------------:|---------------------:|----------------------------:|------------------------------:|
| Autoencoder                            |               0.829 |                 0.855 |              0.153 |                0.182 |                       0.045 |                         0.05  |
| ECOD (PyOD)                            |               0.639 |                 0.64  |              0.014 |                0.011 |                       0.016 |                         0.017 |
| Isolation Forest                       |               0.815 |                 0.811 |              0.22  |                0.235 |                       0.15  |                         0.157 |
| LightGBM, supervised (upper reference) |               0.984 |                 0.983 |              0.93  |                0.925 |                       0.859 |                         0.854 |

## Data audit

```json
{
  "layout": "partition",
  "rows": 257673,
  "attacks": 164673,
  "attack_share": 0.63908,
  "duplicate_rows": 103576,
  "test_rows_also_in_train": 7745,
  "counts_found": {
    "train": 175341,
    "test": 82332
  },
  "counts_documented": {
    "train": 175341,
    "test": 82332
  },
  "counts_match_documented": true,
  "files": [
    "UNSW_NB15_testing-set.csv",
    "UNSW_NB15_training-set.csv"
  ],
  "label_0_with_attack_type": 0,
  "label_1_without_attack_type": 0,
  "not_numeric_cells": {}
}
```

## Run times (seconds)

```json
{
  "main": {
    "robust_z_fit": 0.1,
    "robust_z_score": 0.1,
    "iforest_fit": 0.8,
    "iforest_score": 2.1,
    "lof_fit": 0.7,
    "lof_score": 5.1,
    "hbos_fit": 2.7,
    "hbos_score": 0.3,
    "ecod_fit": 1.4,
    "ecod_score": 7.3,
    "copod_fit": 0.9,
    "copod_score": 7.1,
    "autoencoder_fit": 14.9,
    "autoencoder_score": 0.1,
    "ensemble_fit": 3.1,
    "ensemble_score": 9.5,
    "lightgbm_fit": 5.1,
    "lightgbm_score": 1.3
  },
  "no_ttl": {
    "iforest_fit": 0.8,
    "iforest_score": 2.1,
    "ecod_fit": 1.2,
    "ecod_score": 6.7,
    "autoencoder_fit": 17.8,
    "autoencoder_score": 0.1,
    "lightgbm_fit": 5.6,
    "lightgbm_score": 1.3
  }
}
```
