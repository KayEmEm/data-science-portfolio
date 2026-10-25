# Results

Data: Elliptic. Test: 16,670 labelled transactions of the test steps, 1,083 of them illicit. Selected on validation F1: **LightGBM, all features** (threshold 0.429).

## Models

| Model | Validation F1 | Test F1 (95% interval) | Precision | Recall | F1 at 0.5 | PR AUC | F1 before step 43 | F1 from step 43 | F1 range over seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM, all features | 0.979 | 0.799 (0.624 to 0.877) | 0.897 | 0.721 | 0.806 | 0.799 | 0.895 | 0.028 |  |
| LightGBM, all features and graph measures | 0.977 | 0.789 (0.606 to 0.869) | 0.859 | 0.729 | 0.811 | 0.799 | 0.891 | 0.034 |  |
| LightGBM, local | 0.958 | 0.765 (0.600 to 0.841) | 0.876 | 0.679 | 0.744 | 0.778 | 0.857 | 0.019 |  |
| GraphSAGE, local | 0.924 | 0.697 (0.493 to 0.792) | 0.899 | 0.569 | 0.463 | 0.712 | 0.785 | 0.020 | 0.683 to 0.691 |
| GraphSAGE, all features | 0.942 | 0.640 (0.470 to 0.737) | 0.677 | 0.607 | 0.551 | 0.630 | 0.726 | 0.024 | 0.610 to 0.638 |
| GCN, all features | 0.853 | 0.458 (0.247 to 0.593) | 0.795 | 0.321 | 0.530 | 0.555 | 0.522 | 0.000 | 0.380 to 0.488 |
| Logistic regression, local (baseline) | 0.692 | 0.428 (0.215 to 0.595) | 0.303 | 0.726 | 0.407 | 0.217 | 0.593 | 0.021 |  |

Intervals: 95 percent, from resampling the test time steps with replacement. Graph networks: mean probability over seeds; the last column gives the test F1 of each seed alone.

## Paired differences to LightGBM on all features

| Model | Test F1 difference | 95% interval |
| --- | ---: | ---: |
| Logistic regression, local (baseline) | -0.372 | -0.485 to -0.247 |
| LightGBM, local | -0.035 | -0.045 to -0.024 |
| LightGBM, all features and graph measures | -0.011 | -0.027 to 0.002 |
| GCN, all features | -0.342 | -0.474 to -0.245 |
| GraphSAGE, all features | -0.159 | -0.235 to -0.105 |
| GraphSAGE, local | -0.103 | -0.176 to -0.058 |

## Network

| Measure | Value |
| --- | ---: |
| Transactions | 203,769 |
| Edges | 234,355 |
| Counts match the documented ones | True |
| Edges across time steps | 0 |
| Labelled share | 22.9% |
| Illicit share of labelled | 9.8% |
| Isolated transactions | 0 |
| Mean density per step | 3.18e-04 |
| Mean share of the largest component per step | 1.000 |
| Mean degree assortativity per step | -0.081 |
| Louvain communities (all steps) | 2,815 |
| Mean modularity per step | 0.900 |

### Centrality, illicit against licit

| Measure | Median illicit | Median licit | Mean illicit | Mean licit | ROC AUC alone |
| --- | ---: | ---: | ---: | ---: | ---: |
| in_degree | 1 | 1 | 1.27 | 1.909 | 0.473 |
| out_degree | 1 | 1 | 0.7417 | 1.186 | 0.405 |
| pagerank | 0.5351 | 0.4811 | 0.8435 | 1.154 | 0.505 |
| betweenness | 0 | 0.0002211 | 0.002068 | 0.004558 | 0.420 |
| clustering | 0 | 0 | 0.0003564 | 0.00832 | 0.483 |
| core_number | 1 | 1 | 1.187 | 1.406 | 0.414 |
| component_size | 4296 | 4975 | 4258 | 4900 | 0.376 |

ROC AUC alone: the measure used as the only illicit score, over labelled transactions of all steps; 0.5 is no signal, values below 0.5 mean that illicit transactions have lower values.

### Clustering of illicit transactions

| Measure | Observed | Labels shuffled within step (mean, max of 20) |
| --- | ---: | ---: |
| Edges with an illicit end whose other end is illicit too | 0.370 | 0.096, 0.108 |
| Illicit transactions in communities (at least 5 transactions) whose labelled members are at least half illicit | 0.607 | 0.059, 0.067 |

### Subgraphs drawn

| Center tx_id | Step | Transactions | Illicit | Licit | Flagged | Flagged illicit | Flagged unlabelled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 71988450 | 42 | 97 | 50 | 20 | 50 | 49 | 1 |
| 116058585 | 38 | 84 | 24 | 19 | 28 | 23 | 5 |
| 155674580 | 48 | 66 | 23 | 12 | 4 | 0 | 4 |
| 68995268 | 35 | 26 | 23 | 2 | 24 | 23 | 1 |
