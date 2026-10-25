# Model card: illicit transaction classifier (project 11)

## Model

A LightGBM classifier (LightGBM 4.7.0, up to 1,000 trees, learning rate 0.03, 31 leaves, early stopping on validation log loss at 654 trees, seed 42) that gives each Bitcoin transaction a probability of being illicit. Input: the 93 local and 72 aggregated features of the Elliptic dataset. It was selected on validation F1 among seven models: logistic regression on local features (baseline), LightGBM on local features, on all features, and on all features with seven graph measures, GCN and GraphSAGE on all features, and GraphSAGE on local features.

Decision threshold: 0.429, the probability with the best illicit F1 on the validation steps.

Training: CPU, 4 cores; about 10 seconds for this model, about 35 minutes for the whole pipeline, mostly the graph networks (three seeds each).

## Intended use

A portfolio demonstration of graph analytics for anti money laundering work: network description, community analysis, node classification on a split by time, and graph queries in Neo4j. **Not intended to decide on real transactions or people.** A flag is a lead for review, not evidence.

## Training data

Elliptic Data Set (Elliptic, 2019; research use): labelled transactions of time steps 1 to 30 for training (2,954 illicit, 23,951 licit) and 31 to 34 for validation (508 illicit, 2,481 licit). Unlabelled transactions (77 percent) are not used by the tabular models; the graph networks use their features in message passing. The labels come from heuristics of Elliptic, not from court decisions.

## Evaluation

Test: labelled transactions of time steps 35 to 49 (1,083 illicit, 15,587 licit). The model was chosen on validation steps before the test steps were scored once.

| Metric | LightGBM, all features (selected) | Logistic regression (baseline) | Best graph network (GraphSAGE, local) |
| --- | ---: | ---: | ---: |
| Illicit F1 at the validation threshold | 0.799 (0.624 to 0.877) | 0.428 | 0.697 |
| Precision | 0.897 | 0.303 | 0.899 |
| Recall | 0.721 | 0.726 | 0.569 |
| PR AUC | 0.799 | 0.217 | 0.712 |
| ROC AUC | 0.937 | 0.847 | 0.894 |
| F1 at threshold 0.5 | 0.806 | 0.407 | 0.463 |

Interval: 95 percent, from resampling the 15 test steps with replacement.

## Subgroups

| Test steps | Labelled | Illicit | F1 | Comment |
| --- | ---: | ---: | ---: | --- |
| 35 to 42 | 9,983 | 914 | 0.895 | per step recall 0.63 to 0.97, precision 0.86 to 0.97 |
| 43 to 49 | 6,687 | 169 | 0.028 | after the dark market shutdown; 46 flags, 3 of them illicit |

## Limitations and risks

* **Change after step 43:** a large dark market closed at that step, and this model, like every other model tested, finds almost none of the illicit transactions after it. A model like this needs monitoring of the flag rate and retraining on recent labels.
* **Partial labels:** about 77 percent of transactions have no label and are not scored. The labelled ones need not represent them.
* **Validation differs from test:** validation F1 was 0.979; the validation steps hold more illicit labels (17 percent) than the test steps (6.5 percent).
* **Anonymized features:** the meaning of most features is not published, so explanations of a flag are limited to feature ranks and graph structure.
* **Neo4j scripts:** written and checked for structure and column names, not yet run against a server.

## Files

Written by `make train` under `artifacts/` and not committed: fitted models, probabilities per transaction, graph measures, the Neo4j export and `mlflow.db`. The repository publishes only aggregate metrics, figures and the ids of the four subgraph centers, which are public dataset ids.
