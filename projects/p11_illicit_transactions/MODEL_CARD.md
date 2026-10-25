# Model card: illicit transaction classifier (project 11)

**Status: pending the run on the Elliptic dataset.** The pipeline has run on synthetic graphs only. The selected model, its settings, the evaluation table and the subgroup results are written here after that run; until then no number in this card describes Elliptic.

## Model

Selected on validation F1 among: logistic regression on local features (baseline), LightGBM on local features, on all features, and on all features with structural graph measures, GCN and GraphSAGE on all features, and GraphSAGE on local features. Output: a probability that a transaction is illicit; the decision threshold is the one with the best F1 on the validation steps (31 to 34).

## Intended use

A portfolio demonstration of graph analytics for anti money laundering work: network description, community analysis, node classification on a split by time, and graph queries in Neo4j. **Not intended to decide on real transactions or people.** A flag is a lead for review, not evidence.

## Training data

Elliptic Data Set (Elliptic, 2019; research use): labelled transactions of time steps 1 to 30 for training, 31 to 34 for validation. The labels come from heuristics of Elliptic, not from court decisions.

## Evaluation

Test: labelled transactions of time steps 35 to 49. Metric: F1 on the illicit class at the validation threshold, with intervals from resampled time steps, and F1 per time step.

## Known limitations

* **Change after step 43:** a large dark market closed at that step, and published models lose most of their recall for illicit transactions after it. A model trained on older steps needs monitoring and retraining.
* **Partial labels:** about 77 percent of transactions have no label; the unlabelled ones are neither licit nor illicit for training and are not scored in the evaluation.
* **Anonymized features:** the meaning of most features is not published, so explanations of a flag are limited to graph structure and feature ranks.
* **Neo4j scripts:** written and checked for structure, not yet run against a server.
