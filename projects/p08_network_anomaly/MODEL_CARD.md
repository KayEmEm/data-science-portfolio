# Model card: network flow anomaly detection (project 8)

## Model

An Isolation Forest (scikit-learn, 300 trees, 512 flows per tree, seed 42) that scores each network flow by how easily it is isolated from the training flows. Higher scores are more unusual. It was selected among seven unsupervised detectors (robust z-score, Isolation Forest, LOF, HBOS, ECOD, COPOD, an autoencoder) and an ensemble, on validation PR AUC rescaled to 1 percent attacks.

Input: 36 numeric flow features (signed log, then standard scaling) and one hot columns for the most frequent protocol, service and state values, 70 columns in all. Addresses, ports, times and TCP sequence numbers are not used.

Alarm threshold: the score above which 1 percent of normal validation flows lie. A threshold without labels (top 1 percent of training sample scores) gives nearly the same result.

Training: CPU, 4 cores, scikit-learn 1.9.1, PyOD 3.6.7, PyTorch 2.14.1, LightGBM 4.7.0; under 1 second for Isolation Forest, about 5 minutes for all experiments.

## Intended use

A portfolio demonstration of anomaly detection without labels: detector comparison, thresholds at a fixed false alarm rate, analysis per attack type, the effect of attacks in the training data, and grouped alert triage. **Not intended for use on a production network.** It was evaluated offline on test bed traffic only.

## Training data

UNSW-NB15 official partition (Cyber Range Lab of UNSW Canberra, 2015; academic research use with citation): the training file, 175,341 flows, split 80 / 20 into train and validation by attack type. The detector is fitted without labels on all 44,800 normal training flows and 453 attacks drawn at random (1 percent of the sample). Labels were used only to build this sample and to score the result.

## Evaluation

Test file: 82,332 flows (37,000 normal, 45,332 attacks of nine types).

| Metric | Isolation Forest (selected) | Robust z-score (baseline) | LightGBM, supervised (upper reference) |
| --- | ---: | ---: | ---: |
| Recall at 1% FPR (threshold from validation) | 0.220 (0.216 to 0.223) | 0.216 | 0.930 |
| False alarm rate on test normal flows | 1.4% | 1.9% | 6.9% |
| PR AUC rescaled to 1% attacks | 0.150 (0.141 to 0.161) | 0.049 | 0.859 |
| PR AUC as observed (55% attacks) | 0.844 | 0.799 | 0.989 |
| ROC AUC | 0.815 | 0.785 | 0.984 |
| Precision of the top 1,000 alerts | 1.000 | 0.864 | 1.000 |

Intervals: 95 percent bootstrap over test flows. The detector was chosen on validation flows before the test file was scored once.

## Subgroups

Recall at 1 percent FPR per attack type (test file):

| Attack type | Test flows | Isolation Forest | Autoencoder | LightGBM |
| --- | ---: | ---: | ---: | ---: |
| Generic | 18,871 | 0.519 | 0.033 | 0.999 |
| Exploits | 11,132 | 0.006 | 0.319 | 0.977 |
| Fuzzers | 6,062 | 0.006 | 0.163 | 0.552 |
| DoS | 4,089 | 0.011 | 0.226 | 0.989 |
| Reconnaissance | 3,496 | 0.003 | 0.051 | 0.992 |
| Analysis | 677 | 0.000 | 0.464 | 0.923 |
| Backdoor | 583 | 0.007 | 0.554 | 0.997 |
| Shellcode | 378 | 0.000 | 0.021 | 0.915 |
| Worms | 44 | 0.000 | 0.159 | 0.977 |

The selected detector finds almost only Generic attacks. The autoencoder covers other types but with lower overall recall; a deployment would need both, or a detector chosen per traffic class.

## Limitations and risks

* **Low recall:** at a 1 percent false alarm rate, 78 percent of attack flows raise no alert. On a busy network, 1 percent of normal flows is still a large number of alerts.
* **Test bed data:** UNSW-NB15 was generated in 2015 in a lab, with synthetic normal traffic and attacks from one tool. Results need not hold on other networks or for current attacks.
* **Shift between files:** thresholds set on the training file flag 1.4 percent of normal test flows instead of 1 percent; a threshold must be checked on recent traffic.
* **Contaminated training data:** attacks hidden in the training sample lower detection, for the autoencoder most (recall at 1% FPR 0.44 when trained on normal flows only, 0.13 with 1 percent attacks). The training window should be checked for known incidents.
* **Repeated flows:** 103,576 flows repeat another one exactly and 7,745 test flows also occur in the training file, mostly Generic attacks. They are kept as published; they favor the supervised reference.
* **Evasion:** an attacker who knows the features can shape traffic to look normal. Project 13 tests this.

## Files

Written by `make train` under `artifacts/` and not committed: fitted detectors (`main/models/*.joblib`, autoencoder weights per epoch), scores per split, the preprocessor and `mlflow.db`. The repository publishes only aggregate metrics, figures and the grouped triage example.
