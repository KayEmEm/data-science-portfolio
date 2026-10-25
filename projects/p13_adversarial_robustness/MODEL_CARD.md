# Model card: defended variants of the models of projects 1, 6 and 8 (project 13)

## Models

Project 13 trains no new kind of model. It rebuilds three portfolio models with the code of their projects and adds defended variants:

* **Credit default (project 1):** LightGBM with isotonic calibration and a profit threshold. Defended variants: without the self reported fields (education, marital status); with adversarial training on manipulated copies of the training defaulters (one round, copies found against the undefended model, label kept at default); both.
* **Network flows (project 8):** supervised LightGBM (upper reference) and Isolation Forest (the project 8 selection), thresholds at 1% false alarms on normal validation flows. Defended variants: LightGBM without the TTL features; LightGBM with adversarial training (one random feasible change of each training attack flow added as an attack); Isolation Forest refitted after dropping the 5% highest scored training flows; LightGBM trained with Gaussian noise and smoothed over 100 noisy copies (randomized smoothing, sigma 0.25 and 0.5).
* **Road damage (project 6):** YOLO11n, weights of the project 6 run "full". Defended variants: JPEG compression (quality 75), 3x3 median filter, Gaussian noise (sigma 8/255) before the detector. The detector itself is not retrained.

Training: CPU, 4 cores; PyTorch 2.14.1, Ultralytics 8.4.184, LightGBM 4.7.0, scikit-learn 1.9.1, Adversarial Robustness Toolbox 1.20.1. The whole sample run takes about 30 minutes.

## Intended use

A portfolio demonstration of robustness testing: attacks within stated limits, a scorecard of clean against attacked performance, and the cost of each defense. **Not intended for production use and not a security certification** of any system. All attacks target only the portfolio's own models.

## Data

A seeded 0.2 sample of each project's data (see the README): 6,001 credit clients, 51,546 network flows, 579 Czech road images. The splits are those of the target projects.

## Evaluation (sample 0.2)

| Model | Clean | Under attack |
| --- | ---: | ---: |
| Credit, project 1 model | ROC AUC 0.770 | 24.1% of rejected applicants approved after a change |
| Credit, both defenses | ROC AUC 0.777 | 6.2% |
| Flows, LightGBM | recall 0.909 at 1% FPR | 0.075 after evasion |
| Flows, LightGBM with adversarial training | 0.898 | 0.874 |
| Flows, LightGBM without TTL | 0.892 | 0.809 |
| Flows, Isolation Forest, Generic attacks | 0.409 | 0.048 with 1% poisoned training traffic |
| Flows, Isolation Forest refitted after trimming | 0.491 | 0.493 at 1% poison; 0.000 at 5% |
| Road damage, YOLO11n | mAP50 0.504 | 0.121 (patch), 0.000 (PGD 2/255) |
| Road damage, median filter | 0.505 | 0.197 (patch), 0.003 (PGD 2/255) |

Full tables: [`reports/results.md`](reports/results.md).

## Limitations and risks

* **Small samples:** the image results rest on 100 test images with 44 boxes, and the credit results on 1,201 test clients. Differences of a few points in ROC AUC or mAP50 are within noise; no intervals were computed.
* **Attack limits are assumptions:** the payment budget, the flow change factors and the image epsilons are choices of this project. A stronger attacker (larger budget, other features, adaptive search) would do better; the defended numbers are therefore upper bounds.
* **No adaptive image attacks:** the image attacks do not know the input transformation. Attacks that account for it are known to defeat such defenses.
* **Feature recomputation of flows is approximate:** derived features are rescaled by ratios; jitter is assumed to scale with duration. A real change of traffic may move other features too.
* **Credit attack assumptions:** the extra payment is assumed to be borrowed, so it does not lower true risk. Where it is the client's own money, an approval after paying down debt is not an attack.
* **Defenses move the weakness:** adversarial training of the credit model alone raised the success of the self reported attack from 1.5% to 16.3%. Each defense must be tested against all attack sets, not only the one it targets.
* **Not covered:** prompt injection and data leakage of the project 2 system, and adversarial training of the detector.
