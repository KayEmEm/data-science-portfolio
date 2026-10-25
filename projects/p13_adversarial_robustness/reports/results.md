# Project 13 results (sample 0.2)

## Credit model (project 1)

Test clients: 1,201. Budget: extra payment up to 10% of the credit limit in the latest month; any education and marital status.

| Model                        |   ROC AUC |   Approval rate |   Rejections turned (payment) |   Rejections turned (self reported) |   Rejections turned (both) |   Defaulters among them turned |   Profit clean (M NT$) |   Profit attacked (M NT$) |
|:-----------------------------|----------:|----------------:|------------------------------:|------------------------------------:|---------------------------:|-------------------------------:|-----------------------:|--------------------------:|
| Project 1 model              |     0.770 |           0.619 |                         0.230 |                               0.015 |                      0.241 |                          0.084 |                  6.950 |                     7.168 |
| Without self reported fields |     0.785 |           0.703 |                         0.182 |                               0.000 |                      0.182 |                          0.082 |                  7.099 |                     6.681 |
| Adversarial training         |     0.786 |           0.674 |                         0.066 |                               0.163 |                      0.219 |                          0.125 |                  7.277 |                     6.490 |
| Both defenses                |     0.777 |           0.678 |                         0.062 |                               0.000 |                      0.062 |                          0.017 |                  7.015 |                     7.066 |

## Flow detectors (project 8)

Initial TTL values the attacker tries: [31, 254, 0].

| Detector                               |   Recall at 1% FPR |   Test FPR |   Detected flows attacked |   Evaded (size and timing) |   Evaded (TTL) |   Evaded (all) |   Recall after attack |
|:---------------------------------------|-------------------:|-----------:|--------------------------:|---------------------------:|---------------:|---------------:|----------------------:|
| LightGBM (project 8 reference)         |              0.909 |      0.043 |                      2000 |                      0.096 |          0.908 |          0.917 |                 0.075 |
| LightGBM without TTL features          |              0.892 |      0.037 |                      2000 |                      0.093 |          0.000 |          0.093 |                 0.809 |
| LightGBM, adversarial training         |              0.898 |      0.037 |                      2000 |                      0.026 |          0.001 |          0.027 |                 0.874 |
| Isolation Forest (project 8 selection) |              0.174 |      0.011 |                      1581 |                      0.271 |          0.000 |          0.271 |                 0.127 |

Unconstrained reference (HopSkipJump, any feature): 100.0% of 200 detected flows evade LightGBM, median L2 change 2.46 in the standardized space.

### Poisoning of Isolation Forest

|   Poison share |   Poison flows |   Generic recall (none) |   Other recall (none) |   Generic recall (trim) |   Other recall (trim) |   Generic recall (dedup) |   Other recall (dedup) |
|---------------:|---------------:|------------------------:|----------------------:|------------------------:|----------------------:|-------------------------:|-----------------------:|
|          0.000 |          0.000 |                   0.409 |                 0.007 |                   0.491 |                 0.002 |                    0.447 |                  0.003 |
|          0.010 |         91.000 |                   0.048 |                 0.007 |                   0.493 |                 0.006 |                    0.091 |                  0.006 |
|          0.050 |        476.000 |                   0.000 |                 0.007 |                   0.000 |                 0.005 |                    0.000 |                  0.006 |
|          0.100 |       1006.000 |                   0.000 |                 0.010 |                   0.000 |                 0.005 |                    0.000 |                  0.007 |

### Randomized smoothing of LightGBM (numeric features, standardized log scale)

Largest feasible change of size and timing, median L2: 1.42.

|   Sigma |   Accuracy |   Recall |   FPR |   Abstain |   Certified at radius 0 |   Certified at radius 0.25 |   Certified at radius 0.5 |   Certified at radius 1 |
|--------:|-----------:|---------:|------:|----------:|------------------------:|---------------------------:|--------------------------:|------------------------:|
|   0.250 |      0.852 |    0.986 | 0.315 |     0.041 |                   0.832 |                      0.791 |                     0.000 |                   0.000 |
|   0.500 |      0.827 |    0.992 | 0.377 |     0.036 |                   0.812 |                      0.796 |                     0.759 |                   0.000 |

## Road damage detector (project 6)

Test images: 100 (34 with damage), weights best.pt, input 512 pixels.

| Attack                       |   none |   JPEG 75 |   median 3x3 |   Gaussian noise |
|:-----------------------------|-------:|----------:|-------------:|-----------------:|
| No attack                    |  0.504 |     0.485 |        0.505 |            0.442 |
| Random noise patch (control) |  0.500 |     0.482 |        0.510 |            0.433 |
| Adversarial patch            |  0.121 |     0.096 |        0.197 |            0.060 |
| FGSM, epsilon 2/255          |  0.013 |     0.045 |        0.055 |            0.041 |
| PGD, epsilon 2/255           |  0.000 |     0.035 |        0.003 |            0.004 |
| FGSM, epsilon 8/255          |  0.005 |     0.013 |        0.007 |            0.021 |
| PGD, epsilon 8/255           |  0.000 |     0.000 |        0.000 |            0.000 |

## Defense trade offs

| Target                           | Defense                        | Clean metric     |   Clean change | Attack metric                  |   Attacked before |   Attacked after |
|:---------------------------------|:-------------------------------|:-----------------|---------------:|:-------------------------------|------------------:|-----------------:|
| Credit model (project 1)         | Without self reported fields   | ROC AUC          |          0.015 | rejections turned (both)       |             0.241 |            0.182 |
| Credit model (project 1)         | Adversarial training           | ROC AUC          |          0.016 | rejections turned (both)       |             0.241 |            0.219 |
| Credit model (project 1)         | Both defenses                  | ROC AUC          |          0.006 | rejections turned (both)       |             0.241 |            0.062 |
| Flow detector (project 8)        | LightGBM without TTL features  | recall at 1% FPR |         -0.017 | recall after evasion           |             0.075 |            0.809 |
| Flow detector (project 8)        | LightGBM, adversarial training | recall at 1% FPR |         -0.010 | recall after evasion           |             0.075 |            0.874 |
| Road damage detector (project 6) | JPEG 75                        | mAP50            |         -0.019 | mAP50 under Adversarial patch  |             0.121 |            0.096 |
| Road damage detector (project 6) | median 3x3                     | mAP50            |          0.001 | mAP50 under Adversarial patch  |             0.121 |            0.197 |
| Road damage detector (project 6) | Gaussian noise                 | mAP50            |         -0.062 | mAP50 under Adversarial patch  |             0.121 |            0.060 |
| Road damage detector (project 6) | JPEG 75                        | mAP50            |         -0.019 | mAP50 under PGD, epsilon 2/255 |             0.000 |            0.035 |
| Road damage detector (project 6) | median 3x3                     | mAP50            |          0.001 | mAP50 under PGD, epsilon 2/255 |             0.000 |            0.003 |
| Road damage detector (project 6) | Gaussian noise                 | mAP50            |         -0.062 | mAP50 under PGD, epsilon 2/255 |             0.000 |            0.004 |
