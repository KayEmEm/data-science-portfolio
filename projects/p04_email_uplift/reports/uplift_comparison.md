# Uplift model comparison

Test split: 19,200 customers, each arm compared with control. Intervals are 95 percent bootstrap intervals over test customers; for the random order they are the spread of the coefficient over random rankings, which is the noise floor a model must clear.

## Qini coefficient, visit (percentage points per treated customer)

| model                 | mens email               | womens email             |
|:----------------------|:-------------------------|:-------------------------|
| Random order          | -0.009 (-0.329 to 0.339) | -0.007 (-0.317 to 0.319) |
| Past spend rule       | 0.104 (-0.265 to 0.515)  | -0.109 (-0.433 to 0.235) |
| Category rule         | 0.203 (-0.156 to 0.576)  | 0.787 (0.463 to 1.138)   |
| T learner             | 0.002 (-0.391 to 0.376)  | 0.494 (0.136 to 0.864)   |
| X learner             | 0.058 (-0.304 to 0.432)  | 0.521 (0.202 to 0.890)   |
| Causal forest         | 0.076 (-0.297 to 0.427)  | 0.820 (0.529 to 1.189)   |
| DR learner (CausalML) | 0.068 (-0.289 to 0.447)  | 0.468 (0.125 to 0.817)   |

## Qini coefficient, spend (dollars per treated customer)

| model                 | mens email              | womens email             |
|:----------------------|:------------------------|:-------------------------|
| Random order          | 0.000 (-0.152 to 0.138) | -0.000 (-0.135 to 0.136) |
| Past spend rule       | 0.020 (-0.121 to 0.189) | 0.061 (-0.086 to 0.250)  |
| Category rule         | 0.040 (-0.108 to 0.222) | 0.038 (-0.077 to 0.176)  |
| T learner             | 0.075 (-0.073 to 0.240) | -0.001 (-0.158 to 0.142) |
| X learner             | 0.098 (-0.068 to 0.260) | 0.013 (-0.150 to 0.152)  |
| Causal forest         | 0.083 (-0.075 to 0.255) | 0.040 (-0.135 to 0.229)  |
| DR learner (CausalML) | 0.078 (-0.083 to 0.249) | 0.038 (-0.128 to 0.196)  |

## Model selection on the validation split

| model                 |   validation policy value (dollars per customer) |
|:----------------------|-------------------------------------------------:|
| T learner             |                                           0.3252 |
| X learner             |                                           0.2368 |
| Causal forest         |                                           0.3003 |
| DR learner (CausalML) |                                           0.2733 |

Selected model: T learner. Selected policy: Mens email to all.
