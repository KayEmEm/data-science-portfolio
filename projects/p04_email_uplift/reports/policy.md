# Targeting policies

Profit per customer = 30% margin on two week spend, less $0.10 per email sent. Both are assumptions. Values are estimated on the test split from the randomized arms (Horvitz Thompson), with paired bootstrap intervals.

| policy                       | profit per customer (95% CI)   | difference to Mens email to all (95% CI)   | share mens   | share womens   | share none   |
|:-----------------------------|:-------------------------------|:-------------------------------------------|:-------------|:---------------|:-------------|
| No email                     | 0.188 (0.119 to 0.261)         | -0.212 (-0.377 to -0.053)                  | 0%           | 0%             | 100%         |
| Mens email to all            | 0.396 (0.249 to 0.543)         | 0.000 (0.000 to 0.000)                     | 100%         | 0%             | 0%           |
| Womens email to all          | 0.259 (0.146 to 0.385)         | -0.141 (-0.316 to 0.047)                   | 0%           | 100%           | 0%           |
| Category rule                | 0.339 (0.208 to 0.484)         | -0.058 (-0.166 to 0.047)                   | 55%          | 45%            | 0%           |
| T learner policy             | 0.288 (0.162 to 0.396)         | -0.111 (-0.261 to 0.019)                   | 38%          | 32%            | 30%          |
| X learner policy             | 0.367 (0.217 to 0.513)         | -0.028 (-0.178 to 0.109)                   | 38%          | 32%            | 30%          |
| Causal forest policy         | 0.344 (0.221 to 0.480)         | -0.050 (-0.194 to 0.082)                   | 45%          | 29%            | 26%          |
| DR learner (CausalML) policy | 0.356 (0.230 to 0.500)         | -0.039 (-0.175 to 0.099)                   | 40%          | 32%            | 28%          |

## Sensitivity to the email cost

|   cost per email |   No email |   Mens email to all |   Category rule |   T learner policy |   share mailed by the model policy |
|-----------------:|-----------:|--------------------:|----------------:|-------------------:|-----------------------------------:|
|             0    |     0.1876 |              0.4955 |          0.44   |             0.3807 |                             0.7651 |
|             0.05 |     0.1876 |              0.4455 |          0.3895 |             0.3341 |                             0.7326 |
|             0.1  |     0.1876 |              0.3955 |          0.3391 |             0.2875 |                             0.6958 |
|             0.15 |     0.1876 |              0.3455 |          0.2887 |             0.2578 |                             0.6578 |
|             0.2  |     0.1876 |              0.2955 |          0.2383 |             0.2161 |                             0.616  |
|             0.25 |     0.1876 |              0.2455 |          0.1879 |             0.1812 |                             0.5734 |
|             0.3  |     0.1876 |              0.1955 |          0.1375 |             0.1492 |                             0.5318 |
|             0.4  |     0.1876 |              0.0955 |          0.0366 |             0.0933 |                             0.4485 |
|             0.5  |     0.1876 |             -0.0045 |         -0.0642 |             0.0749 |                             0.3778 |
