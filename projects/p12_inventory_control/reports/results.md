# Results: inventory control with reinforcement learning

Data: M5. Test: 100 simulated years per product; replay: the real test year with 40 lead time draws.

## Products and demand model

| Product   | Series             |   Mean units per day |   Dispersion |   Price ($) |   Lost margin per unit ($) |   Holding per unit and day ($) |   Capacity (units) |   Test year mean / model | Test days in 80% interval   |
|:----------|:-------------------|---------------------:|-------------:|------------:|---------------------------:|-------------------------------:|-------------------:|-------------------------:|:----------------------------|
| fast      | FOODS_3_586_CA_1   |              48.07   |       21.77  |      1.671  |                     0.6683 |                      0.0006866 |               2019 |                   0.8678 | 78.6%                       |
| medium    | FOODS_3_348_CA_1   |               7.093  |        2.026 |      0.8004 |                     0.3202 |                      0.0003289 |                298 |                   0.9674 | 74.2%                       |
| slow      | HOBBIES_1_010_CA_1 |               0.6742 |       10.89  |      2.97   |                     1.188  |                      0.001221  |                 29 |                   0.8699 | 95.6%                       |

## Policies

RL rows are the seed with the lowest validation cost. Against (s, S): mean paired difference of the test cost.

| Product   | Policy            | Test cost per year, 95% CI ($)   | Against (s, S)   | Fill rate   |   Orders per year |   Mean on hand (units) |   Replay cost ($) | Replay fill rate   |
|:----------|:------------------|:---------------------------------|:-----------------|:------------|------------------:|-----------------------:|------------------:|:-------------------|
| fast      | Base stock (R, S) | 352 (349 to 354)                 | +6.7%            | 100.0%      |              14.0 |                  825.6 |               411 | 99.7%              |
| fast      | (s, S)            | 329 (327 to 332)                 |                  | 100.0%      |              14.4 |                  721.1 |               327 | 99.9%              |
| fast      | PPO               | 370 (367 to 373)                 | +12.3%           | 99.9%       |              12.8 |                  928.3 |               372 | 99.9%              |
| fast      | SAC               | 369 (366 to 372)                 | +12.1%           | 100.0%      |              13.1 |                  933.8 |               364 | 100.0%             |
| medium    | Base stock (R, S) | 162 (161 to 162)                 | +27.3%           | 99.8%       |              14.0 |                  165.3 |               151 | 100.0%             |
| medium    | (s, S)            | 127 (125 to 129)                 |                  | 99.0%       |              10.2 |                  140.6 |               126 | 98.8%              |
| medium    | PPO               | 151 (149 to 154)                 | +19.3%           | 99.8%       |              13.1 |                  152.4 |               132 | 99.5%              |
| medium    | SAC               | 153 (151 to 155)                 | +20.5%           | 100.0%      |              13.4 |                  157.0 |               130 | 99.9%              |
| slow      | Base stock (R, S) | 125 (123 to 128)                 | +13.8%           | 90.3%       |               9.0 |                   13.4 |               105 | 96.8%              |
| slow      | (s, S)            | 110 (109 to 112)                 |                  | 97.2%       |               9.6 |                   13.6 |                89 | 98.9%              |
| slow      | PPO               | 129 (127 to 131)                 | +16.9%           | 99.2%       |              12.0 |                   14.9 |               117 | 99.9%              |
| slow      | SAC               | 165 (162 to 169)                 | +50.1%           | 99.9%       |              15.8 |                   16.0 |               147 | 100.0%             |

## Spread over seeds

| product   | algorithm   |   seeds |   test_mean |   test_sd |   test_min |   test_max |   pct_against_s_S |   beats_s_S |
|:----------|:------------|--------:|------------:|----------:|-----------:|-----------:|------------------:|------------:|
| fast      | ppo         |       1 |       369.9 |       nan |      369.9 |      369.9 |              12.3 |           0 |
| fast      | sac         |       1 |       369.2 |       nan |      369.2 |      369.2 |              12.1 |           0 |
| medium    | ppo         |       1 |       151.4 |       nan |      151.4 |      151.4 |              19.3 |           0 |
| medium    | sac         |       1 |       153.0 |       nan |      153.0 |      153.0 |              20.5 |           0 |
| slow      | ppo         |       1 |       128.9 |       nan |      128.9 |      128.9 |              16.9 |           0 |
| slow      | sac         |       1 |       165.4 |       nan |      165.4 |      165.4 |              50.1 |           0 |

`beats_s_S`: seeds whose paired 95% interval of the test cost difference lies below zero.

## Reward shaping (medium product, PPO)

| shaping   |   seeds |   test_cost |   sd |   fill |   orders |
|:----------|--------:|------------:|-----:|-------:|---------:|
| cost      |       1 |       151.4 |  nan |    1.0 |     13.1 |
| potential |       1 |       154.9 |  nan |    1.0 |     13.4 |
| service   |       1 |       156.3 |  nan |    1.0 |     13.6 |

* `cost`: negative daily cost (main runs)
* `potential`: plus a potential on the distance of the position to the middle of the tuned (s, S) band
* `service`: plus an extra penalty equal to the lost margin on each lost unit

## Validation against the exact optimum (slow product, lead time fixed at 3 days)

Expected cost of the optimal policy by backward induction: 102.47 $ per year. Simulated over 400 years:

| Policy                                    | Simulated cost per year ($)   | Above the optimum   | Parameters             |
|:------------------------------------------|:------------------------------|:--------------------|:-----------------------|
| Optimal (dynamic programming)             | 103.08 +- 0.81                |                     |                        |
| (s, S)                                    | 106.74 +- 0.80                | +3.6%               | {'s': 2, 'S': 29}      |
| Base stock (R, S)                         | 121.96 +- 1.13                | +18.3%              | {'R': 42, 'S': 29}     |
| PPO, seed with the lowest validation cost | 128.49 +- 0.94                | +24.7%              | slow_ppo_fixedlead_s42 |
