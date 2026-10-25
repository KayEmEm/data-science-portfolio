# Model card: inventory reorder policies (project 12)

## Models

Daily reorder policies for one product in one store, in a simulated environment with random demand, random lead times (2 to 6 days) and lost sales. Each policy maps the state (stock on hand, units due on each of the next 6 days, inventory position, weekday) to a target inventory position; the order is the gap to the position.

* **PPO** (Stable Baselines3 2.9.0): actor and critic with two layers of 64 units, 8 parallel environments, 512 steps per environment per update, learning rate 3e-4, discount 0.99, 200,704 steps per product, seed 42.
* **SAC** (Stable Baselines3 2.9.0): two layers of 64 units, learning rate 3e-4, replay buffer of 40,000 steps, 40,000 steps per product, seed 42.
* **Reference policies:** (s, S) and periodic review base stock (R, S), tuned by grid search on simulated years; and, for the slow product with a fixed lead time of 3 days, the optimal policy by backward induction.

The checkpoint with the lowest mean cost on 10 validation years is kept for each run. Training: CPU, 4 cores, about 8 minutes for the 9 runs of the reported sample run (`SAMPLE=0.2`).

## Intended use

A portfolio demonstration of reinforcement learning for inventory control: environment design, validation against the exact optimum, tuned classic baselines, and an honest comparison. **Not intended to place real orders.** The costs and lead times are assumptions, not data of any retailer.

## Training data

Simulated demand: negative binomial per day with one mean per weekday and one dispersion, fitted on the 52 weeks before the test year (M5 days 1241 to 1604) of three item x store series of store CA_1 (M5 Forecasting, Walmart, 2011 to 2016): FOODS_3_586 (48.1 units per day), FOODS_3_348 (7.1) and HOBBIES_1_010 (0.67). Costs: 10 $ per order, holding of 25 percent per year of the unit cost, lost margin per unit lost; unit cost 60 percent of the mean price; capacity 42 days of mean demand.

## Evaluation

Test: 100 simulated years per product, from a seed stream used by no other step; mean cost per year in dollars with 95 percent intervals, and paired differences against (s, S).

| Product | (s, S) | Base stock (R, S) | PPO | SAC |
| --- | ---: | ---: | ---: | ---: |
| fast | **329** (327 to 332) | 352 (+6.7%) | 370 (+12.3%) | 369 (+12.1%) |
| medium | **127** (125 to 129) | 162 (+27.3%) | 151 (+19.3%) | 153 (+20.5%) |
| slow | **110** (109 to 112) | 125 (+13.8%) | 129 (+16.9%) | 165 (+50.1%) |
| Replay of the real test year: fast, medium, slow | 327, 126, 90 | 411, 151, 105 | 372, 132, 117 | 364, 130, 147 |

Fixed lead time of 3 days, slow product, 400 simulated years: optimal policy 103.1 $ per year (102.5 expected by backward induction); (s, S) +3.6 percent, base stock +18.3 percent, PPO trained on this setting +24.7 percent.

## Limitations and risks

* **Short training, one seed:** the reported run is a sample run. The spread over seeds is not measured, and some learning curves still fall at the end. A full run (`make all`, about 4 hours on 4 cores) trains 5 seeds with 5 times the steps.
* **No RL policy beats (s, S) here,** and in the one setting with a known optimum, (s, S) is within 4 percent of it. A learned policy is worth its cost only where simple policies are far from optimal, for example with several products sharing capacity or a budget.
* **The RL policies order on one weekday** (PPO on Fridays, SAC on Mondays for the medium product) rather than at a reorder point. Such a policy reacts late to a demand peak early in its week.
* **Demand model:** M5 records sales, not demand, so days out of stock count as zero demand; the medium product has 17 percent zero days in the test year against 5 percent in the model. The model has no trend, yearly season, prices or events; the real test year sold about 13 percent less than the model for the fast and slow products.
* **Assumed costs and lead times:** the ranking of the policies depends on them. A different order cost or holding rate changes the tuned parameters and may change the gaps.

## Files

Written under `artifacts/` and not committed: the M5 tables, tuned baselines, trained models, evaluation tables and `mlflow.db`. The repository publishes aggregate results, figures and the animation.
