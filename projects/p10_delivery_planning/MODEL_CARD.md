# Model card: delivery routes and driver shifts (project 10)

## Models

Two optimization models, no learned parameters.

* **Routing:** vehicle routing with capacity and time windows in OR-Tools 9.15 (routing library). Main configuration: parallel cheapest insertion for the first solution, then guided local search for 60 seconds on one CPU core. Objective: total distance (DIMACS convention: distances truncated to one decimal), or the cost aware objective (distance, a fixed cost per vehicle, driver time per minute of route).
* **Driver shifts:** a mixed integer model in Pyomo 6.10. Each route is a duty for one driver; the model assigns duties to drivers or to an agency, and minimizes driver pay (paid span with a minimum of four hours, weekly guarantee, overtime) plus agency cost under availability, window, overlap, span, weekly hours and days rules. Solved with HiGHS 1.15, CBC (development build git 146ce89, from cbcbox 2.935) or Gurobi 13 (size limited license), one thread, relative gap 0.0001, 300 seconds.

Hardware: CPU, 4 cores, one search or MIP per core; about 100 minutes for all routing searches and 40 minutes for all MIPs.

## Intended use

A portfolio demonstration of mathematical optimization: a routing benchmark against best known solutions, a staffing model with three solvers, a greedy baseline, and a sensitivity study. **Not intended to plan real deliveries or staff real drivers:** the shift scenario, the driver pool and all cost rates are synthetic, and labor rules (breaks, rest between days, local law) are simplified.

## Data

The 56 Solomon VRPTW instances (100 customers each, six classes) with best known solutions from CVRPLIB, in VRPLIB format from the PyVRP instance collection (MIT license), pinned by commit and SHA256. Every best known solution was checked for feasibility and its distance recomputed. The shift scenario uses R101 to R107 as the seven days of one week and a synthetic pool of 30 drivers (seed 42).

## Evaluation

| Routing configuration (56 instances) | Mean gap to best known | Median | Largest | Instances at best known |
| --- | ---: | ---: | ---: | ---: |
| Cheapest insertion, first solution (baseline) | 52.9% | 54.5% | 94.3% | 0 |
| Guided local search, 10 s | 4.1% | 3.9% | 16.8% | 13 |
| Guided local search, 60 s (main) | 1.9% | 2.0% | 6.1% | 15 |
| Guided local search, 300 s | 0.9% | 0.5% | 5.7% | 18 |

| Shift model, full week (105 duties, 30 drivers) | Cost (CU) | Gap to lower bound | Seconds |
| --- | ---: | ---: | ---: |
| Greedy baseline | 27,924.5 | n/a | under 1 |
| HiGHS | 27,685.5 | 1.5% | 300 (limit) |
| CBC | 27,933.3 | 2.3% | 300 (limit) |
| Gurobi | not run: over the license size limit | | |

On two and four days all solvers that can run reach the same optimum in 1.2 seconds or less.

## Subgroups

| Instance class | Mean gap, 60 s | Mean gap, 300 s | At best known, 300 s |
| --- | ---: | ---: | ---: |
| C1 (clustered, short horizon) | 0.0% | 0.0% | 8 of 9 |
| C2 (clustered, long horizon) | 0.3% | 0.1% | 7 of 8 |
| R1 (random, short horizon) | 2.3% | 1.0% | 1 of 12 |
| R2 (random, long horizon) | 2.2% | 0.8% | 0 of 11 |
| RC1 (mixed, short horizon) | 4.4% | 2.9% | 0 of 8 |
| RC2 (mixed, long horizon) | 2.4% | 0.6% | 2 of 8 |

## Limitations and risks

* **Best known values follow one convention.** The gaps use DIMACS distances (truncated to one decimal) and distance as the only objective. The classic Solomon results rank by number of vehicles first; those values are not comparable with these gaps.
* **Time limits make runs vary.** Guided local search stops at a time limit, so results depend on the machine and on other load; four searches ran in parallel on four cores.
* **The full week is not solved to optimality.** The best roster has a gap of 1.5 percent to the lower bound; cost differences smaller than that are not established.
* **Synthetic scenario:** the mapping of depot hours onto 08:00 to 18:00, the driver pool, the wages, the agency rate, the vehicle and distance costs and the penalty for an unserved customer are assumptions. The fleet size result (the cost aware objective lowers the week's cost by 5.6 percent) depends on them.
* **Simplified labor rules:** no breaks, no minimum rest between days (the operating day leaves at least 13 hours), paid time is the shift span including idle time between duties.
* **Gurobi** ran only on the smallest scenario because of the size limit of its pip license.

## Files

Written by `make train` and `make evaluate` under `artifacts/` and not committed: one JSON file per routing search and per MIP, the duty, driver, roster and shift tables, and `mlflow.db`. The repository publishes aggregate tables, figures and three route maps.
