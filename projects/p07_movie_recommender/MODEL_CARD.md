# Model card: movie recommendations (project 7)

## Model

A two stage recommender that returns ten movies per user.

1. **Candidates:** the top 100 movies of implicit ALS (128 factors, regularization 0.5, 15 iterations, BM25 weighting) and the top 100 of a two tower model (PyTorch, 64 dimensions, 6 epochs, exact search in a FAISS flat index). Movies the user has rated are excluded.
2. **Ranking:** a LightGBM LambdaRank model (300 trees, 63 leaves, learning rate 0.05) reorders the union of both lists (170 movies per user on average) with 16 features: retrieval scores and ranks, movie popularity overall and over 90 days, mean rating, days in the catalog, release year, user activity, mean rating and idle days, genre affinity, and the gap between the movie's and the user's mean rating.

Training: CPU, 4 cores, PyTorch 2.14.1, implicit 0.7.3, FAISS 1.15.1, LightGBM 4.7.0; 40 minutes for both stages. Seed 42.

## Intended use

A portfolio demonstration of recommendation: candidate generation, learning to rank, time based offline evaluation and cold start analysis. **Not intended for a production service.** It was evaluated offline only; an offline gain does not prove that users would watch or like more movies.

## Training data

MovieLens 32M (GroupLens Research; research use with citation, no redistribution, no commercial use without permission): 32,000,204 ratings of 200,948 users on 87,585 movies, January 1995 to October 2023, with genres, titles and 2,000,072 user tags. The test stage fits on the 31,360,199 ratings before 20 February 2023. The ranking model is trained on candidates for the validation window (7 June 2022 to 20 February 2023), with retrieval models fitted on the ratings before it.

## Evaluation

Test window: 20 February to 13 October 2023 (the last 2 percent of ratings). Target: movies a user rated 4 or more in the window. 4,770 users with at least one earlier rating and one such movie.

| Metric | Selected model | Best baseline (popularity, last 90 days) |
| --- | ---: | ---: |
| NDCG at 10 | 0.134 ± 0.005 | 0.095 ± 0.004 |
| Recall at 10 | 0.092 ± 0.005 | 0.075 ± 0.005 |
| Hit rate at 10 | 0.488 | 0.413 |
| Catalog coverage at 10 | 1.9% | 0.2% |
| Novelty at 10 (bits) | 4.9 | 4.8 |

Paired by user, the NDCG gain over the baseline is +0.039 (95% interval 0.034 to 0.044). The model was chosen on the validation window before the test window was scored once.

## Subgroups

| Ratings before the window | Users | NDCG at 10, selected | NDCG at 10, best other model |
| --- | ---: | ---: | ---: |
| 1 to 9 | 46 | 0.238 | 0.273 (two tower) |
| 10 to 49 | 300 | 0.166 | 0.143 (two tower) |
| 50 to 199 | 1,198 | 0.128 | 0.104 (ALS) |
| 200 or more | 3,226 | 0.132 | 0.098 (popularity, 90 days) |

Users without any earlier rating (2,701) cannot be served by this model; the 90 day popularity list reaches NDCG 0.273 for them. Movies without any earlier rating cannot be candidates of ALS; the two tower model can place them by features alone (recall at 10 of 0.074 among new movies, against 0.006 for a random order).

## Limitations and risks

* **Popularity bias:** the strongest ranking feature is recent popularity. The top 10 lists cover 1.9 percent of the catalog, so most movies are never recommended and many users receive the same movies.
* **Offline target:** a rating in MovieLens is a choice to rate, not a view. A movie a user would like but never rates counts as a miss.
* **Population:** MovieLens users are volunteers on a research site; their tastes and rating habits need not match any other audience. Users active in 2022 and 2023 are a small and heavy rating subset (3,226 of the 4,770 test users had 200 or more ratings).
* **Two tower training stopped at the last epoch tried:** validation recall still rose at epoch 6, so the two tower results may understate what the model can do.
* **Tags** are free text written by users; the 1,000 most frequent are used as movie features without moderation.

## Files

Written by `make train` under `artifacts/` and not committed: `test/lists/*.npz` (top 100 lists per model), `test/two_tower/epoch_06.pt` (two tower weights), `mlflow.db` (scores). The repository publishes only aggregate metrics and three example lists.
