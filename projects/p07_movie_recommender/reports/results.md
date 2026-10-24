# Project 7 results

Data: 32,000,204 ratings. Selected on validation NDCG at 10: **ALS + two tower candidates, LightGBM ranker**.

## Windows

| stage      | window_start        | window_end          |   history_ratings |   catalog_movies |   window_ratings |   liked_ratings |   users_scored |   users_without_history |   liked_ratings_on_new_movies |
|:-----------|:--------------------|:--------------------|------------------:|-----------------:|-----------------:|----------------:|---------------:|------------------------:|------------------------------:|
| validation | 2022-06-07 04:10:49 | 2023-02-20 06:26:51 |          30720195 |            76378 |           640004 |          307491 |           4949 |                    2811 |                        0.029  |
| test       | 2023-02-20 06:26:52 | 2023-10-13 02:29:07 |          31360199 |            80647 |           640005 |          309149 |           4770 |                    2701 |                        0.0248 |

## Test window (all scored users)

| model                                       |   users |   recall@10 |   recall@10_ci95 |   ndcg@10 |   ndcg@10_ci95 |   hit_rate@10 |   coverage@10 |   novelty@10 |
|:--------------------------------------------|--------:|------------:|-----------------:|----------:|---------------:|--------------:|--------------:|-------------:|
| ALS + two tower candidates, LightGBM ranker |    4770 |      0.0919 |           0.0048 |    0.1339 |         0.0053 |        0.4878 |        0.0187 |       4.9473 |
| Popularity, last 90 days (baseline)         |    4770 |      0.075  |           0.0046 |    0.095  |         0.0043 |        0.4126 |        0.0024 |       4.8456 |
| Implicit ALS                                |    4770 |      0.056  |           0.0036 |    0.0947 |         0.0047 |        0.37   |        0.0286 |       3.9525 |
| Two tower retrieval (FAISS)                 |    4770 |      0.0584 |           0.0039 |    0.0859 |         0.0043 |        0.3688 |        0.0336 |       5.3377 |
| Popularity, all time (baseline)             |    4770 |      0.0267 |           0.0026 |    0.0496 |         0.0035 |        0.2168 |        0.0043 |       1.6228 |

## Test NDCG differences, paired by user

| model             | reference         |   ndcg@10_difference |   ci95 |
|:------------------|:------------------|---------------------:|-------:|
| popularity        | popularity_recent |              -0.0454 | 0.0046 |
| popularity        | als               |              -0.0451 | 0.0047 |
| popularity_recent | als               |               0.0003 | 0.0056 |
| als               | popularity_recent |              -0.0003 | 0.0056 |
| two_tower         | popularity_recent |              -0.0091 | 0.0049 |
| two_tower         | als               |              -0.0087 | 0.0048 |
| ranker            | popularity_recent |               0.0389 | 0.0047 |
| ranker            | als               |               0.0392 | 0.0048 |

## Validation window (comparison half of the users)

| model                                       |   users |   recall@10 |   recall@10_ci95 |   ndcg@10 |   ndcg@10_ci95 |   hit_rate@10 |   coverage@10 |   novelty@10 |
|:--------------------------------------------|--------:|------------:|-----------------:|----------:|---------------:|--------------:|--------------:|-------------:|
| ALS + two tower candidates, LightGBM ranker |    2437 |      0.0974 |           0.0067 |    0.1487 |         0.0078 |        0.5236 |        0.0172 |       4.6018 |
| Popularity, last 90 days (baseline)         |    2437 |      0.0879 |           0.0071 |    0.1061 |         0.0063 |        0.4493 |        0.0023 |       3.9931 |
| Two tower retrieval (FAISS)                 |    2437 |      0.0579 |           0.005  |    0.0908 |         0.0062 |        0.3915 |        0.0291 |       5.2485 |
| Implicit ALS                                |    2437 |      0.0496 |           0.0047 |    0.0821 |         0.006  |        0.3455 |        0.0259 |       3.9446 |
| Popularity, all time (baseline)             |    2437 |      0.0236 |           0.0033 |    0.0458 |         0.0047 |        0.206  |        0.0036 |       1.627  |

## ALS settings (validation)

|   factors |   regularization |   alpha |   iterations |   recall@10 |   seconds |
|----------:|-----------------:|--------:|-------------:|------------:|----------:|
|        64 |             0.05 |       1 |           15 |      0.0487 |      61.8 |
|       128 |             0.05 |       1 |           15 |      0.0492 |     104.8 |
|       128 |             0.5  |       1 |           15 |      0.0496 |      76.7 |
|       256 |             0.5  |       1 |           15 |      0.0445 |     128.2 |

## Two tower epochs (validation sample)

|   epoch |   examples |   loss |   recall@10 |
|--------:|-----------:|-------:|------------:|
|       1 | 3.5798e+06 | 6.5479 |      0.0453 |
|       2 | 3.5798e+06 | 6.1612 |      0.0502 |
|       3 | 3.5798e+06 | 6.0479 |      0.0545 |
|       4 | 3.5798e+06 | 5.981  |      0.0508 |
|       5 | 3.5798e+06 | 5.9339 |      0.0542 |
|       6 | 3.5798e+06 | 5.9039 |      0.0579 |

## Cold start: users by history length (test)

| model             | ratings_before   |   users |   recall@10 |   ndcg@10 |
|:------------------|:-----------------|--------:|------------:|----------:|
| popularity        | 1 to 9           |      46 |      0.05   |    0.1423 |
| popularity        | 10 to 49         |     300 |      0.0517 |    0.1026 |
| popularity        | 50 to 199        |    1198 |      0.0256 |    0.0497 |
| popularity        | 200+             |    3226 |      0.0244 |    0.0433 |
| popularity_recent | 1 to 9           |      46 |      0.0777 |    0.1789 |
| popularity_recent | 10 to 49         |     300 |      0.0707 |    0.0975 |
| popularity_recent | 50 to 199        |    1198 |      0.0732 |    0.0838 |
| popularity_recent | 200+             |    3226 |      0.076  |    0.0977 |
| als               | 1 to 9           |      46 |      0.0694 |    0.2223 |
| als               | 10 to 49         |     300 |      0.0862 |    0.1382 |
| als               | 50 to 199        |    1198 |      0.0672 |    0.1044 |
| als               | 200+             |    3226 |      0.0489 |    0.0852 |
| two_tower         | 1 to 9           |      46 |      0.1181 |    0.2731 |
| two_tower         | 10 to 49         |     300 |      0.0825 |    0.1426 |
| two_tower         | 50 to 199        |    1198 |      0.0627 |    0.0909 |
| two_tower         | 200+             |    3226 |      0.0538 |    0.0761 |
| ranker            | 1 to 9           |      46 |      0.0937 |    0.2376 |
| ranker            | 10 to 49         |     300 |      0.0845 |    0.1656 |
| ranker            | 50 to 199        |    1198 |      0.0966 |    0.1277 |
| ranker            | 200+             |    3226 |      0.0909 |    0.1317 |
| popularity_recent | 0 (new users)    |    2701 |      0.0636 |    0.2725 |

## Cold start: movies first rated in the test window

| model               |   users |   new_movies |   share_of_liked_ratings_on_new_movies |   recall@10 |   ndcg@10 |   hit_rate@10 |
|:--------------------|--------:|-------------:|---------------------------------------:|------------:|----------:|--------------:|
| two_tower_new_items |    1941 |         1368 |                                 0.0248 |      0.0735 |    0.0418 |        0.151  |
| random_new_items    |    1941 |         1368 |                                 0.0248 |      0.0057 |    0.0044 |        0.0196 |

## Ranking model features

| feature          |     gain |   share |
|:-----------------|---------:|--------:|
| item_log_recent  | 33982.2  |  0.1678 |
| item_days_listed | 22277.2  |  0.11   |
| item_mean_rating | 18582    |  0.0918 |
| tt_rank          | 12689.2  |  0.0627 |
| als_score        | 12526.5  |  0.0619 |
| als_rank         | 12135.1  |  0.0599 |
| user_days_idle   | 11780.9  |  0.0582 |
| genre_affinity   | 11735.4  |  0.058  |
| item_log_ratings | 11111.8  |  0.0549 |
| tt_score         | 10511.3  |  0.0519 |
| item_year        |  9427.51 |  0.0466 |
| user_log_ratings |  8990.22 |  0.0444 |
| user_days_active |  8789.88 |  0.0434 |
| rating_gap       |  8653.78 |  0.0427 |
| user_mean_rating |  7188.91 |  0.0355 |
| in_both          |  2076.4  |  0.0103 |

## Run times (seconds)

| step              |   validation |   test |
|:------------------|-------------:|-------:|
| popularity        |          6.3 |    7.3 |
| popularity_recent |          7.2 |    5.9 |
| als               |         76.7 |   73.7 |
| two_tower         |        937.4 |  884.2 |
| ranker            |         17.4 |   28.9 |
