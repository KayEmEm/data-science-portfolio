# Project 7: Movie recommendations

Recommend ten movies per user from the rating history of MovieLens 32M, and compare candidate generation and ranking approaches on a time based split.

**Status: pipeline complete and tested on synthetic data; results on MovieLens 32M pending.**

## Approach

1. **Data** (`data.py`): ratings, movies and tags of the MovieLens 32M zip, checked against the counts of the release. The last 2 percent of ratings by time form the test window, the 2 percent before them the validation window.
2. **Target:** the movies a user rates 4 or more in the window. Every model sees only the ratings before the window and never recommends a movie the user has rated.
3. **Baselines** (`baselines.py`): the most rated movies over the whole history and over its last 90 days.
4. **Implicit ALS** (`als.py`): every rating as a view, BM25 weighting, four settings compared on validation.
5. **Two tower retrieval** (`two_tower.py`): PyTorch towers for users (recent history, genre profile) and movies (id, genres, decade, tags), trained with an in batch softmax with log frequency correction; exact search in a FAISS index. Hidden ids in training let the movie tower place movies without ratings.
6. **Ranking** (`ranker.py`, `rank.py`): LightGBM LambdaRank on the union of the ALS and two tower top 100, with retrieval scores, popularity, rating and genre affinity features. Trained on half of the validation users, compared on the other half.
7. **Evaluation** (`metrics.py`, `evaluate.py`): recall, NDCG and hit rate at 10, catalog coverage and novelty; cold start by user history length, for users without history, and for movies first rated in the test window.
8. **Demo** (`demo/app.py`): a Streamlit page that shows a user's history and the top 10 of every model.

Planned but left out: LightFM (its build fails on Python 3.13); the two tower model with item features covers the hybrid content and collaborative case.

## Reproduce

From this folder, with Python 3.11 or later:

```bash
make install                         # this project, the demo extra and the shared common package
make all SOURCE=path/to/ml-32m.zip   # or without SOURCE to download from GroupLens
make smoke                           # the same pipeline on a synthetic archive, no download
```

`make train` stores every finished step under `artifacts/` and skips it when run again, so a stopped run resumes; the two tower model also saves each epoch.

## Dataset and license

[MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) (GroupLens Research, University of Minnesota). The GroupLens usage license allows research use, requires a citation (Harper and Konstan 2015) and forbids redistribution without permission, so this repository does not include the data.

## Cost

No paid service used: 0 CAD.
