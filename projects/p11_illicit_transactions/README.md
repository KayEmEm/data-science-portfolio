# Project 11: Illicit transactions in a payment network

Find illicit Bitcoin transactions and the clusters around them in the Elliptic transaction graph: describe the network with statistics, centrality and communities, then classify transactions with graph neural networks (GCN, GraphSAGE) and compare them with LightGBM models that see only the transaction features, on a split by time.

## Headline result

**Pending.** The pipeline, tests and CI run on a synthetic graph. The results on the Elliptic dataset follow once the files are in the build environment; this section, the figures and [`MODEL_CARD.md`](MODEL_CARD.md) are filled in then. Numbers from `make smoke` come from synthetic data and say nothing about Elliptic.

The main metric is F1 on the illicit class over the 15 test time steps, with the decision threshold chosen on validation steps, as a review team works through a fixed list of flagged transactions and both missed and false flags cost time. Also reported: precision, recall, PR AUC, F1 per time step, F1 before and from the dark market shutdown at step 43, and F1 at a threshold of 0.5 (the convention of the published benchmark).

## Dataset and license

[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (Elliptic, 2019): a graph of Bitcoin transactions, where a node is a transaction and a directed edge a flow of bitcoins from one transaction to the next. The numbers below are recalled from the dataset description; `make data` counts them in the files and records whether they match:

* about 203,769 transactions in 49 time steps (about two weeks apart), and 234,355 edges. Edges join transactions of the same time step only, so each step is its own set of graphs;
* 166 features per transaction: the time step, 93 local features (inputs and outputs, fees, volume and others; anonymized) and 72 features aggregated over the one hop neighborhood;
* labels: 4,545 illicit (2 percent), 42,019 licit (21 percent), the rest unlabelled.

Cite: M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson and C. E. Leiserson, 2019, Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics, KDD Workshop on Anomaly Detection in Finance.

**License:** to my knowledge the Kaggle page lists the data for research use and does not grant redistribution. The page could not be opened from the build environment, so these terms were not checked there. This repository does not include the data.

Kaggle downloads need a login, so `make data` reads the files from `data/raw/` (or `SOURCE=<folder>`): `elliptic_txs_features.csv`, `elliptic_txs_classes.csv` and `elliptic_txs_edgelist.csv`. `SOURCE` may also be the Kaggle zip as downloaded.

## Approach

1. **Data** (`data.py`): the three files are read and checked (167 columns, time steps 1 to 49, class values `1`, `2` or `unknown`, every edge between known transactions). Duplicate edges and self loops are counted and dropped; edges that join two time steps are counted. Counts are compared with the documented ones and recorded in `data/audit.json`.
2. **Split by time:** train on steps 1 to 30, validate on 31 to 34, test on 35 to 49. Steps 35 to 49 are the test period of the published benchmark. As no edge joins two steps, no message passes between the periods, and no model sees a later transaction during training.
3. **Network analysis** (`network.py`, NetworkX): per time step, density, weakly connected components, degree assortativity, clustering and Louvain communities with their modularity; per transaction, in and out degree, PageRank, betweenness (estimated from 256 source transactions per step), clustering coefficient, core number and component size. Each measure is compared between illicit and licit transactions, also as the only score (ROC AUC). Two checks test whether illicit transactions cluster: the share of edges with an illicit end whose other end is illicit too, and the share of illicit transactions in communities that are mostly illicit, each against labels shuffled within the time step.
4. **Models** (`train.py`), all trained on the labelled transactions of the training steps:
   * logistic regression on the local features (baseline);
   * LightGBM on the local features, and on the local and aggregated features (the feature only reference); early stopping on validation;
   * LightGBM on all features and the structural measures of step 3 (no labels are used for these);
   * GCN and GraphSAGE (`gnn.py`, PyTorch Geometric, two layers of 128 units, edges in both directions, class weights by inverse frequency, early stopping on validation PR AUC, three seeds each, mean probability over seeds) on all features, and GraphSAGE on the local features only, to see whether message passing recovers what the aggregated features hold.
5. **Evaluation** (`evaluate.py`, `metrics.py`): the threshold with the best F1 on the validation steps, then test F1, precision and recall at that threshold, F1 per time step and before and from step 43. Intervals come from resampling the 15 test steps with replacement (200 times), because transactions of one step share their graph and period. Every model is also compared with the feature only LightGBM model as a paired difference over the same resampled steps. The model with the best validation F1 is selected.
6. **Subgraphs** (`report.py`): two hop neighborhoods around illicit test transactions, colored by label, with the transactions the selected model flags.
7. **Neo4j** (`neo4j_export.py`, `neo4j/`): an export of transactions (with class, model score, community, PageRank) and payments as CSV, a load script with `LOAD CSV` and six example Cypher queries: labelled transactions per step, unlabelled transactions next to two or more illicit ones, chains of illicit transactions, the highest scored unlabelled test transactions, communities with the most illicit transactions and the neighborhood of the most central illicit one. The scripts were written for Neo4j 5.26 and later; a Neo4j server could not be run in the build environment, so they are checked by tests for structure and column names, not yet run against a server.
8. **Tracking:** metrics are logged to a local MLflow store under `artifacts/` (`mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db`).

## Reproduce

From this folder, with Python 3.11 or later:

```bash
make install                      # this project and the shared common package
make all SOURCE=path/to/archive.zip  # or the CSV folder, or place the files in data/raw/
make smoke                        # the same pipeline on a synthetic graph, no data needed
```

`make train` stores every finished model (and every seed of a graph network) under `artifacts/` and skips it when run again, so a stopped run resumes.

To load the graph into Neo4j (Docker, from this folder, after `make all`):

```bash
docker run -d --name p11-neo4j -p 7474:7474 -p 7687:7687 \
  -v "$PWD/artifacts/neo4j:/import" -e NEO4J_AUTH=neo4j/<password> neo4j:5.26
pip install -e ".[neo4j]"
NEO4J_PASSWORD=<password> make neo4j-load   # runs neo4j/load.cypher, then neo4j/queries.cypher
```

The answers of the queries are written to `artifacts/neo4j/answers.md`. The queries can also be pasted into the Neo4j Browser at `http://localhost:7474`.

## Layout

```
src/illicit_graph/  config, data, network, gnn, metrics, train, evaluate, report,
                    neo4j_export, synthetic
neo4j/              load.cypher, queries.cypher
tests/              unit tests (data checks, network measures, metrics, models, Cypher files) and an end to end test
reports/            results.md, results.json, figures (after the run on Elliptic)
```

## Cost

No paid service used: 0 CAD.
