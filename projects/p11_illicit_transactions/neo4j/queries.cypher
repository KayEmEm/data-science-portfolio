// Example queries on the loaded graph. Each statement ends with a semicolon, and the comment line
// `// name:` gives the heading under which `make neo4j-load` writes its answer.

// name: Labelled transactions per time step
MATCH (t:Transaction)
WHERE t.class <> 'unknown'
RETURN t.time_step AS time_step,
       count(*) AS labelled,
       sum(CASE WHEN t.class = 'illicit' THEN 1 ELSE 0 END) AS illicit
ORDER BY time_step;

// name: Unlabelled transactions next to two or more illicit ones
MATCH (u:Transaction {class: 'unknown'})-[:PAYS]-(i:Transaction {class: 'illicit'})
WITH u, count(DISTINCT i) AS illicit_neighbors
WHERE illicit_neighbors >= 2
RETURN u.tx_id AS tx_id, u.time_step AS time_step, illicit_neighbors, round(u.score, 3) AS score
ORDER BY illicit_neighbors DESC, score DESC
LIMIT 20;

// name: Chains of four or more illicit transactions, each paying the next
MATCH p = (a:Transaction {class: 'illicit'})-[:PAYS*3..6]->(b:Transaction {class: 'illicit'})
WHERE all(n IN nodes(p) WHERE n.class = 'illicit')
  AND NOT EXISTS { MATCH (:Transaction {class: 'illicit'})-[:PAYS]->(a) }
RETURN a.tx_id AS first_tx, b.tx_id AS last_tx, a.time_step AS time_step, length(p) + 1 AS transactions
ORDER BY transactions DESC
LIMIT 20;

// name: Highest scored unlabelled transactions of the test period
MATCH (t:Transaction {class: 'unknown', split: 'test'})
OPTIONAL MATCH (t)-[:PAYS]-(n:Transaction)
RETURN t.tx_id AS tx_id, t.time_step AS time_step, round(t.score, 3) AS score,
       count(n) AS neighbors,
       sum(CASE WHEN n.class = 'illicit' THEN 1 ELSE 0 END) AS illicit_neighbors
ORDER BY score DESC
LIMIT 20;

// name: Communities with the most illicit transactions
MATCH (t:Transaction)
WITH t.community AS community, min(t.time_step) AS time_step, count(*) AS size,
     sum(CASE WHEN t.class = 'illicit' THEN 1 ELSE 0 END) AS illicit,
     sum(CASE WHEN t.class <> 'unknown' THEN 1 ELSE 0 END) AS labelled
WHERE illicit > 0
RETURN community, time_step, size, labelled, illicit,
       round(toFloat(illicit) / labelled, 3) AS illicit_share_of_labelled
ORDER BY illicit DESC
LIMIT 15;

// name: Two hop neighborhood of the illicit transaction with the highest PageRank
MATCH (i:Transaction {class: 'illicit'})
WITH i ORDER BY i.pagerank DESC LIMIT 1
MATCH (i)-[:PAYS*1..2]-(n:Transaction)
WITH i, n.class AS class, count(DISTINCT n) AS transactions
RETURN i.tx_id AS center, i.time_step AS time_step, class, transactions
ORDER BY transactions DESC;
