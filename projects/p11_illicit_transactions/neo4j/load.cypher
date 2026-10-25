// Load the export of `make neo4j-export` into Neo4j 5.26 or later.
// The two CSV files must sit in the import folder of the server (the docker command in the README
// mounts artifacts/neo4j there). Each statement ends with a semicolon.

// name: unique transaction ids
CREATE CONSTRAINT transaction_id IF NOT EXISTS
FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE;

// name: index on the class
CREATE INDEX transaction_class IF NOT EXISTS
FOR (t:Transaction) ON (t.class);

// name: transactions
LOAD CSV WITH HEADERS FROM 'file:///transactions.csv' AS row
CALL (row) {
  MERGE (t:Transaction {tx_id: toInteger(row.tx_id)})
  SET t.time_step = toInteger(row.time_step),
      t.split = row.split,
      t.class = row.class,
      t.score = toFloat(row.score),
      t.community = toInteger(row.community),
      t.pagerank = toFloat(row.pagerank),
      t.betweenness = toFloat(row.betweenness)
} IN TRANSACTIONS OF 10000 ROWS;

// name: payments
LOAD CSV WITH HEADERS FROM 'file:///payments.csv' AS row
CALL (row) {
  MATCH (a:Transaction {tx_id: toInteger(row.src_id)})
  MATCH (b:Transaction {tx_id: toInteger(row.dst_id)})
  MERGE (a)-[:PAYS]->(b)
} IN TRANSACTIONS OF 10000 ROWS;
