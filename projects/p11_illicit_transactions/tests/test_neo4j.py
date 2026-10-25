import re

from illicit_graph.config import NEO4J_DIR
from illicit_graph.neo4j_export import PAYMENT_COLUMNS, TRANSACTION_COLUMNS, statements


def test_cypher_files_split_into_named_statements():
    load = statements(NEO4J_DIR / "load.cypher")
    queries = statements(NEO4J_DIR / "queries.cypher")
    assert [name for name, _ in load] == [
        "unique transaction ids",
        "index on the class",
        "transactions",
        "payments",
    ]
    assert len(queries) == 6
    for _, body in load + queries:
        assert not body.endswith(";")
        assert body.count("(") == body.count(")")
        assert body.count("{") == body.count("}")


def test_load_script_reads_the_exported_columns():
    load = dict(statements(NEO4J_DIR / "load.cypher"))
    assert set(re.findall(r"row\.(\w+)", load["transactions"])) == set(TRANSACTION_COLUMNS)
    assert set(re.findall(r"row\.(\w+)", load["payments"])) == set(PAYMENT_COLUMNS)


def test_queries_use_only_loaded_properties():
    loaded = set(TRANSACTION_COLUMNS) - {"class"} | {"class"}
    for _, body in statements(NEO4J_DIR / "queries.cypher"):
        for prop in re.findall(r"\b[a-z]\.(\w+)", body):
            assert prop in loaded, prop
