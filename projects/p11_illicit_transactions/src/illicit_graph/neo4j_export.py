"""Export the graph for Neo4j, and load it into a running server.

``export`` writes two CSV files with a header row under ``artifacts/neo4j/``, in the form that
``neo4j/load.cypher`` reads with ``LOAD CSV``:

* ``transactions.csv``: ``tx_id, time_step, split, class, score, community, pagerank, betweenness``,
  where ``class`` is ``illicit``, ``licit`` or ``unknown`` and ``score`` is the illicit probability
  of the selected model;
* ``payments.csv``: ``src_id, dst_id``, one row per directed edge.

``load`` (needs the ``neo4j`` extra and a server, see the README) runs ``neo4j/load.cypher`` and then
each query of ``neo4j/queries.cypher``, and writes the first rows of every answer to
``artifacts/neo4j/answers.md``.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from illicit_graph.config import ARTIFACTS_DIR, DATA_DIR, ILLICIT, LICIT, NEO4J_DIR
from illicit_graph.data import load
from illicit_graph.network import load_metrics

TRANSACTION_COLUMNS = [
    "tx_id",
    "time_step",
    "split",
    "class",
    "score",
    "community",
    "pagerank",
    "betweenness",
]
PAYMENT_COLUMNS = ["src_id", "dst_id"]
CLASS_NAMES = {ILLICIT: "illicit", LICIT: "licit"}


def export(data_dir: Path, artifacts: Path) -> Path:
    nodes, edges = load(data_dir)
    metrics = load_metrics(artifacts)
    selection = json.loads((artifacts / "evaluation" / "selection.json").read_text())
    scores = np.load(artifacts / "models" / selection["model"] / "scores.npy")
    out = artifacts / "neo4j"
    out.mkdir(parents=True, exist_ok=True)
    transactions = pd.DataFrame(
        {
            "tx_id": nodes["tx_id"],
            "time_step": nodes["time_step"].astype(int),
            "split": nodes["split"].astype(str),
            "class": nodes["label"].map(CLASS_NAMES).fillna("unknown"),
            "score": scores.round(5),
            "community": metrics["community"],
            "pagerank": metrics["pagerank"].round(5),
            "betweenness": metrics["betweenness"].round(7),
        }
    )
    transactions[TRANSACTION_COLUMNS].to_csv(out / "transactions.csv", index=False)
    edges[PAYMENT_COLUMNS].to_csv(out / "payments.csv", index=False)
    return out


def statements(path: Path) -> list[tuple[str, str]]:
    """Statements of a Cypher file, each with the comment line that names it (``// name: ...``)."""
    out, name, body = [], None, []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            if match := re.match(r"//\s*name:\s*(.+)", stripped):
                name = match.group(1).strip()
            continue
        if stripped:
            body.append(line.rstrip())
        if stripped.endswith(";"):
            text = "\n".join(body)[:-1].strip()
            out.append((name or f"statement {len(out) + 1}", text))
            name, body = None, []
    if body:
        raise ValueError(f"{path.name} ends with a statement without a semicolon")
    return out


def load_into_server(artifacts: Path, uri: str, user: str, password: str, limit: int = 10) -> Path:
    from neo4j import GraphDatabase  # optional extra

    answers = ["# Answers of the example Cypher queries", ""]
    with GraphDatabase.driver(uri, auth=(user, password)) as driver, driver.session() as session:
        for _, body in statements(NEO4J_DIR / "load.cypher"):
            session.run(body).consume()
        for name, body in statements(NEO4J_DIR / "queries.cypher"):
            records = list(session.run(body))
            frame = pd.DataFrame([r.data() for r in records[:limit]])
            answers += [f"## {name}", "", "```cypher", body, "```", ""]
            answers += [frame.to_markdown(index=False) if len(frame) else "(no rows)", ""]
    path = artifacts / "neo4j" / "answers.md"
    path.write_text("\n".join(answers))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "load"])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    args = parse_args(parser, __file__)
    if args.action == "export":
        out = export(args.data_dir, args.artifacts)
        print(f"wrote {out / 'transactions.csv'} and {out / 'payments.csv'}")
    else:
        password = os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise SystemExit("set NEO4J_PASSWORD to the password of the Neo4j server")
        print(f"wrote {load_into_server(args.artifacts, args.uri, args.user, password)}")


if __name__ == "__main__":
    main()
