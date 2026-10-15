"""Seed the UC1 demo: a SQLite warehouse and, optionally, the Neo4j semantic layer.

    python demo/seed.py --warehouse demo/warehouse.db
    python demo/seed.py --warehouse demo/warehouse.db --neo4j bolt://127.0.0.1:7687 \
        --neo4j-user neo4j --neo4j-password-env SEMANTIC_NEO4J_PASSWORD

Then ``WAREHOUSE_URL=sqlite+aiosqlite:///demo/warehouse.db``. The semantic model
is ``demo/semantic.json`` (tables, columns, joins); ``semantic_cypher()`` turns
it into the ``(:Table)-[:HAS_COLUMN]->(:Column)`` / ``[:JOINS]`` graph the agent
queries.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

CUSTOMERS = [
    (1, "Nile Telecom", "EMEA"),
    (2, "Atlas Retail", "EMEA"),
    (3, "Sakura Foods", "APAC"),
    (4, "Pampas Energy", "AMER"),
    (5, "Fjord Logistics", "EMEA"),
]

ORDERS = [
    (1, 1, "2026-07-03", 1200.0, "shipped"),
    (2, 1, "2026-08-11", 800.0, "shipped"),
    (3, 2, "2026-07-21", 450.0, "open"),
    (4, 3, "2026-08-02", 2300.0, "shipped"),
    (5, 4, "2026-09-14", 990.0, "cancelled"),
    (6, 5, "2026-09-20", 1750.0, "shipped"),
    (7, 3, "2026-09-28", 640.0, "open"),
]

PRODUCTS = [(1, "Router X2", "network"), (2, "Antenna A7", "radio"), (3, "Cable Kit", "network")]


def seed_warehouse(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS orders; DROP TABLE IF EXISTS customers;
            DROP TABLE IF EXISTS products;
            CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, region TEXT);
            CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, ordered_on TEXT,
                                 amount REAL, status TEXT);
            CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT);
            """
        )
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", CUSTOMERS)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", ORDERS)
        conn.executemany("INSERT INTO products VALUES (?, ?, ?)", PRODUCTS)
    return path


def semantic_model() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((HERE / "semantic.json").read_text(encoding="utf-8"))
    return loaded


def semantic_cypher() -> list[tuple[str, dict[str, Any]]]:
    """``(query, params)`` statements that build the semantic graph."""
    model = semantic_model()
    statements: list[tuple[str, dict[str, Any]]] = [("MATCH (n) WHERE n:Table OR n:Column "
                                                     "DETACH DELETE n", {})]
    for table in model["tables"]:
        statements.append((
            "MERGE (t:Table {name: $table}) SET t.description = $description "
            "WITH t UNWIND $columns AS col "
            "MERGE (c:Column {table: $table, name: col.name}) "
            "SET c.type = col.type, c.description = col.description "
            "MERGE (t)-[:HAS_COLUMN]->(c)",
            table,
        ))
    for join in model["joins"]:
        statements.append((
            "MATCH (a:Table {name: $left}), (b:Table {name: $right}) "
            "MERGE (a)-[j:JOINS]->(b) SET j.on = $on",
            join,
        ))
    return statements


def seed_neo4j(uri: str, user: str, password: str) -> None:
    import neo4j

    with neo4j.GraphDatabase.driver(uri, auth=(user, password)) as driver:
        for query, params in semantic_cypher():
            driver.execute_query(query, params)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--warehouse", default=str(HERE / "warehouse.db"))
    parser.add_argument("--neo4j", default=None, help="bolt URI of the semantic layer")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password-env", default="SEMANTIC_NEO4J_PASSWORD")
    args = parser.parse_args()
    print(f"warehouse: {seed_warehouse(args.warehouse)}")
    if args.neo4j:
        seed_neo4j(args.neo4j, args.neo4j_user, os.environ[args.neo4j_password_env])
        print(f"semantic layer: {args.neo4j}")


if __name__ == "__main__":
    main()
