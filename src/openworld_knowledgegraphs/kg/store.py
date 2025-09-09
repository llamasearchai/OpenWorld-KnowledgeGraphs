from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import sqlite3
import re

from ..data.docs import export_docs


@dataclass
class Triple:
    s: str
    p: str
    o: str


class KGStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        from pathlib import Path
        p = Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def ensure_schema(self) -> None:
        conn = self._conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node TEXT UNIQUE,
                    display TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    s TEXT,
                    p TEXT,
                    o TEXT,
                    UNIQUE(s, p, o)
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_nodes(self, nodes: Iterable[Tuple[str, str]]) -> int:
        conn = self._conn()
        try:
            self.ensure_schema()
            added = 0
            for node, display in nodes:
                try:
                    conn.execute("INSERT INTO nodes(node, display) VALUES(?, ?)", (node, display))
                    added += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
            return added
        finally:
            conn.close()

    def upsert_edges(self, edges: Iterable[Triple]) -> int:
        conn = self._conn()
        try:
            self.ensure_schema()
            added = 0
            for t in edges:
                try:
                    conn.execute("INSERT INTO edges(s, p, o) VALUES(?, ?, ?)", (t.s, t.p, t.o))
                    added += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
            return added
        finally:
            conn.close()

    def neighbors(self, node: str, k: int = 5) -> List[Triple]:
        conn = self._conn()
        try:
            self.ensure_schema()
            cur = conn.execute(
                "SELECT s, p, o FROM edges WHERE s = ? LIMIT ?", (node, max(0, k))
            )
            return [Triple(*row) for row in cur.fetchall()]
        finally:
            conn.close()


_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9_-]*)\s+(builds|links|improves|supports)\s+([A-Z][A-Za-z0-9_-]*)\b")


def build_from_docs(db_path: str, min_count: int = 1) -> int:
    df = export_docs(db_path)
    store = KGStore(db_path)
    store.ensure_schema()
    added = 0
    for text in df["text"].astype(str).tolist():
        for match in _PATTERN.finditer(text):
            s, p, o = match.groups()
            added += store.upsert_nodes([(s, s), (o, o)])
            added += store.upsert_edges([Triple(s=s, p=p, o=o)])
    return added

