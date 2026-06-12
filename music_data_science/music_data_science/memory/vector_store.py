"""Embedded vector store: SQLite + numpy brute-force cosine search.

Strategy A keeps durable vectors *embedded* next to the system of record
instead of running a vector server: embeddings live in an SQLite table and
search is a normalized matrix product over one namespace.  At vault scale
(thousands of chunks) this outperforms operating a separate service; if a
namespace ever outgrows brute force, ``sqlite-vec`` or LanceDB are drop-in
upgrades behind the same interface.

Also provides persistence for :class:`~music_data_science.memory.raptor.RaptorTree`
so the summary tree survives restarts without re-chunking the vault.
"""

from __future__ import annotations

import array
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from music_data_science.memory.raptor import Node, RaptorTree

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    namespace TEXT NOT NULL,
    ref       TEXT NOT NULL,
    text      TEXT NOT NULL DEFAULT '',
    metadata  TEXT NOT NULL DEFAULT '{}',
    dim       INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (namespace, ref)
);
CREATE INDEX IF NOT EXISTS vectors_namespace ON vectors (namespace);
"""


@dataclass
class SearchHit:
    """One nearest-neighbor result."""

    ref: str
    text: str
    metadata: dict[str, Any]
    score: float


def _pack(embedding: list[float]) -> bytes:
    """Serialize an embedding as little-endian float32 bytes."""
    return array.array("f", embedding).tobytes()


def _unpack(blob: bytes) -> list[float]:
    """Deserialize float32 bytes back into a list of floats."""
    values = array.array("f")
    values.frombytes(blob)
    return list(values)


class SQLiteVectorStore:
    """Vector store on a single SQLite file (or ``:memory:``).

    Args:
        path: SQLite database path; parent directories are created as needed.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path))
        self._connection.executescript(_SCHEMA)

    def add(
        self,
        namespace: str,
        ref: str,
        embedding: list[float],
        text: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Insert or replace one vector under ``(namespace, ref)``."""
        self._connection.execute(
            "INSERT OR REPLACE INTO vectors (namespace, ref, text, metadata, dim, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (namespace, ref, text, json.dumps(metadata or {}), len(embedding), _pack(embedding)),
        )
        self._connection.commit()

    def count(self, namespace: str) -> int:
        """Number of vectors stored in ``namespace``."""
        row = self._connection.execute(
            "SELECT COUNT(*) FROM vectors WHERE namespace = ?", (namespace,)
        ).fetchone()
        return int(row[0])

    def clear(self, namespace: str) -> None:
        """Delete every vector in ``namespace``."""
        self._connection.execute("DELETE FROM vectors WHERE namespace = ?", (namespace,))
        self._connection.commit()

    def search(self, namespace: str, query: list[float], n: int = 5) -> list[SearchHit]:
        """Top ``n`` vectors in ``namespace`` by cosine similarity to ``query``."""
        rows = self._connection.execute(
            "SELECT ref, text, metadata, embedding FROM vectors WHERE namespace = ? AND dim = ?",
            (namespace, len(query)),
        ).fetchall()
        if not rows:
            return []
        matrix = np.array([_unpack(blob) for _, _, _, blob in rows], dtype=np.float32)
        query_vector = np.array(query, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query_vector) or 1.0)
        norms[norms == 0.0] = 1.0
        scores = (matrix @ query_vector) / norms
        order = np.argsort(scores)[::-1][:n]
        return [
            SearchHit(ref=rows[i][0], text=rows[i][1], metadata=json.loads(rows[i][2]), score=float(scores[i]))
            for i in order
        ]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()


def persist_raptor(tree: RaptorTree, store: SQLiteVectorStore, namespace: str = "raptor") -> int:
    """Write every node of a built tree into ``store``; returns the node count."""
    store.clear(namespace)
    for node in tree.nodes:
        store.add(
            namespace,
            ref=str(node.node_id),
            embedding=node.embedding,
            text=node.text,
            metadata={
                "level": node.level,
                "source": node.source,
                "heading_path": list(node.heading_path),
                "authority": node.authority,
                "mtime": node.mtime,
                "children": node.children,
            },
        )
    return len(tree.nodes)


def restore_raptor(store: SQLiteVectorStore, tree: RaptorTree, namespace: str = "raptor") -> RaptorTree:
    """Load persisted nodes into ``tree`` (skipping the build step); returns it.

    The tree keeps its configured ``embed``/``summarize``/``weights``; only the
    node set is replaced, so retrieval works immediately without re-chunking.
    """
    rows = store._connection.execute(  # noqa: SLF001 -- companion function, same module
        "SELECT ref, text, metadata, embedding FROM vectors WHERE namespace = ? ORDER BY CAST(ref AS INTEGER)",
        (namespace,),
    ).fetchall()
    nodes = []
    for ref, text, metadata_json, blob in rows:
        metadata = json.loads(metadata_json)
        nodes.append(Node(
            node_id=int(ref),
            level=int(metadata["level"]),
            text=text,
            embedding=_unpack(blob),
            source=metadata.get("source", ""),
            heading_path=tuple(metadata.get("heading_path", ())),
            authority=float(metadata.get("authority", 0.5)),
            mtime=float(metadata.get("mtime", 0.0)),
            children=list(metadata.get("children", [])),
        ))
    tree.nodes = nodes
    return tree
