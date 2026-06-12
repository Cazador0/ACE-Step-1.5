"""RAPTOR-lite: a summary tree over a markdown vault with Best-of-N retrieval.

Leaves are heading-level chunks; parents are bottom-up summaries of groups of
children.  Retrieval is collapsed-tree cosine search over all levels.  Both
the summarizer and the embedder are injectable; the defaults are deterministic
(extractive summaries, hashed bag-of-words vectors) so no LLM is required.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from music_data_science.memory.markdown_chunks import Chunk, chunk_vault

EMBED_DIM = 128
_WORD = re.compile(r"[a-z0-9']+")


def default_embed(text: str) -> list[float]:
    """Deterministic hashed bag-of-words embedding, L2-normalized."""
    vector = [0.0] * EMBED_DIM
    for token in _WORD.findall(text.lower()):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        vector[int(digest, 16) % EMBED_DIM] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def default_summarize(texts: list[str]) -> str:
    """Extractive summary: the first sentence of each input, joined."""
    sentences = []
    for text in texts:
        first_line = next((line for line in text.splitlines() if line.strip()), "")
        sentence = re.split(r"(?<=[.!?])\s", first_line.strip(), maxsplit=1)[0]
        if sentence:
            sentences.append(sentence)
    return " ".join(sentences)[:800]


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 for zero vectors)."""
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


@dataclass
class Node:
    """One tree node: a chunk (level 0) or a summary of children (level > 0)."""

    node_id: int
    level: int
    text: str
    embedding: list[float]
    source: str = ""
    heading_path: tuple[str, ...] = ()
    authority: float = 0.5
    mtime: float = 0.0
    children: list[int] = field(default_factory=list)


@dataclass
class ScoredCandidate:
    """One retrieval candidate with its score components."""

    node: Node
    relevance: float
    recency: float
    authority: float
    total: float


@dataclass
class BestOfNResult:
    """Winner of a Best-of-N retrieval plus the full score trace."""

    winner: Optional[Node]
    score: float
    trace: list[ScoredCandidate]


class RaptorTree:
    """Summary tree over a markdown vault with collapsed-tree retrieval.

    Args:
        vault_dir: Directory of markdown files (an Obsidian vault).
        embed: Text embedding function; deterministic default needs no model.
        summarize: Group summarizer; extractive default needs no LLM.
        group_size: Children per parent when building summary levels.
        weights: (relevance, recency, authority) weights for Best-of-N scoring.
    """

    def __init__(
        self,
        vault_dir: str | Path,
        embed: Callable[[str], list[float]] = default_embed,
        summarize: Callable[[list[str]], str] = default_summarize,
        group_size: int = 4,
        weights: tuple[float, float, float] = (0.5, 0.2, 0.3),
    ) -> None:
        self.vault_dir = Path(vault_dir)
        self.embed = embed
        self.summarize = summarize
        self.group_size = max(2, group_size)
        self.weights = weights
        self.nodes: list[Node] = []

    def _make_node(self, level: int, text: str, **attrs: object) -> Node:
        """Create, embed, and register one node."""
        node = Node(node_id=len(self.nodes), level=level, text=text, embedding=self.embed(text), **attrs)  # type: ignore[arg-type]
        self.nodes.append(node)
        return node

    def _leaf(self, chunk: Chunk) -> Node:
        """Create a leaf node from a markdown chunk."""
        return self._make_node(
            0, chunk.text, source=chunk.source, heading_path=chunk.heading_path,
            authority=chunk.authority, mtime=chunk.mtime,
        )

    def build(self) -> "RaptorTree":
        """Chunk the vault and build summary levels bottom-up; returns self."""
        self.nodes = []
        current = [self._leaf(chunk) for chunk in chunk_vault(self.vault_dir)]
        while len(current) > self.group_size:
            parents = []
            for start in range(0, len(current), self.group_size):
                group = current[start:start + self.group_size]
                summary = self.summarize([node.text for node in group])
                parents.append(self._make_node(
                    group[0].level + 1, summary,
                    source=group[0].source,
                    authority=max(node.authority for node in group),
                    mtime=max(node.mtime for node in group),
                    children=[node.node_id for node in group],
                ))
            current = parents
        return self

    def retrieve(self, query: str, n: int = 5) -> list[tuple[Node, float]]:
        """Collapsed-tree retrieval: top ``n`` nodes (any level) by cosine similarity."""
        query_vector = self.embed(query)
        ranked = sorted(
            ((node, cosine(query_vector, node.embedding)) for node in self.nodes),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:n]

    def best_of_n(
        self,
        query: str,
        n: int = 5,
        scorer: Optional[Callable[[Node, float], float]] = None,
    ) -> BestOfNResult:
        """Score ``n`` retrieval candidates and return the winner with a trace.

        The default score combines relevance (cosine), recency (newest
        candidate = 1.0), and frontmatter source authority using the
        configured weights.  ``scorer`` overrides the combination entirely.
        """
        candidates = self.retrieve(query, n)
        if not candidates:
            return BestOfNResult(winner=None, score=0.0, trace=[])
        mtimes = [node.mtime for node, _ in candidates]
        low, high = min(mtimes), max(mtimes)
        spread = (high - low) or 1.0
        relevance_weight, recency_weight, authority_weight = self.weights
        trace = []
        for node, relevance in candidates:
            recency = (node.mtime - low) / spread if high > low else 1.0
            if scorer is not None:
                total = scorer(node, relevance)
            else:
                total = relevance_weight * relevance + recency_weight * recency + authority_weight * node.authority
            trace.append(ScoredCandidate(node, relevance, recency, node.authority, total))
        trace.sort(key=lambda candidate: candidate.total, reverse=True)
        return BestOfNResult(winner=trace[0].node, score=trace[0].total, trace=trace)
