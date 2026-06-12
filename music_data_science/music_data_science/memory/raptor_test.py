"""Tests for the RAPTOR-lite summary tree and Best-of-N retrieval."""

import os
import tempfile
import unittest
from pathlib import Path

from music_data_science.memory.markdown_chunks import chunk_markdown
from music_data_science.memory.raptor import (
    BestOfNResult,
    RaptorTree,
    cosine,
    default_embed,
    default_summarize,
)


def _write_vault(directory: Path, notes: dict[str, str]) -> None:
    for name, text in notes.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


class DefaultEmbedTest(unittest.TestCase):
    """The fallback embedding is deterministic and L2-normalized."""

    def test_deterministic(self):
        self.assertEqual(default_embed("drums and bass"), default_embed("drums and bass"))

    def test_normalized(self):
        vector = default_embed("syncopated drums")
        self.assertAlmostEqual(sum(value * value for value in vector), 1.0, places=6)

    def test_empty_text_is_zero_vector(self):
        self.assertEqual(sum(default_embed("")), 0.0)

    def test_cosine_of_zero_vector_is_zero(self):
        self.assertEqual(cosine(default_embed(""), default_embed("drums")), 0.0)


class DefaultSummarizeTest(unittest.TestCase):
    """The extractive summarizer joins first sentences."""

    def test_first_sentences_joined(self):
        summary = default_summarize(["Drums drive the groove. More detail.", "Bass anchors harmony."])
        self.assertIn("Drums drive the groove.", summary)
        self.assertIn("Bass anchors harmony.", summary)
        self.assertNotIn("More detail", summary)


class BuildAndRetrieveTest(unittest.TestCase):
    """Tree construction and collapsed-tree retrieval over a small vault."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self._tmp.name)
        _write_vault(self.vault, {
            "drums.md": "---\nauthority: 0.9\n---\n# Drums\nSyncopated drums and percussion grooves.\n",
            "bass.md": "# Bass\nWalking bass lines anchor the harmony.\n",
            "mixing.md": "# Mixing\nGain staging and headroom for every stem.\n## EQ\nCarve space with subtractive EQ.\n",
            "harmony.md": "# Harmony\nVoice leading prefers stepwise motion.\n",
            "form.md": "# Form\nVerse and chorus alternate in popular song form.\n",
        })

    def tearDown(self):
        self._tmp.cleanup()

    def test_build_creates_summary_levels(self):
        tree = RaptorTree(self.vault, group_size=2).build()
        levels = {node.level for node in tree.nodes}
        self.assertIn(0, levels)
        self.assertGreater(max(levels), 0)
        parents = [node for node in tree.nodes if node.level > 0]
        for parent in parents:
            self.assertTrue(parent.children)

    def test_rebuild_resets_nodes(self):
        tree = RaptorTree(self.vault, group_size=2).build()
        first_count = len(tree.nodes)
        tree.build()
        self.assertEqual(len(tree.nodes), first_count)

    def test_retrieve_finds_relevant_chunk(self):
        tree = RaptorTree(self.vault, group_size=2).build()
        results = tree.retrieve("syncopated drums groove", n=3)
        self.assertEqual(len(results), 3)
        top_node, top_score = results[0]
        self.assertIn("drums", top_node.text.lower())
        self.assertGreater(top_score, results[-1][1] - 1e-9)


class BestOfNTest(unittest.TestCase):
    """Best-of-N scoring combines relevance, recency, and authority."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_empty_vault_has_no_winner(self):
        result = RaptorTree(self.vault).build().best_of_n("anything")
        self.assertIsInstance(result, BestOfNResult)
        self.assertIsNone(result.winner)
        self.assertEqual(result.trace, [])

    def test_authority_breaks_relevance_ties(self):
        _write_vault(self.vault, {
            "low.md": "---\nauthority: 0.1\n---\nstem regeneration workflow\n",
            "high.md": "---\nauthority: 1.0\n---\nstem regeneration workflow\n",
        })
        now = 1_000_000.0
        for name in ("low.md", "high.md"):
            os.utime(self.vault / name, (now, now))
        result = RaptorTree(self.vault).build().best_of_n("stem regeneration workflow", n=2)
        self.assertIsNotNone(result.winner)
        self.assertTrue(result.winner.source.endswith("high.md"))
        self.assertEqual(len(result.trace), 2)
        self.assertGreaterEqual(result.trace[0].total, result.trace[1].total)

    def test_custom_scorer_overrides_default(self):
        _write_vault(self.vault, {
            "a.md": "---\nauthority: 1.0\n---\ndrums groove\n",
            "b.md": "---\nauthority: 0.1\n---\ndrums groove\n",
        })
        tree = RaptorTree(self.vault).build()
        result = tree.best_of_n("drums groove", n=2, scorer=lambda node, rel: -node.authority)
        self.assertTrue(result.winner.source.endswith("b.md"))


class AuthorityNormalizationTest(unittest.TestCase):
    """Vault frontmatter on a 1-10 scale is normalized into [0, 1]."""

    def test_ten_scale_normalized(self):
        chunks = chunk_markdown("---\nauthority: 8\n---\nbody text\n")
        self.assertAlmostEqual(chunks[0].authority, 0.8)

    def test_unit_scale_preserved(self):
        chunks = chunk_markdown("---\nauthority: 0.4\n---\nbody text\n")
        self.assertAlmostEqual(chunks[0].authority, 0.4)

    def test_bad_value_falls_back(self):
        chunks = chunk_markdown("---\nauthority: high\n---\nbody text\n")
        self.assertAlmostEqual(chunks[0].authority, 0.5)


if __name__ == "__main__":
    unittest.main()
