"""Tests for the embedded SQLite vector store and RAPTOR persistence."""

import tempfile
import unittest
from pathlib import Path

from music_data_science.memory.raptor import RaptorTree, default_embed
from music_data_science.memory.vector_store import SQLiteVectorStore, persist_raptor, restore_raptor


class StoreBasicsTest(unittest.TestCase):
    """Insert, replace, count, clear, and search behavior."""

    def test_add_and_search_ranks_by_similarity(self):
        store = SQLiteVectorStore()
        store.add("notes", "drums", default_embed("syncopated drums groove"), text="drums note")
        store.add("notes", "bass", default_embed("walking bass line"), text="bass note")
        hits = store.search("notes", default_embed("drums groove"), n=2)
        self.assertEqual(hits[0].ref, "drums")
        self.assertGreater(hits[0].score, hits[1].score)

    def test_replace_same_ref(self):
        store = SQLiteVectorStore()
        store.add("notes", "a", [1.0, 0.0], text="old")
        store.add("notes", "a", [0.0, 1.0], text="new")
        self.assertEqual(store.count("notes"), 1)
        self.assertEqual(store.search("notes", [0.0, 1.0], n=1)[0].text, "new")

    def test_namespaces_are_isolated(self):
        store = SQLiteVectorStore()
        store.add("a", "x", [1.0, 0.0])
        store.add("b", "y", [1.0, 0.0])
        self.assertEqual([hit.ref for hit in store.search("a", [1.0, 0.0])], ["x"])
        store.clear("a")
        self.assertEqual(store.count("a"), 0)
        self.assertEqual(store.count("b"), 1)

    def test_dimension_mismatch_yields_no_hits(self):
        store = SQLiteVectorStore()
        store.add("notes", "a", [1.0, 0.0, 0.0])
        self.assertEqual(store.search("notes", [1.0, 0.0]), [])

    def test_metadata_round_trip(self):
        store = SQLiteVectorStore()
        store.add("notes", "a", [1.0], metadata={"authority": 0.9, "tags": ["sme"]})
        hit = store.search("notes", [1.0], n=1)[0]
        self.assertEqual(hit.metadata["tags"], ["sme"])


class RaptorPersistenceTest(unittest.TestCase):
    """A built tree survives a save/restore round trip on disk."""

    def test_round_trip_preserves_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "vault"
            vault.mkdir()
            (vault / "drums.md").write_text(
                "---\nauthority: 0.9\n---\n# Drums\nSyncopated drums and percussion.\n", encoding="utf-8"
            )
            (vault / "bass.md").write_text("# Bass\nWalking bass anchors harmony.\n", encoding="utf-8")
            tree = RaptorTree(vault).build()
            expected = tree.retrieve("syncopated drums", n=1)[0][0]

            store = SQLiteVectorStore(Path(directory) / "vectors.db")
            count = persist_raptor(tree, store)
            self.assertEqual(count, len(tree.nodes))

            fresh = restore_raptor(store, RaptorTree(vault))
            self.assertEqual(len(fresh.nodes), count)
            restored = fresh.retrieve("syncopated drums", n=1)[0][0]
            self.assertEqual(restored.node_id, expected.node_id)
            self.assertEqual(restored.heading_path, expected.heading_path)
            self.assertAlmostEqual(restored.authority, expected.authority)

    def test_persist_overwrites_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory) / "vault"
            vault.mkdir()
            (vault / "one.md").write_text("# One\nfirst note body\n", encoding="utf-8")
            store = SQLiteVectorStore(Path(directory) / "vectors.db")
            tree = RaptorTree(vault).build()
            persist_raptor(tree, store)
            persist_raptor(tree, store)
            self.assertEqual(store.count("raptor"), len(tree.nodes))


if __name__ == "__main__":
    unittest.main()
