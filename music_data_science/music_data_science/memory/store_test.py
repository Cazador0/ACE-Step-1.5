"""Tests for the SQLite system-of-record."""

import tempfile
import unittest
from pathlib import Path

from music_data_science.memory.store import MemoryStore


class SessionTest(unittest.TestCase):
    """Session creation and lookup."""

    def test_create_and_get_session(self):
        store = MemoryStore(":memory:")
        session_id = store.create_session("demo", {"bpm": 84})
        session = store.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session["name"], "demo")
        self.assertIn("84", session["blueprint_json"])

    def test_missing_session_is_none(self):
        store = MemoryStore(":memory:")
        self.assertIsNone(store.get_session("nope"))


class StemRowsTest(unittest.TestCase):
    """Stem rows are scoped per session and ordered."""

    def test_add_and_list_stems(self):
        store = MemoryStore(":memory:")
        session_id = store.create_session()
        other_id = store.create_session()
        store.add_stem(session_id, "drums", caption="dusty kit", audio_path="/out/d.wav")
        store.add_stem(session_id, "bass", audio_path="/out/b.wav")
        store.add_stem(other_id, "vocals")
        stems = store.list_stems(session_id)
        self.assertEqual([stem["stem_class"] for stem in stems], ["drums", "bass"])
        self.assertEqual(stems[0]["audio_path"], "/out/d.wav")


class GenerationAndScoreTest(unittest.TestCase):
    """Generation/score persistence with decoded JSON columns."""

    def test_record_and_list_generations(self):
        store = MemoryStore(":memory:")
        session_id = store.create_session()
        store.record_generation(
            session_id, "t-1", "repaint",
            request={"task_type": "repaint", "seed": 7},
            result={"files": ["/out/a.wav"]},
            seed=7,
        )
        generations = store.list_generations(session_id)
        self.assertEqual(len(generations), 1)
        self.assertEqual(generations[0]["request"]["seed"], 7)
        self.assertEqual(generations[0]["result"]["files"], ["/out/a.wav"])

    def test_best_generation_ordering(self):
        store = MemoryStore(":memory:")
        session_id = store.create_session()
        low = store.record_generation(session_id, "t-1", "repaint", {}, {}, seed=1)
        high = store.record_generation(session_id, "t-2", "repaint", {}, {}, seed=2)
        store.record_score(low, total=0.4, accepted=False, breakdown={"bpm": 0.4})
        store.record_score(high, total=0.9, accepted=True, breakdown={"bpm": 0.9})
        best = store.best_generation(session_id)
        self.assertIsNotNone(best)
        self.assertEqual(best["task_id"], "t-2")
        self.assertEqual(best["score_total"], 0.9)

    def test_best_generation_none_without_scores(self):
        store = MemoryStore(":memory:")
        session_id = store.create_session()
        self.assertIsNone(store.best_generation(session_id))


class DurabilityTest(unittest.TestCase):
    """Data persists across connections when backed by a file."""

    def test_file_backed_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "mds.sqlite3")
            first = MemoryStore(db_path)
            session_id = first.create_session("durable")
            first.add_stem(session_id, "drums")
            first.close()

            second = MemoryStore(db_path)
            self.assertEqual(second.get_session(session_id)["name"], "durable")
            self.assertEqual(len(second.list_stems(session_id)), 1)
            second.close()


if __name__ == "__main__":
    unittest.main()
