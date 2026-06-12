"""Tests for Best-of-N scoring."""

import unittest
from dataclasses import dataclass, field
from typing import Any

from music_data_science.evaluation.scorer import (
    BestOfNScorer,
    score_bpm,
    score_duration,
    score_key,
)
from music_data_science.stems.models import SongBlueprint


@dataclass
class FakeResult:
    """Stand-in for a generation result (files + metas)."""

    files: list[str] = field(default_factory=list)
    metas: dict[str, Any] = field(default_factory=dict)


def _good_result() -> FakeResult:
    return FakeResult(
        files=["/out/song.mp3"],
        metas={"bpm": 120, "keyscale": "C major", "duration": 30.0, "genres": "lofi hip hop"},
    )


class ComponentScoreTest(unittest.TestCase):
    """The individual metadata adherence scores."""

    def test_bpm_exact_and_falloff(self):
        self.assertEqual(score_bpm(120, 120), 1.0)
        self.assertAlmostEqual(score_bpm(120, 124, tolerance=8), 0.5)
        self.assertEqual(score_bpm(120, 200), 0.0)

    def test_bpm_unscorable_and_garbage(self):
        self.assertIsNone(score_bpm(None, 120))
        self.assertIsNone(score_bpm(120, None))
        self.assertEqual(score_bpm(120, "fast"), 0.0)

    def test_key_normalized_match(self):
        self.assertEqual(score_key("C major", " c MAJOR "), 1.0)
        self.assertEqual(score_key("C major", "A minor"), 0.0)
        self.assertIsNone(score_key("", "C major"))

    def test_duration_falloff(self):
        self.assertEqual(score_duration(30.0, 30.0), 1.0)
        self.assertEqual(score_duration(30.0, 60.0), 0.0)
        self.assertIsNone(score_duration(None, 30.0))
        self.assertEqual(score_duration(30.0, "long"), 0.0)


class BestOfNScorerTest(unittest.TestCase):
    """Weighted totals, rubric, embedding hook, and acceptance threshold."""

    def test_perfect_candidate_passes(self):
        scorer = BestOfNScorer(SongBlueprint(bpm=120, key_scale="C major", duration=30.0))
        breakdown = scorer.score(_good_result())
        self.assertAlmostEqual(breakdown.components["bpm"], 1.0)
        self.assertAlmostEqual(breakdown.components["key"], 1.0)
        self.assertAlmostEqual(breakdown.components["rubric"], 1.0)
        self.assertAlmostEqual(breakdown.total, 1.0)
        self.assertTrue(breakdown.passed)

    def test_unscorable_components_are_skipped(self):
        scorer = BestOfNScorer(SongBlueprint())  # no expectations set
        breakdown = scorer.score(_good_result())
        self.assertNotIn("bpm", breakdown.components)
        self.assertNotIn("key", breakdown.components)
        self.assertIn("rubric", breakdown.components)

    def test_empty_result_fails_rubric_and_threshold(self):
        scorer = BestOfNScorer(SongBlueprint(bpm=120, key_scale="C major"))
        breakdown = scorer.score(FakeResult())
        self.assertAlmostEqual(breakdown.components["rubric"], 0.0)
        self.assertFalse(breakdown.passed)

    def test_embedding_hook_used_and_clamped(self):
        scorer = BestOfNScorer(
            SongBlueprint(global_caption="lofi hip hop"),
            embedding_similarity=lambda expected, candidate: 5.0,
        )
        breakdown = scorer.score(_good_result())
        self.assertEqual(breakdown.components["embedding"], 1.0)

    def test_custom_weights_and_threshold(self):
        scorer = BestOfNScorer(
            SongBlueprint(bpm=120),
            rubric=[],
            weights={"bpm": 1.0},
            threshold=0.4,
        )
        breakdown = scorer.score(FakeResult(files=["x"], metas={"bpm": 124}))
        self.assertAlmostEqual(breakdown.total, 0.5)
        self.assertTrue(breakdown.passed)

    def test_no_scorable_components_totals_zero(self):
        scorer = BestOfNScorer(SongBlueprint(), rubric=[])
        breakdown = scorer.score(FakeResult())
        self.assertEqual(breakdown.total, 0.0)
        self.assertFalse(breakdown.passed)


if __name__ == "__main__":
    unittest.main()
