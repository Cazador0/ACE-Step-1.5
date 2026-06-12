"""Tests for the embedding backends (no model downloads, no heavy imports)."""

import unittest
from dataclasses import dataclass, field
from typing import Any

from music_data_science.evaluation.scorer import BestOfNScorer
from music_data_science.integrations.embeddings import create_embedder, similarity_from_embedder
from music_data_science.memory.raptor import default_embed
from music_data_science.stems.models import SongBlueprint


@dataclass
class FakeResult:
    """Stand-in for a generation result (files + metas)."""

    files: list[str] = field(default_factory=list)
    metas: dict[str, Any] = field(default_factory=dict)


def _sentence_transformers_installed() -> bool:
    try:
        import sentence_transformers  # noqa: F401, PLC0415 -- presence check only
    except ImportError:
        return False
    return True


class CreateEmbedderTest(unittest.TestCase):
    """Backend selection, fallbacks, and missing-dependency errors."""

    def test_hashed_backend_returns_default_embed(self):
        self.assertIs(create_embedder("hashed"), default_embed)

    def test_auto_falls_back_to_hashed_without_sentence_transformers(self):
        if _sentence_transformers_installed():
            self.skipTest("sentence-transformers installed; fallback path not exercised")
        self.assertIs(create_embedder("auto"), default_embed)

    def test_sentence_transformers_backend_error_mentions_extra(self):
        if _sentence_transformers_installed():
            self.skipTest("sentence-transformers installed; error path not exercised")
        with self.assertRaises(ImportError) as caught:
            create_embedder("sentence-transformers")
        self.assertIn("music-data-science[embeddings]", str(caught.exception))

    def test_clap_backend_error_mentions_extra(self):
        try:
            import transformers  # noqa: F401, PLC0415 -- presence check only
        except ImportError:
            with self.assertRaises(ImportError) as caught:
                create_embedder("clap")
            self.assertIn("music-data-science[embeddings]", str(caught.exception))
        else:
            self.skipTest("transformers installed; error path not exercised")

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_embedder("word2vec")


class SimilarityHookTest(unittest.TestCase):
    """The cosine hook ranks similar texts higher and plugs into the scorer."""

    def test_similar_texts_score_higher_than_dissimilar(self):
        similarity = similarity_from_embedder(default_embed)
        similar = similarity("dusty lofi hip hop drums", "lofi hip hop with dusty drums")
        dissimilar = similarity("dusty lofi hip hop drums", "symphonic power metal opera")
        self.assertGreater(similar, dissimilar)
        self.assertGreaterEqual(dissimilar, 0.0)
        self.assertLessEqual(similar, 1.0)

    def test_usable_as_scorer_embedding_similarity(self):
        scorer = BestOfNScorer(
            SongBlueprint(global_caption="lofi hip hop"),
            embedding_similarity=similarity_from_embedder(default_embed),
        )
        result = FakeResult(
            files=["/out/song.mp3"],
            metas={"bpm": 96, "keyscale": "C minor", "duration": 30.0, "genres": "lofi hip hop"},
        )
        breakdown = scorer.score(result)
        self.assertIn("embedding", breakdown.components)
        self.assertGreater(breakdown.components["embedding"], 0.9)


if __name__ == "__main__":
    unittest.main()
