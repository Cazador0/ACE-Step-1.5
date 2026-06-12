"""Tests for StemSession workflows and local mixdown (mocked HTTP, tmp dirs)."""

import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import httpx

from music_data_science.data.cache import InMemoryCache, stem_path_key, task_result_key
from music_data_science.evaluation.scorer import BestOfNScorer
from music_data_science.memory.store import MemoryStore
from music_data_science.pipeline.client import AceStepClient
from music_data_science.pipeline.mixdown import mix_wavs
from music_data_science.pipeline.orchestrator import StemSession
from music_data_science.stems.models import SongBlueprint, StemClass, StemEdit, StemSpec


class FakeAceStep:
    """Mock transport: assigns task IDs and succeeds every task immediately."""

    def __init__(self, metas=None):
        self.metas = metas or {"bpm": 84, "duration": 120.0, "keyscale": "F major"}
        self.submitted = []
        self._counter = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/release_task":
            self._counter += 1
            task_id = f"task-{self._counter}"
            self.submitted.append((task_id, body))
            data = {"task_id": task_id, "status": "queued", "queue_position": 1}
        else:
            task_ids = json.loads(body["task_id_list"])
            data = [{
                "task_id": task_id,
                "status": 1,
                "result": json.dumps([{"file": f"/out/{task_id}.wav", "status": 1, "metas": self.metas}]),
                "progress_text": "",
            } for task_id in task_ids]
        return httpx.Response(200, json={"data": data, "code": 200, "error": None, "timestamp": 0, "extra": None})


def make_session(server=None, cache=None):
    """Build a StemSession over a mocked client and in-memory store."""
    server = server or FakeAceStep()
    client = AceStepClient(transport=httpx.MockTransport(server), sleep=lambda _s: None)
    blueprint = SongBlueprint(
        global_caption="warm lo-fi hip hop", bpm=84, key_scale="F major", duration=120.0,
        stems=[StemSpec(stem_class=StemClass.DRUMS, caption="dusty kit")],
    )
    return StemSession(client, blueprint, store=MemoryStore(":memory:"), name="test", cache=cache), server


class WorkflowTest(unittest.TestCase):
    """generate / separate / regenerate / add_layer record everything."""

    def test_generate_records_and_sets_song_path(self):
        session, server = make_session()
        outcome = session.generate()
        self.assertEqual(session.song_path, f"/out/{outcome.task_id}.wav")
        self.assertEqual(server.submitted[0][1]["task_type"], "text2music")
        generations = session.store.list_generations(session.session_id)
        self.assertEqual(len(generations), 1)
        self.assertEqual(generations[0]["task_type"], "text2music")
        self.assertEqual(len(session.frames.generations()), 1)

    def test_separate_requires_source(self):
        session, _server = make_session()
        with self.assertRaises(ValueError):
            session.separate()

    def test_separate_extracts_each_stem(self):
        session, server = make_session()
        outcomes = session.separate(audio_path="/in/song.wav", stems=[StemClass.VOCALS, StemClass.DRUMS])
        self.assertEqual(set(outcomes), {StemClass.VOCALS, StemClass.DRUMS})
        for _task_id, payload in server.submitted:
            self.assertEqual(payload["task_type"], "extract")
            self.assertEqual(payload["src_audio_path"], "/in/song.wav")
        stems = session.store.list_stems(session.session_id)
        self.assertEqual({stem["stem_class"] for stem in stems}, {"vocals", "drums"})
        self.assertIn(StemClass.DRUMS, session.stem_paths)

    def test_regenerate_uses_stem_path_as_source(self):
        session, server = make_session()
        session.stem_paths[StemClass.DRUMS] = "/stems/drums.wav"
        edit = StemEdit(target=StemClass.DRUMS, intent="punchier", window_start=45.0, window_end=70.0)
        session.regenerate(edit)
        payload = server.submitted[0][1]
        self.assertEqual(payload["task_type"], "repaint")
        self.assertEqual(payload["src_audio_path"], "/stems/drums.wav")
        self.assertEqual(payload["repainting_start"], 45.0)

    def test_add_layer_sends_lego_with_track_fields(self):
        session, server = make_session()
        session.song_path = "/out/song.wav"
        spec = StemSpec(stem_class=StemClass.STRINGS, caption="warm pads", seed=11)
        session.add_layer(spec)
        payload = server.submitted[0][1]
        self.assertEqual(payload["task_type"], "lego")
        self.assertEqual(payload["track_name"], "strings")
        self.assertEqual(payload["track_classes"], ["strings"])
        self.assertEqual(payload["seed"], 11)
        self.assertFalse(payload["use_random_seed"])
        self.assertIn(StemClass.STRINGS, session.stem_paths)


class CacheWiringTest(unittest.TestCase):
    """An optional hot cache mirrors task results and stem paths."""

    def test_generate_caches_task_result(self):
        cache = InMemoryCache()
        session, _server = make_session(cache=cache)
        outcome = session.generate()
        cached = cache.get(task_result_key(outcome.task_id))
        self.assertIsNotNone(cached)
        payload = json.loads(cached)
        self.assertEqual(payload["status"], 1)
        self.assertEqual(payload["files"], outcome.result.files)
        self.assertEqual(payload["metas"]["bpm"], 84)

    def test_separate_caches_stem_paths(self):
        cache = InMemoryCache()
        session, _server = make_session(cache=cache)
        outcomes = session.separate(audio_path="/in/song.wav", stems=[StemClass.VOCALS, StemClass.DRUMS])
        for stem, outcome in outcomes.items():
            self.assertEqual(
                cache.get(stem_path_key(session.session_id, stem.value)),
                outcome.result.files[0],
            )

    def test_no_cache_by_default(self):
        session, _server = make_session()
        self.assertIsNone(session.cache)
        session.generate()  # must not raise without a cache


class BestOfNTest(unittest.TestCase):
    """Best-of-N submits n seeds, scores, ranks, and records all candidates."""

    def test_best_of_n_ranks_and_records(self):
        session, server = make_session()
        session.song_path = "/out/song.wav"
        scorer = BestOfNScorer(expected=session.blueprint, threshold=0.5)
        edit = StemEdit(target=StemClass.DRUMS, intent="punchier drums")
        candidates = session.best_of_n(edit, n=3, scorer=scorer, base_seed=100)

        self.assertEqual(len(candidates), 3)
        seeds = sorted(payload["seed"] for _tid, payload in server.submitted)
        self.assertEqual(seeds, [100, 101, 102])
        for _tid, payload in server.submitted:
            self.assertFalse(payload["use_random_seed"])
        totals = [candidate.breakdown.total for candidate in candidates]
        self.assertEqual(totals, sorted(totals, reverse=True))
        # All three generations and scores persisted.
        self.assertEqual(len(session.store.list_generations(session.session_id)), 3)
        self.assertEqual(len(session.frames.scores()), 3)
        best = session.store.best_generation(session.session_id)
        self.assertIsNotNone(best)

    def test_best_of_n_rejects_invalid_n(self):
        session, _server = make_session()
        scorer = BestOfNScorer(expected=session.blueprint)
        with self.assertRaises(ValueError):
            session.best_of_n(StemEdit(target=StemClass.DRUMS), n=0, scorer=scorer)


def write_sine_wav(path: Path, frequency: float, amplitude: float = 0.3, frames: int = 4410):
    """Write a mono 16-bit PCM sine wave for mixdown tests."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        samples = [
            int(amplitude * 32767 * math.sin(2 * math.pi * frequency * index / 44100))
            for index in range(frames)
        ]
        handle.writeframes(struct.pack(f"<{frames}h", *samples))


class MixdownTest(unittest.TestCase):
    """Local recombination of stem WAVs."""

    def test_mix_two_stems_and_recombine(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            drums, bass = tmp_path / "drums.wav", tmp_path / "bass.wav"
            write_sine_wav(drums, 220.0, amplitude=0.6)
            write_sine_wav(bass, 110.0, amplitude=0.6)
            output = mix_wavs([drums, bass], tmp_path / "mix.wav")
            with wave.open(str(output), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getframerate(), 44100)
                self.assertEqual(handle.getnframes(), 4410)

            session, _server = make_session()
            session.stem_paths = {StemClass.DRUMS: str(drums), StemClass.BASS: str(bass)}
            mixed = session.recombine(tmp_path / "recombined.wav")
            self.assertTrue(mixed.exists())

    def test_mix_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            mix_wavs([], "/tmp/never.wav")

    def test_recombine_without_stems_raises(self):
        session, _server = make_session()
        with self.assertRaises(ValueError):
            session.recombine("/tmp/never.wav")


if __name__ == "__main__":
    unittest.main()
