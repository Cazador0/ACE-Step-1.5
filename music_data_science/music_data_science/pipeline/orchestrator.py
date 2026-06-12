"""Stem workflows: generate, separate, regenerate, add-layer, Best-of-N, recombine.

:class:`StemSession` is the workflow object holding one song and its stems.
All network traffic goes through :class:`~music_data_science.pipeline.client.AceStepClient`;
every request/result pair is recorded to the SQLite system-of-record and the
pandas telemetry frames so sessions are replayable and auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from music_data_science.data.cache import Cache, stem_path_key, task_result_key
from music_data_science.data.frames import TelemetryFrames
from music_data_science.evaluation.scorer import BestOfNScorer, ScoreBreakdown
from music_data_science.memory.store import MemoryStore
from music_data_science.pipeline.client import AceStepClient, TaskResult
from music_data_science.pipeline.mixdown import mix_wavs
from music_data_science.stems.models import SongBlueprint, StemClass, StemEdit, StemSpec
from music_data_science.stems.translator import blueprint_to_request, to_request

#: Stems extracted by default when none are specified for :meth:`StemSession.separate`.
DEFAULT_SEPARATION = (StemClass.VOCALS, StemClass.DRUMS, StemClass.BASS, StemClass.GUITAR)

#: Hot-cache TTL for task results; stale results are re-readable from the SQLite store.
TASK_RESULT_TTL_SECONDS = 3600.0


@dataclass
class GenerationOutcome:
    """One executed generation: submitted request, task ID, and terminal result."""

    task_id: str
    request: dict
    result: TaskResult
    seed: Optional[int] = None


@dataclass
class Candidate:
    """A scored Best-of-N candidate."""

    outcome: GenerationOutcome
    breakdown: ScoreBreakdown


class StemSession:
    """Workflow object holding a song, its stems, and their provenance.

    Args:
        client: Configured ACE-Step API client.
        blueprint: Structured song representation driving all requests.
        store: SQLite system-of-record; an in-memory store is created if omitted.
        frames: Telemetry accumulator; created if omitted.
        name: Human-readable session name.
        cache: Optional hot cache mirroring task results and stem paths under
            the documented keyspace; ``None`` disables caching entirely.
    """

    def __init__(
        self,
        client: AceStepClient,
        blueprint: SongBlueprint,
        store: Optional[MemoryStore] = None,
        frames: Optional[TelemetryFrames] = None,
        name: str = "",
        cache: Optional[Cache] = None,
    ) -> None:
        self.client = client
        self.blueprint = blueprint
        self.store = store if store is not None else MemoryStore(":memory:")
        self.frames = frames if frames is not None else TelemetryFrames()
        self.cache = cache
        self.session_id = self.store.create_session(name, blueprint.model_dump())
        self.song_path: Optional[str] = None
        self.stem_paths: dict[StemClass, str] = {}

    def _execute(self, payload: dict, stem_class: str = "") -> GenerationOutcome:
        """Submit one payload, poll to completion, and record the round trip."""
        task_id = self.client.submit(payload)
        result = self.client.poll(task_id)
        seed = payload.get("seed")
        seed_value = int(seed) if isinstance(seed, int) and seed >= 0 else None
        outcome = GenerationOutcome(task_id=task_id, request=payload, result=result, seed=seed_value)
        self._record(outcome, stem_class)
        return outcome

    def _record(self, outcome: GenerationOutcome, stem_class: str) -> int:
        """Persist one outcome to the store and telemetry frames; return the row ID."""
        result = outcome.result
        if self.cache is not None:
            # Audio codes are intentionally not cached: the /query_result metas
            # contract only carries bpm/duration/genres/keyscale/timesignature.
            self.cache.set(
                task_result_key(outcome.task_id),
                json.dumps({"status": int(result.status), "files": result.files, "metas": result.metas}),
                ttl_seconds=TASK_RESULT_TTL_SECONDS,
            )
        generation_id = self.store.record_generation(
            session_id=self.session_id,
            task_id=outcome.task_id,
            task_type=str(outcome.request.get("task_type", "")),
            request=outcome.request,
            result={"status": int(result.status), "files": result.files, "metas": result.metas},
            seed=outcome.seed,
        )
        self.frames.add_generation(
            session_id=self.session_id,
            task_id=outcome.task_id,
            task_type=str(outcome.request.get("task_type", "")),
            stem_class=stem_class,
            seed=outcome.seed,
            status=int(result.status),
            bpm=result.metas.get("bpm"),
            duration=result.metas.get("duration"),
            audio_path=result.files[0] if result.files else "",
        )
        return generation_id

    def generate(self, blueprint: Optional[SongBlueprint] = None) -> GenerationOutcome:
        """Generate the full song (``text2music``) from the blueprint."""
        if blueprint is not None:
            self.blueprint = blueprint
        outcome = self._execute(blueprint_to_request(self.blueprint))
        if outcome.result.files:
            self.song_path = outcome.result.files[0]
        return outcome

    def separate(
        self,
        audio_path: Optional[str] = None,
        stems: Optional[list[StemClass]] = None,
    ) -> dict[StemClass, GenerationOutcome]:
        """Extract individual stems from the song (one ``extract`` task per stem)."""
        source = audio_path or self.song_path
        if not source:
            raise ValueError("no source audio: pass audio_path or call generate() first")
        outcomes: dict[StemClass, GenerationOutcome] = {}
        for stem in stems if stems is not None else list(DEFAULT_SEPARATION):
            edit = StemEdit(target=stem, operation="separate")  # type: ignore[arg-type]
            outcome = self._execute(to_request(edit, self.blueprint, source), stem.value)
            outcomes[stem] = outcome
            self._register_stem(stem, outcome, caption=f"extracted {stem.value}")
        return outcomes

    def regenerate(self, edit: StemEdit, src_audio_path: Optional[str] = None) -> GenerationOutcome:
        """Re-generate one stem via repaint/cover according to ``edit``."""
        source = src_audio_path or self.stem_paths.get(edit.target) or self.song_path
        outcome = self._execute(to_request(edit, self.blueprint, source), edit.target.value)
        self._register_stem(edit.target, outcome, caption=edit.intent)
        return outcome

    def add_layer(self, spec: StemSpec, src_audio_path: Optional[str] = None) -> GenerationOutcome:
        """Add a new stem on top of the song context (``lego`` task)."""
        source = src_audio_path or self.song_path
        payload = blueprint_to_request(
            self.blueprint,
            task_type="lego",
            prompt=spec.caption or spec.role,
            track_name=spec.stem_class.value,
            track_classes=[spec.stem_class.value],
        )
        if source is not None:
            payload["src_audio_path"] = source
        if spec.seed is not None:
            payload.update(seed=spec.seed, use_random_seed=False)
        outcome = self._execute(payload, spec.stem_class.value)
        self._register_stem(spec.stem_class, outcome, caption=spec.caption)
        return outcome

    def best_of_n(
        self,
        edit: StemEdit,
        n: int,
        scorer: BestOfNScorer,
        src_audio_path: Optional[str] = None,
        base_seed: int = 1000,
    ) -> list[Candidate]:
        """Run ``edit`` with ``n`` pinned seeds, score every result, rank descending.

        Every candidate (not just the winner) is recorded to the store and
        telemetry frames together with its score breakdown.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        source = src_audio_path or self.stem_paths.get(edit.target) or self.song_path
        first_seed = edit.seed if edit.seed is not None else base_seed
        candidates: list[Candidate] = []
        for index in range(n):
            seeded = edit.model_copy(update={"seed": first_seed + index})
            payload = to_request(seeded, self.blueprint, source)
            task_id = self.client.submit(payload)
            result = self.client.poll(task_id)
            outcome = GenerationOutcome(task_id=task_id, request=payload, result=result, seed=payload["seed"])
            generation_id = self._record(outcome, edit.target.value)
            breakdown = scorer.score(result)
            self.store.record_score(generation_id, breakdown.total, breakdown.passed, breakdown.components)
            self.frames.add_score(
                self.session_id, task_id, breakdown.total, breakdown.passed, breakdown.components
            )
            candidates.append(Candidate(outcome=outcome, breakdown=breakdown))
        candidates.sort(key=lambda candidate: candidate.breakdown.total, reverse=True)
        if candidates and candidates[0].outcome.result.files:
            self._register_stem(edit.target, candidates[0].outcome, caption=edit.intent)
        return candidates

    def recombine(self, output_path: str | Path, stems: Optional[list[StemClass]] = None) -> Path:
        """Mix the session's rendered stem WAVs into one output file."""
        selected = stems if stems is not None else list(self.stem_paths)
        paths = [self.stem_paths[stem] for stem in selected if stem in self.stem_paths]
        if not paths:
            raise ValueError("no rendered stems available to recombine")
        return mix_wavs(list(paths), output_path)

    def _register_stem(self, stem: StemClass, outcome: GenerationOutcome, caption: str = "") -> None:
        """Track the latest rendered audio path for ``stem`` in memory and store."""
        if not outcome.result.files:
            return
        path = outcome.result.files[0]
        self.stem_paths[stem] = path
        self.store.add_stem(self.session_id, stem.value, caption=caption, audio_path=path)
        if self.cache is not None:
            self.cache.set(stem_path_key(self.session_id, stem.value), path)
