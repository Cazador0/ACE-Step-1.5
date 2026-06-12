"""Best-of-N scoring: metadata adherence, rubric checks, optional embedding hook.

A scorer turns one generation result (``files`` + ``metas``) into a
:class:`ScoreBreakdown` against the expectations encoded in a
:class:`~music_data_science.stems.models.SongBlueprint`.  All components are
in [0, 1]; the total is a weighted average over the available components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from music_data_science.stems.models import SongBlueprint


class ScorableResult(Protocol):
    """Minimal shape a scorable generation result must expose."""

    files: Sequence[str]
    metas: Mapping[str, Any]


@dataclass
class ScoreBreakdown:
    """Score components, weighted total, and acceptance flag for one candidate."""

    components: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    passed: bool = False


def score_bpm(expected: Optional[int], actual: Any, tolerance: int = 8) -> Optional[float]:
    """Score tempo adherence with a linear falloff; ``None`` when unscorable."""
    if expected is None or actual in (None, ""):
        return None
    try:
        delta = abs(float(actual) - float(expected))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, 1.0 - delta / float(tolerance))


def score_key(expected: str, actual: Any) -> Optional[float]:
    """Score key/scale adherence by normalized exact match."""
    if not expected:
        return None
    normalized = str(actual or "").strip().lower()
    return 1.0 if normalized == expected.strip().lower() else 0.0


def score_duration(expected: Optional[float], actual: Any, rel_tolerance: float = 0.15) -> Optional[float]:
    """Score duration adherence with linear falloff inside a relative tolerance."""
    if not expected or actual in (None, ""):
        return None
    try:
        delta = abs(float(actual) - float(expected))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, 1.0 - delta / (float(expected) * rel_tolerance))


def default_rubric() -> list[tuple[str, Callable[[ScorableResult], bool]]]:
    """Professional-standards checklist applied to every candidate.

    Returns:
        Named boolean checks: output exists, metadata is reported, and the
        reported duration is positive and plausible.
    """
    return [
        ("has_audio_file", lambda result: len(result.files) > 0),
        ("has_metadata", lambda result: len(result.metas) > 0),
        ("reports_bpm", lambda result: result.metas.get("bpm") not in (None, "")),
        ("reports_key", lambda result: bool(result.metas.get("keyscale"))),
        ("positive_duration", lambda result: _as_float(result.metas.get("duration")) > 0.0),
    ]


def _as_float(value: Any) -> float:
    """Coerce a metadata value to float, treating bad values as 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class BestOfNScorer:
    """Score candidates against a blueprint for Best-of-N selection.

    Args:
        expected: Blueprint encoding the metadata the output should adhere to.
        embedding_similarity: Optional hook ``(expected_text, candidate_text) -> [0, 1]``
            for semantic caption/genre similarity (for example CLAP embeddings).
        rubric: Override for the professional-standards checklist.
        weights: Per-component weights; unscored components are skipped.
        threshold: Minimum total for ``passed`` acceptance.
    """

    DEFAULT_WEIGHTS = {"bpm": 0.3, "key": 0.2, "duration": 0.2, "rubric": 0.2, "embedding": 0.1}

    def __init__(
        self,
        expected: SongBlueprint,
        embedding_similarity: Optional[Callable[[str, str], float]] = None,
        rubric: Optional[list[tuple[str, Callable[[ScorableResult], bool]]]] = None,
        weights: Optional[dict[str, float]] = None,
        threshold: float = 0.7,
    ) -> None:
        self.expected = expected
        self.embedding_similarity = embedding_similarity
        self.rubric = default_rubric() if rubric is None else rubric
        self.weights = dict(self.DEFAULT_WEIGHTS if weights is None else weights)
        self.threshold = threshold

    def _component_scores(self, result: ScorableResult) -> dict[str, float]:
        """Compute every scorable component for one candidate."""
        metas = result.metas
        components: dict[str, Optional[float]] = {
            "bpm": score_bpm(self.expected.bpm, metas.get("bpm")),
            "key": score_key(self.expected.key_scale, metas.get("keyscale")),
            "duration": score_duration(self.expected.duration, metas.get("duration")),
        }
        if self.rubric:
            checks = [1.0 if check(result) else 0.0 for _, check in self.rubric]
            components["rubric"] = sum(checks) / len(checks)
        if self.embedding_similarity is not None:
            candidate_text = str(metas.get("genres") or "") or str(metas.get("caption") or "")
            similarity = self.embedding_similarity(self.expected.global_caption, candidate_text)
            components["embedding"] = min(1.0, max(0.0, similarity))
        return {name: value for name, value in components.items() if value is not None}

    def score(self, result: ScorableResult) -> ScoreBreakdown:
        """Score one candidate; total is the weighted mean of available components."""
        components = self._component_scores(result)
        weighted = [(self.weights.get(name, 0.0), value) for name, value in components.items()]
        weight_sum = sum(weight for weight, _ in weighted)
        total = sum(weight * value for weight, value in weighted) / weight_sum if weight_sum else 0.0
        return ScoreBreakdown(components=components, total=total, passed=total >= self.threshold)
