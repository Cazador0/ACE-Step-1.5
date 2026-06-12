"""Deterministic prompt -> ``StemEdit`` / ``EditPlan`` translation.

A keyword + regex intent grammar maps user edit prompts (for example
"make the drums punchier from 0:45 to 1:10") onto typed :class:`StemEdit`
objects with explicit ACE-Step parameters.  An optional ``llm_refine`` hook
can post-process plans, but everything works without any LLM.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from music_data_science.stems.models import (
    EditPlan,
    SongBlueprint,
    StemClass,
    StemEdit,
    StemOperation,
)
from music_data_science.stems.vocabulary import (
    AGGRESSIVE_CUES,
    CONSERVATIVE_CUES,
    OPERATION_CUES,
    STEM_SYNONYMS,
)

_CLOCK_RANGE = re.compile(
    r"(\d+):(\d{2})\s*(?:to|until|through|-|–|—)\s*(\d+):(\d{2})", re.IGNORECASE
)
_SECONDS_RANGE = re.compile(
    r"(\d+(?:\.\d+)?)\s*s(?:ec(?:onds)?)?\s*(?:to|until|-|–|—)\s*(\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)
_STRENGTH_BY_MODE = {"conservative": 0.2, "balanced": 0.5, "aggressive": 0.9}


def _find_stem(prompt: str) -> Optional[StemClass]:
    """Return the earliest (longest-phrase-first) stem mentioned in ``prompt``."""
    lowered = prompt.lower()
    best: Optional[tuple[int, int, StemClass]] = None
    for phrase, stem in STEM_SYNONYMS:
        match = re.search(rf"\b{re.escape(phrase)}\b", lowered)
        if match and (best is None or (match.start(), -len(phrase)) < (best[0], best[1])):
            best = (match.start(), -len(phrase), stem)
    return best[2] if best else None


def _find_window(prompt: str) -> tuple[float, Optional[float]]:
    """Extract an edit time window in seconds from ``prompt``; defaults to whole song."""
    clock = _CLOCK_RANGE.search(prompt)
    if clock:
        m1, s1, m2, s2 = (int(group) for group in clock.groups())
        return float(m1 * 60 + s1), float(m2 * 60 + s2)
    seconds = _SECONDS_RANGE.search(prompt)
    if seconds:
        return float(seconds.group(1)), float(seconds.group(2))
    return 0.0, None


def _find_operation(prompt: str) -> StemOperation:
    """Map intent keywords to an operation; defaults to REGENERATE (repaint)."""
    lowered = prompt.lower()
    for operation, cues in OPERATION_CUES:
        if any(re.search(rf"\b{cue}\b", lowered) for cue in cues):
            return operation
    return StemOperation.REGENERATE


def _find_repaint_mode(prompt: str) -> str:
    """Heuristic: 'subtle/tweak' -> conservative, 'completely different' -> aggressive."""
    lowered = prompt.lower()
    if any(cue in lowered for cue in AGGRESSIVE_CUES):
        return "aggressive"
    if any(cue in lowered for cue in CONSERVATIVE_CUES):
        return "conservative"
    return "balanced"


class PromptTranslator:
    """Translate natural-language edit prompts into typed ACE-Step edit plans.

    Args:
        llm_refine: Optional hook called with the deterministic plan for
            LLM-based refinement.  When ``None`` the deterministic plan is final.
    """

    def __init__(self, llm_refine: Optional[Callable[[EditPlan], EditPlan]] = None) -> None:
        self.llm_refine = llm_refine

    def translate_edit(self, prompt: str, default_target: StemClass = StemClass.VOCALS) -> StemEdit:
        """Translate one prompt into a single :class:`StemEdit`."""
        target = _find_stem(prompt) or default_target
        start, end = _find_window(prompt)
        mode = _find_repaint_mode(prompt)
        return StemEdit(
            target=target,
            intent=prompt.strip(),
            window_start=start,
            window_end=end,
            operation=_find_operation(prompt),
            repaint_mode=mode,  # type: ignore[arg-type]
            repaint_strength=_STRENGTH_BY_MODE[mode],
        )

    def translate(self, prompts: list[str] | str) -> EditPlan:
        """Translate one or more prompts into an ordered :class:`EditPlan`."""
        items = [prompts] if isinstance(prompts, str) else list(prompts)
        plan = EditPlan(
            edits=[self.translate_edit(prompt) for prompt in items],
            acceptance_criteria=[f"Output reflects intent: {prompt.strip()}" for prompt in items],
        )
        if self.llm_refine is not None:
            plan = self.llm_refine(plan)
        return plan


def blueprint_to_request(blueprint: SongBlueprint, **overrides: object) -> dict:
    """Emit a ``text2music`` ``GenerateMusicRequest``-compatible payload."""
    payload: dict = {
        "task_type": "text2music",
        "prompt": blueprint.global_caption,
        "global_caption": blueprint.global_caption,
        "lyrics": blueprint.lyrics,
        "bpm": blueprint.bpm,
        "key_scale": blueprint.key_scale,
        "time_signature": blueprint.time_signature,
        "vocal_language": blueprint.vocal_language,
        "audio_duration": blueprint.duration,
        "use_random_seed": True,
        "seed": -1,
    }
    payload.update(overrides)
    return payload


def to_request(
    edit: StemEdit,
    blueprint: SongBlueprint,
    src_audio_path: Optional[str] = None,
    **overrides: object,
) -> dict:
    """Emit a valid ``GenerateMusicRequest`` payload for one stem edit.

    Args:
        edit: The stem edit to execute.
        blueprint: Song-level context (metadata, lyrics, per-stem specs).
        src_audio_path: Server-side path of the source audio being edited.
        **overrides: Extra/override request fields merged in last.

    Returns:
        A dict matching ACE-Step's ``GenerateMusicRequest`` schema.
    """
    spec = blueprint.stem(edit.target)
    seed = edit.seed if edit.seed is not None else (spec.seed if spec else None)
    payload: dict = {
        "task_type": edit.operation.task_type,
        "prompt": edit.intent or (spec.caption if spec else ""),
        "global_caption": blueprint.global_caption,
        "lyrics": blueprint.lyrics,
        "bpm": (spec.bpm if spec and spec.bpm else blueprint.bpm),
        "key_scale": (spec.key_scale if spec and spec.key_scale else blueprint.key_scale),
        "time_signature": blueprint.time_signature,
        "vocal_language": blueprint.vocal_language,
        "audio_duration": blueprint.duration,
        "use_random_seed": seed is None,
        "seed": -1 if seed is None else seed,
    }
    if src_audio_path is not None:
        payload["src_audio_path"] = src_audio_path
    if edit.operation is StemOperation.REGENERATE:
        payload["repainting_start"] = edit.window_start
        payload["repainting_end"] = edit.window_end
        payload["repaint_mode"] = edit.repaint_mode
        payload["repaint_strength"] = edit.repaint_strength
    elif edit.operation is StemOperation.REPLACE_STYLE:
        payload["audio_cover_strength"] = edit.cover_strength
    elif edit.operation in (StemOperation.ADD_LAYER, StemOperation.SEPARATE):
        payload["track_name"] = edit.target.value
        payload["track_classes"] = [edit.target.value]
    elif edit.operation is StemOperation.EXTEND:
        payload["track_classes"] = [edit.target.value]
    payload.update(overrides)
    return payload
