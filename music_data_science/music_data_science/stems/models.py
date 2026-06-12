"""Pydantic v2 domain model for stems, song blueprints, and stem edits.

``StemClass`` mirrors ACE-Step's ``TRACK_NAMES`` constant exactly so values can
be passed straight through as ``track_name`` / ``track_classes`` in
``GenerateMusicRequest`` payloads.  ``StemOperation`` maps 1:1 onto ACE-Step
task types (``TASK_TYPES_BASE``).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

RepaintMode = Literal["conservative", "balanced", "aggressive"]


class StemClass(str, Enum):
    """Stem/track classes, aligned with ACE-Step ``acestep.constants.TRACK_NAMES``."""

    WOODWINDS = "woodwinds"
    BRASS = "brass"
    FX = "fx"
    SYNTH = "synth"
    STRINGS = "strings"
    PERCUSSION = "percussion"
    KEYBOARD = "keyboard"
    GUITAR = "guitar"
    BASS = "bass"
    DRUMS = "drums"
    BACKING_VOCALS = "backing_vocals"
    VOCALS = "vocals"


class StemOperation(str, Enum):
    """Edit operations, each mapping 1:1 to an ACE-Step ``task_type``."""

    REGENERATE = "regenerate"
    REPLACE_STYLE = "replace_style"
    ADD_LAYER = "add_layer"
    SEPARATE = "separate"
    EXTEND = "extend"

    @property
    def task_type(self) -> str:
        """Return the ACE-Step task type this operation maps to."""
        return _OPERATION_TASK_TYPES[self]


_OPERATION_TASK_TYPES: dict[StemOperation, str] = {
    StemOperation.REGENERATE: "repaint",
    StemOperation.REPLACE_STYLE: "cover",
    StemOperation.ADD_LAYER: "lego",
    StemOperation.SEPARATE: "extract",
    StemOperation.EXTEND: "complete",
}


class StemSpec(BaseModel):
    """Specification for a single stem within a song.

    Attributes:
        stem_class: Track class (matches ACE-Step track vocabulary).
        role: Free-text role description ("driving four-on-the-floor groove").
        caption: Per-track text prompt used as the ``prompt`` field for lego tasks.
        bpm: Tempo override; falls back to the blueprint value when ``None``.
        key_scale: Musical key, e.g. ``"C major"``.
        time_signature: Meter, e.g. ``"4"`` for 4/4.
        duration: Stem duration in seconds.
        seed: Pinned seed for reproducible generation; ``None`` means random.
        source_start: Start of the source span (seconds) this stem was taken from.
        source_end: End of the source span (seconds).
        audio_path: Path of the rendered stem audio, if materialized.
    """

    stem_class: StemClass
    role: str = ""
    caption: str = ""
    bpm: Optional[int] = None
    key_scale: str = ""
    time_signature: str = ""
    duration: Optional[float] = None
    seed: Optional[int] = None
    source_start: Optional[float] = None
    source_end: Optional[float] = None
    audio_path: Optional[str] = None


class SongBlueprint(BaseModel):
    """Structured intermediate representation of a whole song.

    Mirrors ACE-Step's LM "song blueprint" concept: global metadata plus the
    per-stem specs needed to (re)generate any layer deterministically.
    """

    global_caption: str = ""
    lyrics: str = ""
    bpm: Optional[int] = None
    key_scale: str = ""
    time_signature: str = ""
    duration: Optional[float] = None
    vocal_language: str = "en"
    stems: list[StemSpec] = Field(default_factory=list)

    def stem(self, stem_class: StemClass) -> Optional[StemSpec]:
        """Return the first spec for ``stem_class``, or ``None`` if absent."""
        for spec in self.stems:
            if spec.stem_class == stem_class:
                return spec
        return None


class StemEdit(BaseModel):
    """A single prompt-driven change to one stem.

    Attributes:
        target: Stem class the edit applies to.
        intent: Natural-language intent text ("make the drums punchier").
        window_start: Edit window start in seconds (repaint window).
        window_end: Edit window end in seconds; ``None`` means until song end.
        operation: Operation, mapping 1:1 onto an ACE-Step task type.
        repaint_mode: Preservation mode for repaint edits.
        repaint_strength: Balanced-mode repaint intensity in [0, 1].
        cover_strength: ``audio_cover_strength`` for style-replacement edits.
        seed: Pinned seed; ``None`` means random seed.
    """

    target: StemClass
    intent: str = ""
    window_start: float = 0.0
    window_end: Optional[float] = None
    operation: StemOperation = StemOperation.REGENERATE
    repaint_mode: RepaintMode = "balanced"
    repaint_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    cover_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: Optional[int] = None


class EditPlan(BaseModel):
    """An ordered batch of stem edits plus how to put the song back together.

    Attributes:
        edits: Edits to apply, in order.
        recombine_instructions: Free-text mixdown/recombination notes.
        acceptance_criteria: Human/scorer-checkable criteria for accepting output.
    """

    edits: list[StemEdit] = Field(default_factory=list)
    recombine_instructions: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
