"""Keyword vocabulary tables backing the deterministic prompt grammar.

Kept separate from :mod:`music_data_science.stems.translator` so the parsing
logic stays small and the vocabulary stays easy to review and extend.
"""

from __future__ import annotations

from music_data_science.stems.models import StemClass, StemOperation

# (phrase, stem) pairs; matching prefers the earliest, longest phrase in the
# prompt so "backing vocals" wins over "vocals" at the same position.
STEM_SYNONYMS: list[tuple[str, StemClass]] = [
    ("backing vocals", StemClass.BACKING_VOCALS),
    ("backup vocals", StemClass.BACKING_VOCALS),
    ("harmonies", StemClass.BACKING_VOCALS),
    ("vocals", StemClass.VOCALS),
    ("vocal", StemClass.VOCALS),
    ("voice", StemClass.VOCALS),
    ("singer", StemClass.VOCALS),
    ("singing", StemClass.VOCALS),
    ("drums", StemClass.DRUMS),
    ("drum", StemClass.DRUMS),
    ("kick", StemClass.DRUMS),
    ("snare", StemClass.DRUMS),
    ("hi-hat", StemClass.DRUMS),
    ("percussion", StemClass.PERCUSSION),
    ("shaker", StemClass.PERCUSSION),
    ("tambourine", StemClass.PERCUSSION),
    ("conga", StemClass.PERCUSSION),
    ("bassline", StemClass.BASS),
    ("bass", StemClass.BASS),
    ("guitar", StemClass.GUITAR),
    ("keyboard", StemClass.KEYBOARD),
    ("keys", StemClass.KEYBOARD),
    ("piano", StemClass.KEYBOARD),
    ("organ", StemClass.KEYBOARD),
    ("rhodes", StemClass.KEYBOARD),
    ("synthesizer", StemClass.SYNTH),
    ("synth", StemClass.SYNTH),
    ("strings", StemClass.STRINGS),
    ("violin", StemClass.STRINGS),
    ("cello", StemClass.STRINGS),
    ("brass", StemClass.BRASS),
    ("trumpet", StemClass.BRASS),
    ("trombone", StemClass.BRASS),
    ("horns", StemClass.BRASS),
    ("horn", StemClass.BRASS),
    ("woodwinds", StemClass.WOODWINDS),
    ("flute", StemClass.WOODWINDS),
    ("clarinet", StemClass.WOODWINDS),
    ("saxophone", StemClass.WOODWINDS),
    ("sax", StemClass.WOODWINDS),
    ("fx", StemClass.FX),
    ("effects", StemClass.FX),
    ("riser", StemClass.FX),
    ("sweep", StemClass.FX),
]

# Operation cue words, checked in order; first matching operation wins.
# Regenerate is the fallback and therefore has no cue list here.
OPERATION_CUES: list[tuple[StemOperation, tuple[str, ...]]] = [
    (StemOperation.SEPARATE, ("separate", "extract", "isolate", "split out")),
    (StemOperation.ADD_LAYER, ("add", "layer", "overlay", "introduce", "bring in")),
    (StemOperation.REPLACE_STYLE, ("replace", "swap", "substitute")),
    (StemOperation.EXTEND, ("extend", "continue", "complete", "finish", "lengthen")),
]

# Repaint preservation heuristics: "subtle/tweak" -> conservative,
# "completely different" -> aggressive, everything else -> balanced.
CONSERVATIVE_CUES: tuple[str, ...] = (
    "subtle", "subtly", "slight", "slightly", "tweak", "gentle", "gently",
    "a bit", "a little", "minor", "barely", "touch up",
)
AGGRESSIVE_CUES: tuple[str, ...] = (
    "completely", "totally", "entirely", "drastically", "radically",
    "from scratch", "overhaul", "completely different", "reimagine",
)
