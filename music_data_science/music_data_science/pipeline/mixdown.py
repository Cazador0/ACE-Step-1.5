"""Deterministic local recombination (mixdown) of stem WAV files.

ACE-Step has no "mix" task type -- recombining already-rendered stems is plain
signal addition, so it is done locally with the stdlib ``wave`` module and
numpy: sum 16-bit PCM stems sample-by-sample and peak-normalize to avoid
clipping.  All stems must share sample rate and channel count.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def _read_wav(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Read a 16-bit PCM WAV file; return (samples, sample_rate, channels)."""
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV stems are supported")
        frames = handle.readframes(handle.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        return samples, handle.getframerate(), handle.getnchannels()


def mix_wavs(stem_paths: list[str | Path], output_path: str | Path) -> Path:
    """Mix stem WAV files into one peak-normalized 16-bit PCM WAV.

    Args:
        stem_paths: Paths of 16-bit PCM WAV stems with matching rate/channels.
        output_path: Destination WAV path.

    Returns:
        The written output path.

    Raises:
        ValueError: If no stems are given or formats are inconsistent.
    """
    if not stem_paths:
        raise ValueError("at least one stem is required for a mixdown")
    mixed: np.ndarray | None = None
    rate = channels = 0
    for path in stem_paths:
        samples, sample_rate, channel_count = _read_wav(path)
        if mixed is None:
            mixed, rate, channels = samples, sample_rate, channel_count
            continue
        if (sample_rate, channel_count) != (rate, channels):
            raise ValueError(f"{path}: sample rate/channels mismatch with first stem")
        if len(samples) > len(mixed):
            mixed = np.pad(mixed, (0, len(samples) - len(mixed)))
        mixed[: len(samples)] += samples

    assert mixed is not None
    peak = float(np.max(np.abs(mixed))) or 1.0
    limit = float(np.iinfo(np.int16).max)
    if peak > limit:
        mixed = mixed * (limit / peak)
    output = Path(output_path)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(mixed.astype(np.int16).tobytes())
    return output
