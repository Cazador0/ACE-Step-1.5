"""Real text-embedding backends behind the injectable embedding hooks.

The deterministic hashed default (:func:`~music_data_science.memory.raptor.default_embed`)
keeps every workflow reproducible and dependency-free; it is always available.
``sentence-transformers`` is the cheap semantic upgrade for caption/genre
similarity scoring and RAPTOR vault retrieval.  The CLAP text tower embeds
text into the same space as audio, so scores stay comparable once audio-side
embedding lands.  All heavy dependencies are imported lazily
(``pip install music-data-science[embeddings]``); importing this module never
requires them.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from music_data_science.memory.raptor import cosine, default_embed

DEFAULT_SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2"
DEFAULT_CLAP_MODEL = "laion/clap-htsat-unfused"

_INSTALL_HINT = "pip install music-data-science[embeddings]"


def _sentence_transformer_embedder(model_name: str) -> Callable[[str], list[float]]:
    """Build an embedder over a ``sentence_transformers`` model; raises ImportError with install help."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 -- optional dependency, imported lazily
    except ImportError as error:
        raise ImportError(
            f"the 'sentence-transformers' backend needs the sentence-transformers package; {_INSTALL_HINT}"
        ) from error
    model = SentenceTransformer(model_name)

    def embed(text: str) -> list[float]:
        """Encode one string to a normalized embedding vector."""
        return [float(value) for value in model.encode(text, normalize_embeddings=True)]

    return embed


def _clap_embedder(model_name: str) -> Callable[[str], list[float]]:
    """Build a CLAP text-tower embedder; raises ImportError with install help."""
    try:
        from transformers import ClapModel, ClapProcessor  # noqa: PLC0415 -- optional dependency, imported lazily
    except ImportError as error:
        raise ImportError(f"the 'clap' backend needs the transformers package; {_INSTALL_HINT}") from error
    model = ClapModel.from_pretrained(model_name)
    processor = ClapProcessor.from_pretrained(model_name)

    def embed(text: str) -> list[float]:
        """Embed one string via CLAP ``get_text_features``, L2-normalized."""
        inputs = processor(text=[text], return_tensors="pt")
        values = [float(value) for value in model.get_text_features(**inputs)[0].tolist()]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values

    return embed


def create_embedder(backend: str = "auto", model_name: Optional[str] = None) -> Callable[[str], list[float]]:
    """Create a text-embedding function for the injectable embedding hooks.

    Args:
        backend: One of ``"hashed"`` (deterministic, no dependencies),
            ``"sentence-transformers"``, ``"clap"`` (text tower), or
            ``"auto"`` (sentence-transformers when installed, hashed otherwise).
        model_name: Backend-specific model identifier; sensible default per backend.

    Returns:
        A ``text -> list[float]`` embedding function with L2-normalized output.

    Raises:
        ImportError: If an explicit backend's optional dependency is missing.
        ValueError: If ``backend`` is not one of the supported names.
    """
    if backend == "hashed":
        return default_embed
    if backend == "sentence-transformers":
        return _sentence_transformer_embedder(model_name or DEFAULT_SENTENCE_TRANSFORMER)
    if backend == "clap":
        return _clap_embedder(model_name or DEFAULT_CLAP_MODEL)
    if backend == "auto":
        try:
            return _sentence_transformer_embedder(model_name or DEFAULT_SENTENCE_TRANSFORMER)
        except ImportError:
            return default_embed
    raise ValueError(f"unknown embedding backend: {backend!r}")


def similarity_from_embedder(embed: Callable[[str], list[float]]) -> Callable[[str, str], float]:
    """Turn an embedder into a similarity hook for ``BestOfNScorer(embedding_similarity=...)``.

    Returns raw cosine similarity floored at 0.0: for L2-normalized text
    embeddings, anti-correlated vectors mean "not similar" rather than a
    meaningful negative signal, and flooring keeps the hook inside the [0, 1]
    range the scorer expects without compressing the useful positive band
    (which a ``(cos + 1) / 2`` remap would).

    Args:
        embed: Text embedding function, e.g. from :func:`create_embedder`.

    Returns:
        A ``(expected_text, candidate_text) -> [0, 1]`` similarity function.
    """

    def similarity(expected: str, candidate: str) -> float:
        """Cosine similarity of the two texts' embeddings, floored at 0.0."""
        return max(0.0, cosine(embed(expected), embed(candidate)))

    return similarity
