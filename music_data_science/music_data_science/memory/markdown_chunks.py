"""Heading-based chunking of markdown vault files (Obsidian-style).

Each chunk carries its heading path, source file, frontmatter-derived
authority weight, and file mtime so retrieval can score relevance, recency,
and source authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """One markdown section: heading path plus body text and source metadata."""

    text: str
    source: str = ""
    heading_path: tuple[str, ...] = ()
    authority: float = 0.5
    mtime: float = 0.0
    frontmatter: dict[str, str] = field(default_factory=dict)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split simple ``key: value`` YAML frontmatter from a markdown body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    frontmatter: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return frontmatter, "\n".join(lines[index + 1:])
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()
    return {}, text


def _heading_level(line: str) -> int:
    """Return the markdown heading level of ``line`` (0 when not a heading)."""
    stripped = line.lstrip()
    count = len(stripped) - len(stripped.lstrip("#"))
    return count if 0 < count <= 6 and stripped[count:count + 1] == " " else 0


def chunk_markdown(text: str, source: str = "", mtime: float = 0.0) -> list[Chunk]:
    """Chunk one markdown document by heading into :class:`Chunk` objects.

    Args:
        text: Full markdown text (frontmatter allowed).
        source: Source identifier (usually the file path).
        mtime: File modification time for recency scoring.

    Returns:
        One chunk per heading section with a non-empty body; the preamble
        before the first heading becomes a chunk with an empty heading path.
    """
    frontmatter, body = parse_frontmatter(text)
    try:
        authority = float(frontmatter.get("authority", 0.5))
    except ValueError:
        authority = 0.5
    if authority > 1.0:  # vault notes use a 1-10 scale; scoring expects [0, 1]
        authority = authority / 10.0
    authority = min(1.0, max(0.0, authority))
    chunks: list[Chunk] = []
    heading_stack: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            prefix = " > ".join(heading_stack)
            chunk_text = f"{prefix}\n{content}" if prefix else content
            chunks.append(Chunk(
                text=chunk_text,
                source=source,
                heading_path=tuple(heading_stack),
                authority=authority,
                mtime=mtime,
                frontmatter=frontmatter,
            ))
        buffer.clear()

    for line in body.splitlines():
        level = _heading_level(line)
        if level:
            flush()
            del heading_stack[level - 1:]
            heading_stack.append(line.lstrip().lstrip("#").strip())
        else:
            buffer.append(line)
    flush()
    return chunks


def chunk_vault(vault_dir: str | Path) -> list[Chunk]:
    """Chunk every ``*.md`` file under ``vault_dir`` (recursively, sorted)."""
    chunks: list[Chunk] = []
    for path in sorted(Path(vault_dir).rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, source=str(path), mtime=path.stat().st_mtime))
    return chunks
