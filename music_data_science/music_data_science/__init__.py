"""Music Data Science (MDS): stem orchestration, data, and memory layer over ACE-Step 1.5.

The package wraps the ACE-Step ``/release_task`` HTTP API with a stem-oriented
domain model: prompt-driven stem edits are translated into typed requests
(repaint / cover / lego / extract / complete), executed through a thin client,
scored with Best-of-N evaluation, and persisted to SQLite + pandas telemetry.
"""

from music_data_science.stems.models import (
    EditPlan,
    SongBlueprint,
    StemClass,
    StemEdit,
    StemOperation,
    StemSpec,
)

__version__ = "0.1.0"

__all__ = [
    "EditPlan",
    "SongBlueprint",
    "StemClass",
    "StemEdit",
    "StemOperation",
    "StemSpec",
    "__version__",
]
