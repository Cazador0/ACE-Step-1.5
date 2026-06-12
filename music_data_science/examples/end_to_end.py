"""End-to-end Music Data Science walkthrough.

Runs the full Strategy A loop: blueprint -> generate -> separate stems ->
prompt-driven stem edit -> Best-of-N scoring, with telemetry in pandas, an
SQLite system of record, queue messages on the task queue, and (when a vault
path is given) RAPTOR Best-of-N context retrieval.

Dry run (no server, no GPU — exercises every layer against a mock transport),
from the ``music_data_science/`` package root:

    PYTHONPATH=. uv run --with pandas --with pydantic --with httpx --with numpy \
        python examples/end_to_end.py --dry-run

Live run (after ``pip install -e .`` and starting an ACE-Step API server with a
base model, which is required for the extract/lego stem tasks):

    python examples/end_to_end.py --base-url http://127.0.0.1:8001 \
        --vault /path/to/awesome-python-audio/vault
"""

from __future__ import annotations

import argparse
import json

import httpx

from music_data_science.data.frames import TelemetryFrames
from music_data_science.data.queue import create_queue, generation_stream
from music_data_science.evaluation.scorer import BestOfNScorer
from music_data_science.memory.raptor import RaptorTree
from music_data_science.memory.store import MemoryStore
from music_data_science.pipeline.client import AceStepClient
from music_data_science.pipeline.orchestrator import StemSession
from music_data_science.stems.models import SongBlueprint, StemClass, StemSpec
from music_data_science.stems.translator import PromptTranslator

EDIT_PROMPT = "make the drums punchier and more aggressive from 0:10 to 0:20"


def mock_transport() -> httpx.MockTransport:
    """A fake ACE-Step server: accepts any task and reports instant success."""
    counter = {"task": 0}

    def envelope(data: object) -> dict:
        return {"data": data, "code": 200, "error": None, "timestamp": 0, "extra": None}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/release_task"):
            counter["task"] += 1
            return httpx.Response(200, json=envelope({"task_id": f"dry-run-{counter['task']}"}))
        body = json.loads(request.content)
        task_ids = json.loads(body["task_id_list"])
        items = [{
            "task_id": task_id,
            "status": 1,
            "progress_text": "done",
            "result": json.dumps([{
                "file": f"/outputs/{task_id}.mp3",
                "status": 1,
                "metas": {"bpm": 96, "keyscale": "C minor", "duration": 30.0},
            }]),
        } for task_id in task_ids]
        return httpx.Response(200, json=envelope(items))

    return httpx.MockTransport(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--dry-run", action="store_true", help="use a mock ACE-Step server")
    parser.add_argument("--vault", default=None, help="markdown vault for RAPTOR context retrieval")
    parser.add_argument("--redis-url", default=None, help="optional Redis/Valkey URL for cache+queue")
    args = parser.parse_args()

    transport = mock_transport() if args.dry_run else None
    client = AceStepClient(args.base_url, transport=transport)
    store = MemoryStore(":memory:" if args.dry_run else "mds.db")
    frames = TelemetryFrames()
    queue = create_queue(args.redis_url)

    if args.vault:
        context = RaptorTree(args.vault).build().best_of_n("how to regenerate a drum stem", n=5)
        if context.winner:
            print(f"[memory] context: {context.winner.source} (score {context.score:.2f})")

    blueprint = SongBlueprint(
        global_caption="dusty lofi hip hop, warm tape saturation, mellow rhodes",
        bpm=96, key_scale="C minor", time_signature="4/4", duration=30.0,
        stems=[StemSpec(stem_class=StemClass.DRUMS, caption="dusty boom-bap kit, heavy swing")],
    )

    session = StemSession(client=client, blueprint=blueprint, store=store, frames=frames)
    queue.enqueue(generation_stream(), {"task": "generate", "caption": blueprint.global_caption})

    outcome = session.generate()
    print(f"[generate] files: {outcome.result.files}")

    plan = PromptTranslator().translate(EDIT_PROMPT)
    edit = plan.edits[0]
    print(f"[translate] {EDIT_PROMPT!r} -> {edit.operation.value} "
          f"window {edit.window_start}-{edit.window_end}s mode={edit.repaint_mode}")

    candidates = session.best_of_n(edit, n=3, scorer=BestOfNScorer(blueprint))
    winner = candidates[0].breakdown
    print(f"[best-of-n] winner total={winner.total:.2f} passed={winner.passed}")

    print(f"[telemetry] {len(frames.generations())} generations, {len(frames.scores())} scores recorded")


if __name__ == "__main__":
    main()
