"""Tests for the queue worker and the StemSession message adapter (mocked HTTP)."""

import json
import unittest

import httpx

from music_data_science.data.queue import MemoryStreamQueue, QueueMessage, generation_stream
from music_data_science.memory.store import MemoryStore
from music_data_science.pipeline.client import AceStepClient
from music_data_science.pipeline.orchestrator import StemSession
from music_data_science.pipeline.worker import QueueWorker, session_handler
from music_data_science.stems.models import SongBlueprint, StemClass, StemSpec
from music_data_science.stems.translator import PromptTranslator


class FakeAceStep:
    """Mock transport: assigns task IDs and succeeds every task immediately."""

    def __init__(self):
        self.submitted = []
        self._counter = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/release_task":
            self._counter += 1
            task_id = f"task-{self._counter}"
            self.submitted.append((task_id, body))
            data = {"task_id": task_id, "status": "queued", "queue_position": 1}
        else:
            task_ids = json.loads(body["task_id_list"])
            data = [{
                "task_id": task_id,
                "status": 1,
                "result": json.dumps([{
                    "file": f"/out/{task_id}.wav",
                    "status": 1,
                    "metas": {"bpm": 84, "duration": 120.0, "keyscale": "F major"},
                }]),
                "progress_text": "",
            } for task_id in task_ids]
        return httpx.Response(200, json={"data": data, "code": 200, "error": None, "timestamp": 0, "extra": None})


def make_session():
    """Build a StemSession over a mocked client and in-memory store."""
    server = FakeAceStep()
    client = AceStepClient(transport=httpx.MockTransport(server), sleep=lambda _s: None)
    blueprint = SongBlueprint(
        global_caption="warm lo-fi hip hop", bpm=84, key_scale="F major", duration=120.0,
        stems=[StemSpec(stem_class=StemClass.DRUMS, caption="dusty kit")],
    )
    return StemSession(client, blueprint, store=MemoryStore(":memory:"), name="test"), server


class RecordingHandler:
    """Fake handler that records handled fields and fails on demand."""

    def __init__(self, fail_on: frozenset[str] = frozenset()):
        self.fail_on = fail_on
        self.handled: list[dict[str, str]] = []

    def __call__(self, message: QueueMessage) -> None:
        if message.fields.get("task") in self.fail_on:
            raise RuntimeError(f"boom: {message.fields['task']}")
        self.handled.append(message.fields)


class QueueWorkerTest(unittest.TestCase):
    """Ack-on-success, pending-on-failure, and draining."""

    def test_success_acks_message(self):
        queue = MemoryStreamQueue()
        queue.enqueue(generation_stream(), {"task": "a"})
        handler = RecordingHandler()
        worker = QueueWorker(queue, handler)
        self.assertEqual(worker.run_once(), 1)
        self.assertEqual(handler.handled, [{"task": "a"}])
        self.assertEqual(queue.pending(generation_stream(), "pipeline"), 0)
        self.assertEqual(worker.errors, [])

    def test_failure_leaves_message_pending_and_records_error(self):
        queue = MemoryStreamQueue()
        queue.enqueue(generation_stream(), {"task": "bad"})
        worker = QueueWorker(queue, RecordingHandler(fail_on=frozenset({"bad"})))
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(queue.pending(generation_stream(), "pipeline"), 1)
        self.assertEqual(len(worker.errors), 1)
        self.assertIsInstance(worker.errors[0][1], RuntimeError)

    def test_drain_processes_multiple_and_stops_when_empty(self):
        queue = MemoryStreamQueue()
        for task in ("a", "b", "c"):
            queue.enqueue(generation_stream(), {"task": task})
        handler = RecordingHandler()
        worker = QueueWorker(queue, handler)
        self.assertEqual(worker.drain(), 3)
        self.assertEqual([fields["task"] for fields in handler.handled], ["a", "b", "c"])
        self.assertEqual(queue.pending(generation_stream(), "pipeline"), 0)

    def test_drain_skips_failures_but_keeps_going(self):
        queue = MemoryStreamQueue()
        for task in ("a", "bad", "b"):
            queue.enqueue(generation_stream(), {"task": task})
        worker = QueueWorker(queue, RecordingHandler(fail_on=frozenset({"bad"})))
        self.assertEqual(worker.drain(), 2)
        self.assertEqual(queue.pending(generation_stream(), "pipeline"), 1)

    def test_drain_respects_message_cap(self):
        queue = MemoryStreamQueue()
        for task in ("a", "b", "c"):
            queue.enqueue(generation_stream(), {"task": task})
        worker = QueueWorker(queue, RecordingHandler())
        self.assertEqual(worker.drain(max_messages=2), 2)


class SessionHandlerTest(unittest.TestCase):
    """The adapter dispatches generate/edit messages onto a StemSession."""

    def test_generate_message_runs_text2music(self):
        session, server = make_session()
        queue = MemoryStreamQueue()
        queue.enqueue(generation_stream(), {"kind": "generate"})
        worker = QueueWorker(queue, session_handler(session, PromptTranslator()))
        self.assertEqual(worker.run_once(), 1)
        self.assertEqual(server.submitted[0][1]["task_type"], "text2music")
        self.assertIsNotNone(session.song_path)
        self.assertEqual(queue.pending(generation_stream(), "pipeline"), 0)

    def test_edit_message_translates_and_regenerates(self):
        session, server = make_session()
        session.song_path = "/out/song.wav"
        queue = MemoryStreamQueue()
        queue.enqueue(
            generation_stream(),
            {"kind": "edit", "prompt": "make the drums punchier from 0:45 to 1:10"},
        )
        worker = QueueWorker(queue, session_handler(session, PromptTranslator()))
        self.assertEqual(worker.run_once(), 1)
        payload = server.submitted[0][1]
        self.assertEqual(payload["task_type"], "repaint")
        self.assertEqual(payload["repainting_start"], 45.0)

    def test_unknown_kind_raises_and_stays_pending(self):
        session, _server = make_session()
        queue = MemoryStreamQueue()
        queue.enqueue(generation_stream(), {"kind": "delete"})
        worker = QueueWorker(queue, session_handler(session, PromptTranslator()))
        self.assertEqual(worker.run_once(), 0)
        self.assertEqual(queue.pending(generation_stream(), "pipeline"), 1)
        self.assertIsInstance(worker.errors[0][1], ValueError)


if __name__ == "__main__":
    unittest.main()
