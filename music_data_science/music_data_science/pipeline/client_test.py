"""Tests for AceStepClient using httpx.MockTransport (no network)."""

import json
import unittest

import httpx

from music_data_science.pipeline.client import (
    AceStepAPIError,
    AceStepClient,
    AceStepTaskError,
    AceStepTimeoutError,
    TaskStatus,
)


def envelope(data, code=200, error=None):
    """Build the standard ACE-Step response envelope."""
    return {"data": data, "code": code, "error": error, "timestamp": 0, "extra": None}


def query_item(task_id, status, files=(), metas=None):
    """Build one /query_result item with a JSON-encoded result string."""
    result = [
        {"file": path, "status": status, "metas": metas or {}}
        for path in files
    ] or [{"file": "", "status": status, "metas": metas or {}}]
    return {"task_id": task_id, "result": json.dumps(result), "status": status, "progress_text": "step 3/8"}


class FakeServer:
    """Scriptable handler emulating /release_task + /query_result."""

    def __init__(self, query_responses):
        self.query_responses = list(query_responses)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/release_task":
            return httpx.Response(200, json=envelope({"task_id": "t-1", "status": "queued", "queue_position": 1}))
        if request.url.path == "/query_result":
            payload = self.query_responses.pop(0) if len(self.query_responses) > 1 else self.query_responses[0]
            return httpx.Response(200, json=envelope(payload))
        return httpx.Response(404)

    def client(self, **kwargs):
        return AceStepClient(transport=httpx.MockTransport(self), sleep=lambda _seconds: None, **kwargs)


class SubmitTest(unittest.TestCase):
    """Submission contract."""

    def test_submit_returns_task_id_and_posts_payload(self):
        server = FakeServer([[]])
        with server.client() as client:
            task_id = client.submit({"task_type": "text2music", "prompt": "lo-fi"})
        self.assertEqual(task_id, "t-1")
        body = json.loads(server.requests[0].content)
        self.assertEqual(body["task_type"], "text2music")

    def test_submit_raises_on_error_envelope(self):
        def handler(_request):
            return httpx.Response(200, json=envelope(None, code=500, error="boom"))

        with AceStepClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(AceStepAPIError):
                client.submit({})

    def test_token_sent_as_bearer_header(self):
        server = FakeServer([[]])
        with AceStepClient(token="secret", transport=httpx.MockTransport(server)) as client:
            client.submit({})
        self.assertEqual(server.requests[0].headers["Authorization"], "Bearer secret")


class QueryAndPollTest(unittest.TestCase):
    """Polling contract: status 0 pending, 1 succeeded, 2 failed."""

    def test_query_parses_files_and_metas(self):
        metas = {"bpm": 84, "duration": 120.0, "keyscale": "F major"}
        server = FakeServer([[query_item("t-1", 1, files=["/out/a.mp3", "/out/b.mp3"], metas=metas)]])
        with server.client() as client:
            results = client.query(["t-1"])
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].status, TaskStatus.SUCCEEDED)
        self.assertEqual(results[0].files, ["/out/a.mp3", "/out/b.mp3"])
        self.assertEqual(results[0].metas["bpm"], 84)
        self.assertEqual(results[0].progress_text, "step 3/8")

    def test_query_sends_json_encoded_task_id_list(self):
        server = FakeServer([[query_item("t-1", 1)]])
        with server.client() as client:
            client.query(["t-1", "t-2"])
        body = json.loads(server.requests[0].content)
        self.assertEqual(json.loads(body["task_id_list"]), ["t-1", "t-2"])

    def test_poll_waits_through_pending_then_succeeds(self):
        server = FakeServer([
            [query_item("t-1", 0)],
            [query_item("t-1", 0)],
            [query_item("t-1", 1, files=["/out/a.mp3"])],
        ])
        with server.client() as client:
            result = client.poll("t-1", timeout=60.0, interval=1.0)
        self.assertIs(result.status, TaskStatus.SUCCEEDED)
        self.assertEqual(result.files, ["/out/a.mp3"])

    def test_poll_raises_on_failed_task(self):
        server = FakeServer([[query_item("t-1", 2)]])
        with server.client() as client:
            with self.assertRaises(AceStepTaskError):
                client.poll("t-1")

    def test_poll_can_return_failure_without_raising(self):
        server = FakeServer([[query_item("t-1", 2)]])
        with server.client() as client:
            result = client.poll("t-1", raise_on_failure=False)
        self.assertIs(result.status, TaskStatus.FAILED)

    def test_poll_times_out(self):
        server = FakeServer([[query_item("t-1", 0)]])
        with server.client() as client:
            with self.assertRaises(AceStepTimeoutError):
                client.poll("t-1", timeout=3.0, interval=1.0)

    def test_malformed_result_string_is_tolerated(self):
        item = {"task_id": "t-1", "result": "not json", "status": 1, "progress_text": ""}
        server = FakeServer([[item]])
        with server.client() as client:
            result = client.query(["t-1"])[0]
        self.assertEqual(result.files, [])
        self.assertEqual(result.metas, {})


if __name__ == "__main__":
    unittest.main()
