"""Thin httpx wrapper around the ACE-Step ``/release_task`` + ``/query_result`` API.

The server wraps every response in the envelope
``{"data": ..., "code": int, "error": str | None, "timestamp": int, "extra": None}``.
``/release_task`` returns ``{"task_id", "status": "queued", "queue_position"}``;
``/query_result`` returns one item per task ID with an integer ``status``
(0 = queued/running, 1 = succeeded, 2 = failed) and a JSON-encoded ``result``
string holding a list of ``{"file", "metas", ...}`` payloads.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional

import httpx


class AceStepError(Exception):
    """Base error for ACE-Step client failures."""


class AceStepAPIError(AceStepError):
    """Raised when the HTTP API returns a non-success envelope or status."""


class AceStepTaskError(AceStepError):
    """Raised when a generation task finishes in the failed state."""


class AceStepTimeoutError(AceStepError):
    """Raised when polling exceeds the configured timeout."""


class TaskStatus(IntEnum):
    """Integer task status codes used by ``/query_result``."""

    PENDING = 0
    SUCCEEDED = 1
    FAILED = 2


@dataclass
class TaskResult:
    """Parsed ``/query_result`` item for one task.

    Attributes:
        task_id: Server-assigned task identifier.
        status: Integer task status.
        files: Output audio file paths reported by the server.
        metas: Metadata of the first result item (bpm, duration, keyscale, ...).
        progress_text: Latest server-side progress/log text.
        items: All decoded result items, preserving the raw contract.
    """

    task_id: str
    status: TaskStatus
    files: list[str] = field(default_factory=list)
    metas: dict[str, Any] = field(default_factory=dict)
    progress_text: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)


def _parse_result_item(item: dict[str, Any]) -> TaskResult:
    """Convert one raw ``/query_result`` item into a :class:`TaskResult`."""
    try:
        decoded = json.loads(item.get("result") or "[]")
    except (TypeError, json.JSONDecodeError):
        decoded = []
    items = [entry for entry in decoded if isinstance(entry, dict)] if isinstance(decoded, list) else []
    files = [entry["file"] for entry in items if entry.get("file")]
    metas = items[0].get("metas", {}) if items else {}
    return TaskResult(
        task_id=str(item.get("task_id", "")),
        status=TaskStatus(int(item.get("status", 0))),
        files=files,
        metas=metas if isinstance(metas, dict) else {},
        progress_text=str(item.get("progress_text") or ""),
        items=items,
    )


class AceStepClient:
    """Synchronous client for submitting and polling ACE-Step generation tasks.

    Args:
        base_url: Root URL of the ACE-Step API server.
        token: Optional bearer token sent as ``Authorization``.
        timeout: Per-request HTTP timeout in seconds.
        poll_interval: Default delay between ``/query_result`` polls.
        transport: Optional httpx transport (use ``httpx.MockTransport`` in tests).
        sleep: Injectable sleep function, for deterministic poll tests.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        token: Optional[str] = None,
        timeout: float = 30.0,
        poll_interval: float = 2.0,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.poll_interval = poll_interval
        self._sleep = sleep
        self._http = httpx.Client(base_url=base_url, timeout=timeout, headers=headers, transport=transport)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "AceStepClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _unwrap(self, response: httpx.Response) -> Any:
        """Validate the standard response envelope and return its ``data``."""
        response.raise_for_status()
        envelope = response.json()
        if envelope.get("code") != 200 or envelope.get("error"):
            raise AceStepAPIError(f"API error (code={envelope.get('code')}): {envelope.get('error')}")
        return envelope.get("data")

    def submit(self, payload: dict[str, Any]) -> str:
        """Submit a ``GenerateMusicRequest`` payload; return the queued task ID."""
        data = self._unwrap(self._http.post("/release_task", json=payload))
        task_id = (data or {}).get("task_id")
        if not task_id:
            raise AceStepAPIError(f"missing task_id in /release_task response: {data!r}")
        return str(task_id)

    def query(self, task_ids: list[str]) -> list[TaskResult]:
        """Query current results for ``task_ids`` in one batch call."""
        body = {"task_id_list": json.dumps(task_ids)}
        data = self._unwrap(self._http.post("/query_result", json=body))
        return [_parse_result_item(item) for item in (data or [])]

    def poll(
        self,
        task_id: str,
        timeout: float = 600.0,
        interval: Optional[float] = None,
        raise_on_failure: bool = True,
    ) -> TaskResult:
        """Poll ``task_id`` until it succeeds or fails.

        Args:
            task_id: Task to wait for.
            timeout: Maximum total wait in seconds.
            interval: Poll delay; defaults to ``self.poll_interval``.
            raise_on_failure: Raise :class:`AceStepTaskError` on failed tasks.

        Returns:
            The terminal :class:`TaskResult`.

        Raises:
            AceStepTimeoutError: If the task does not finish within ``timeout``.
            AceStepTaskError: If the task failed and ``raise_on_failure`` is set.
        """
        delay = self.poll_interval if interval is None else interval
        waited = 0.0
        while True:
            results = self.query([task_id])
            result = results[0] if results else TaskResult(task_id=task_id, status=TaskStatus.PENDING)
            if result.status is TaskStatus.FAILED and raise_on_failure:
                raise AceStepTaskError(f"task {task_id} failed: {result.progress_text or 'unknown error'}")
            if result.status is not TaskStatus.PENDING:
                return result
            if waited >= timeout:
                raise AceStepTimeoutError(f"task {task_id} still pending after {timeout}s")
            self._sleep(delay)
            waited += delay
