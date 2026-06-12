"""Queue consumer driving stem workflows from the Strategy A task queue.

:class:`QueueWorker` claims messages from one stream within a consumer group
and acknowledges a message only when its handler succeeds, so failed work
stays pending (re-claimable, visible via ``TaskQueue.pending``) instead of
being lost.  :func:`session_handler` is the thin adapter that turns
generation-stream messages into :class:`~music_data_science.pipeline.orchestrator.StemSession`
calls.
"""

from __future__ import annotations

from typing import Callable

from music_data_science.data.queue import QueueMessage, TaskQueue, generation_stream
from music_data_science.pipeline.orchestrator import StemSession
from music_data_science.stems.translator import PromptTranslator


class QueueWorker:
    """Claim-and-handle loop over one stream with ack-on-success semantics.

    Handler exceptions are recorded in :attr:`errors` and leave the message
    pending for its consumer group, so a crashed job can be re-claimed or
    inspected rather than silently dropped.

    Args:
        queue: Stream queue backend (Redis Streams or the in-memory fallback).
        handler: Callable invoked once per claimed message; raising marks failure.
        stream: Stream to consume; defaults to the generation stream.
        group: Consumer-group name.
        consumer: This worker's consumer name within the group.

    Attributes:
        errors: ``(message_id, exception)`` pairs for every failed handler call.
    """

    def __init__(
        self,
        queue: TaskQueue,
        handler: Callable[[QueueMessage], None],
        stream: str = generation_stream(),
        group: str = "pipeline",
        consumer: str = "worker-1",
    ) -> None:
        self.queue = queue
        self.handler = handler
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.errors: list[tuple[str, Exception]] = []

    def _claim_and_process(self, count: int) -> tuple[int, int]:
        """Claim up to ``count`` messages; return ``(claimed, processed)``."""
        messages = self.queue.claim(self.stream, self.group, self.consumer, count=count)
        processed = 0
        for message in messages:
            try:
                self.handler(message)
            except Exception as error:  # noqa: BLE001 -- failed work must stay pending, not kill the worker
                self.errors.append((message.message_id, error))
                continue
            self.queue.ack(self.stream, self.group, message.message_id)
            processed += 1
        return len(messages), processed

    def run_once(self, count: int = 1) -> int:
        """Claim up to ``count`` messages and handle each; return how many succeeded.

        A message is acknowledged only when the handler returns without
        raising; failed messages stay pending and their errors are recorded.
        """
        return self._claim_and_process(count)[1]

    def drain(self, max_messages: int = 100) -> int:
        """Run :meth:`run_once` until the stream yields nothing or the cap is hit.

        Args:
            max_messages: Maximum number of messages to claim in this call.

        Returns:
            Total number of successfully processed (acknowledged) messages.
        """
        claimed_total = 0
        processed_total = 0
        while claimed_total < max_messages:
            claimed, processed = self._claim_and_process(1)
            if claimed == 0:
                break
            claimed_total += claimed
            processed_total += processed
        return processed_total


def session_handler(session: StemSession, translator: PromptTranslator) -> Callable[[QueueMessage], None]:
    """Adapter dispatching generation-queue messages onto a :class:`StemSession`.

    Messages carry ``{"kind": "generate" | "edit", "prompt": ...}``:
    ``generate`` renders the session's blueprint, ``edit`` translates the
    prompt and regenerates the targeted stem.  Unknown kinds raise
    :class:`ValueError` so the message stays pending for a human to triage.

    Args:
        session: Session whose blueprint and stems the messages operate on.
        translator: Prompt translator used for ``edit`` messages.

    Returns:
        A handler suitable for :class:`QueueWorker`.
    """

    def handle(message: QueueMessage) -> None:
        """Dispatch one queue message to the session."""
        kind = message.fields.get("kind", "")
        if kind == "generate":
            session.generate()
        elif kind == "edit":
            plan = translator.translate(message.fields.get("prompt", ""))
            session.regenerate(plan.edits[0])
        else:
            raise ValueError(f"unknown message kind: {kind!r}")

    return handle
