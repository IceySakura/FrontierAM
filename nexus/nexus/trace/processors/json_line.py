# /// script
# dependencies = []
#
# [project]
# name = "json_line"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# span_processors = { json_line = "nexus.trace.processors.json_line" }
# ///

from __future__ import annotations

import json
import logging
import queue
import threading

from pathlib import Path

from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.types import Span


logger = logging.getLogger(__name__)


@span_processor_registry.register('json_line')
class JsonLineSpanProcessor(SpanProcessor):
    """A processor that logs span events to a JSON line file."""

    def __init__(self, file_path: str | Path, append: bool = True):
        """Initialize the JSON line processor.

        Args:
            file_path: Path to the JSONL file
            append: Whether to append to existing file (True) or overwrite (False)
        """
        self.file_path = Path(file_path)
        self.append = append
        self._queue = queue.Queue()
        self._writer_thread = None

        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.append:
            with open(self.file_path, 'w'):
                pass

        self._start_writer_thread()

    def _start_writer_thread(self):
        """Start the background writer thread."""
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _writer_loop(self):
        with open(self.file_path, 'a') as f:
            while True:
                try:
                    event = self._queue.get()
                    if event is None:
                        break
                    json.dump(event, f, default=str)
                    f.write('\n')
                    f.flush()
                except Exception as e:
                    logger.error('Error writing event to file: %s', e)

    def finalize(self):
        self._queue.put(None)
        if self._writer_thread:
            self._writer_thread.join(timeout=1.0)

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        self._queue.put(event.model_dump())

    def on_update(self, span: Span, event: SpanUpdateEvent) -> None:
        self._queue.put(event.model_dump())

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        self._queue.put(event.model_dump())
