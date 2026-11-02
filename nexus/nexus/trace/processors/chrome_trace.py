# /// script
# dependencies = []
#
# [project]
# name = "chrome_trace"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# span_processors = { chrome_trace = "nexus.trace.processors.chrome_trace" }
# ///

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from nexus.trace.events import SpanEndEvent, SpanStartEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.types import Span
from nexus.utils.common import serialize_object


@span_processor_registry.register('chrome_trace')
class ChromeTraceProcessor(SpanProcessor):
    """A processor that converts span events to Chrome Trace Event Format and writes to a JSON file.

    For task spans (attribute.type == 'task'), we assign them to virtual threads to better
    visualize concurrent tasks that may share the same OS thread_id.
    """

    def __init__(self, file_path: str | Path):
        """Initialize the Chrome Trace processor.

        Args:
            file_path: Path to the output JSON file
        """
        self.file_path = Path(file_path)
        self._events = []

        # Virtual thread management for tasks
        self._task_virtual_threads = {}  # task_span_id -> virtual_tid
        self._span_to_task = {}  # span_id -> task_span_id (for children of tasks)
        self._free_virtual_threads = []  # List of available virtual thread IDs
        self._next_virtual_tid = 1  # Global counter for virtual thread IDs

        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        """Called when a span is started.

        Args:
            span: The newly created Span object
            event: The SpanStartEvent that triggered the creation
        """
        # If this is a task span, allocate a virtual thread
        if span.attributes.get('type') == 'task':
            vtid = self._get_virtual_thread_for_task(span.span_id)
            self._task_virtual_threads[span.span_id] = vtid
        elif span.parent_span_id:
            # Check if parent belongs to a task
            if span.parent_span_id in self._span_to_task:
                # Parent is a child of a task, inherit the same task
                task_span_id = self._span_to_task[span.parent_span_id]
                self._span_to_task[span.span_id] = task_span_id
            elif span.parent_span_id in self._task_virtual_threads:
                # Parent is a task itself
                self._span_to_task[span.span_id] = span.parent_span_id

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        """Called when a span is ended.

        Args:
            span: The completed Span object (with end_time set)
            event: The SpanEndEvent that triggered the end
        """
        # Convert to Chrome trace event and add to events
        chrome_event = self._convert_to_chrome_trace_event(span)
        if chrome_event:
            self._events.append(chrome_event)

        # Clean up tracking data
        if span.span_id in self._span_to_task:
            del self._span_to_task[span.span_id]

        # Release virtual thread if this was a task span
        if span.attributes.get('type') == 'task' and span.span_id in self._task_virtual_threads:
            self._release_virtual_thread(span.span_id)
            del self._task_virtual_threads[span.span_id]

    def finalize(self):
        """Stop the processor and write the final trace file."""
        # Write the complete trace file
        trace_data = {'traceEvents': self._events, 'displayTimeUnit': 'ms'}

        with open(self.file_path, 'w') as f:
            json.dump(trace_data, f, indent=2, default=serialize_object)

    def _get_virtual_thread_for_task(self, span_id: str) -> int:
        """Allocate a virtual thread for a task span.

        Args:
            span_id: The span ID to allocate a virtual thread for

        Returns:
            The virtual thread ID to use
        """
        # Try to reuse a free virtual thread
        if self._free_virtual_threads:
            vtid = self._free_virtual_threads.pop(0)
        else:
            # Create a new virtual thread
            vtid = self._next_virtual_tid
            self._next_virtual_tid += 1

        return vtid

    def _release_virtual_thread(self, span_id: str):
        """Release a virtual thread when a task span ends."""
        if span_id in self._task_virtual_threads:
            vtid = self._task_virtual_threads[span_id]
            self._free_virtual_threads.append(vtid)

    def _convert_to_chrome_trace_event(self, span: Span) -> dict[str, Any] | None:
        """Convert a completed span to Chrome Trace Event format.

        Args:
            span: The completed span to convert

        Returns:
            Chrome trace event dictionary or None if span is not complete
        """
        if span.end_time_unix_nano is None:
            return None

        # Convert nanoseconds to microseconds for Chrome trace format
        ts = span.start_time_unix_nano // 1000
        dur = (span.end_time_unix_nano - span.start_time_unix_nano) // 1000

        # Get thread ID
        # - Use virtual thread for task spans
        # - Use virtual thread for children of task spans
        # - Use original thread_id for everything else
        thread_id = span.attributes.get('thread_id', 0)

        if span.span_id in self._task_virtual_threads:
            # This is a task span
            thread_id = self._task_virtual_threads[span.span_id]
        elif span.span_id in self._span_to_task:
            # This is a child of a task span
            task_span_id = self._span_to_task[span.span_id]
            thread_id = self._task_virtual_threads.get(task_span_id, thread_id)

        # Create complete event (X phase)
        event = {
            'name': span.name,
            'cat': span.attributes.get('type', 'unknown'),
            'ph': 'X',
            'ts': ts,
            'dur': dur,
            'pid': span.attributes.get('process_id', 1),
            'tid': thread_id,
        }

        # Add args
        args = {
            k: v
            for k, v in span.attributes.items()
            if k not in ('hostname', 'process_id', 'thread_id', 'task_id', 'type')
        }
        args['trace_id'] = span.trace_id
        args['span_id'] = span.span_id
        if span.parent_span_id:
            args['parent_span_id'] = span.parent_span_id
        event['args'] = args

        return event
