from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
import socket
import threading

from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from logging import getLogger
from typing import Any, cast

from nexus.trace.context import context_processors as _context_processors
from nexus.trace.controller import SpanController
from nexus.trace.events import SpanEndEvent, SpanStartEvent, SpanUpdateEvent
from nexus.trace.processors import SpanProcessor
from nexus.trace.scopes import DEFAULT_SCOPE, validate_processor_scopes
from nexus.trace.types import Event, Span, Status, run_attribution_attributes
from nexus.utils.common import random_urlsafe_id, serialize_object


logger = getLogger(__name__)


_project_id: str | None = None

# Context variable to store the stack of active span IDs for the current context
# This ensures proper isolation between threads and async tasks
_active_span_stack: ContextVar[list[str] | None] = ContextVar('active_span_stack', default=None)
_span_attribute_context: ContextVar[dict[str, Any] | None] = ContextVar('span_attribute_context', default=None)
_inherit_task_trace: ContextVar[bool] = ContextVar('inherit_task_trace', default=False)


def set_project_id(project_id: str):
    global _project_id
    _project_id = project_id


def get_current_span_id() -> str | None:
    """Get the current active span ID from the context stack."""
    stack = _active_span_stack.get()
    return stack[-1] if stack else None


def get_current_span() -> Span | None:
    """Get the current active span from the context stack.

    Returns:
        The current Span object, or None if no span is active
    """
    span_id = get_current_span_id()
    if not span_id:
        return None

    try:
        span_controller: SpanController = SpanController.get_instance_sync()
        return span_controller.get_span(span_id)
    except KeyError:
        return None


def _push_span_id(span_id: str) -> None:
    """Push a span ID onto the context stack."""
    stack = _active_span_stack.get()
    if stack is None:
        stack = []
    stack = stack.copy()
    stack.append(span_id)
    _active_span_stack.set(stack)


def _pop_span_id() -> str | None:
    """Pop a span ID from the context stack."""
    stack = _active_span_stack.get()
    if not stack:
        return None
    stack = stack.copy()
    span_id = stack.pop()
    _active_span_stack.set(stack)
    return span_id


@contextmanager
def trace_attributes_context(attributes: dict[str, Any]):
    """Attach default attributes to spans created inside this context."""
    current = _span_attribute_context.get() or {}
    token = _span_attribute_context.set({**current, **attributes})
    try:
        yield
    finally:
        _span_attribute_context.reset(token)


@contextmanager
def run_trace_context(run_id: str, run_span_id: str, run_kind: str):
    """Attribute normal trace spans to a runner lifecycle span."""
    with trace_attributes_context(
        run_attribution_attributes(run_id=run_id, run_span_id=run_span_id, run_kind=run_kind)
    ):
        yield


@contextmanager
def run_span_parent_context(run_span_id: str):
    """Make one active run span the parent of the task started in this context."""
    span_controller: SpanController = SpanController.get_instance_sync()
    span_controller.get_span(run_span_id)
    mirror_activated = span_controller.otel_bridge.try_activate_mirror(run_span_id)
    stack = list(_active_span_stack.get() or [])
    stack_token = _active_span_stack.set([*stack, run_span_id])
    inherit_token = _inherit_task_trace.set(True)
    try:
        yield
    finally:
        _inherit_task_trace.reset(inherit_token)
        _active_span_stack.reset(stack_token)
        if mirror_activated:
            span_controller.otel_bridge.try_deactivate_mirror(run_span_id)


@contextmanager
def with_remote_trace_context(
    trace_id: str,
    parent_span_id: str,
    *,
    run_id: str | None = None,
    run_span_id: str | None = None,
    run_kind: str | None = None,
    otel_carrier: dict[str, str] | None = None,
):
    """Set remote trace context by injecting a placeholder span into the stack.

    This is used by the runtime server to set the parent trace context
    when executing requests from clients. A placeholder span is created
    and pushed onto the stack, so all spans created within this context
    will naturally inherit it as their parent.

    Args:
        trace_id: The trace ID from the client
        parent_span_id: The parent span ID from the client

    Example:
        with with_remote_trace_context(trace_id, parent_span_id):
            # All spans created here will have parent_span_id as parent
            with span_context('server_operation') as span:
                do_work()
    """
    from nexus.utils.otel.propagation import use_remote_context

    with use_remote_context(otel_carrier, fallback_only=True):
        span_controller: SpanController = SpanController.get_instance_sync()

        attributes = run_attribution_attributes(
            run_id=run_id,
            run_span_id=run_span_id,
            run_kind=run_kind,
        )

        # Create a placeholder span to represent the remote parent
        placeholder_span = Span(
            trace_id=trace_id,
            span_id=parent_span_id,
            parent_span_id='',
            name='_remote_parent',
            start_time_unix_nano=0,
            attributes=attributes,
        )

        # Register in controller and push to stack
        span_controller.active_span_dict[parent_span_id] = placeholder_span
        _push_span_id(parent_span_id)
        inherit_token = _inherit_task_trace.set(_inherit_task_trace.get() or run_span_id is not None)

        try:
            with trace_attributes_context(attributes):
                yield
        finally:
            _inherit_task_trace.reset(inherit_token)
            # Clean up: pop from stack and remove from controller
            popped_span_id = _pop_span_id()
            if popped_span_id != parent_span_id:
                logger.warning(
                    f'Span stack mismatch: expected to pop {parent_span_id}, but got {popped_span_id}. '
                    'This may indicate unbalanced span push/pop operations.'
                )
            span_controller.active_span_dict.pop(parent_span_id, None)


class SpanStub:
    def __init__(self, span_id: str):
        self.span_id = span_id
        self._span: Span | None = None

    @property
    def span(self) -> Span | None:
        if self._span is not None:
            return self._span
        try:
            sc: SpanController = SpanController.get_instance_sync()
            self._span = sc.get_span(self.span_id)
            return self._span
        except KeyError:
            return None

    def update(self, attributes: dict | None = None, events: list[Event] | None = None, status: Status | None = None):
        update_span(self.span_id, attributes, events, status)

    def finish(self):
        # Cache the Span reference before end_span removes it from active_span_dict.
        # end_span sets end_time_unix_nano on the same object, so the cached
        # reference will reflect the final timestamps after exit.
        if self._span is None:
            try:
                sc: SpanController = SpanController.get_instance_sync()
                self._span = sc.get_span(self.span_id)
            except KeyError:
                pass
        end_span(self.span_id)


def start_span(
    name: str,
    attributes: dict | None = None,
    events: list[Event] | None = None,
    status: Status | None = None,
) -> SpanStub:
    """Start a new span.

    Args:
        name: The name of the span
        attributes: Optional attributes for the span
        events: Optional initial events for the span
        status: Optional initial status for the span

    If attributes['type'] == task, a new trace is started.

    Returns:
        The span ID of the newly created span
    """
    span_controller: SpanController = SpanController.get_instance_sync()

    start_args = {}
    start_args['name'] = name
    parent_span_id = get_current_span_id() or ''
    start_args['parent_span_id'] = parent_span_id

    if parent_span_id:
        try:
            parent_span = span_controller.get_span(parent_span_id)
            trace_id = parent_span.trace_id
            start_args['trace_id'] = trace_id
        except KeyError:
            # Parent span no longer exists in active_span_dict.
            # This can happen in distributed tracing scenarios where:
            # 1. ContextVar inherited a span_id from a parent async task
            # 2. But the span has already ended or was never registered locally
            # In this case, start a new trace instead of failing.
            # TODO: fix context var consistency
            logger.debug(f'Parent span {parent_span_id} not found, starting new trace')
            start_args['parent_span_id'] = ''

    context_attributes = _span_attribute_context.get() or {}
    attributes = dict(context_attributes) if attributes is None else {**context_attributes, **attributes}
    attributes['hostname'] = socket.gethostname()
    attributes['process_id'] = os.getpid()
    attributes['thread_id'] = threading.get_ident()

    # Get current task ID if in async context
    try:
        task = asyncio.current_task()
        if task:
            attributes['task_id'] = id(task)
    except RuntimeError:
        # Not in async context
        pass

    if _project_id is not None:
        attributes['project_id'] = _project_id

    if attributes.get('type') == 'task' and not _inherit_task_trace.get():
        start_args['trace_id'] = random_urlsafe_id(16)
        start_args['parent_span_id'] = ''

    start_args['attributes'] = attributes
    start_event = SpanStartEvent(**start_args)

    if events is not None:
        start_event.events = events
    if status is not None:
        start_event.status = status

    span_controller.add_event(start_event)

    # Make the OTel mirror current before exposing the Nexus span as current.
    span_controller.otel_bridge.try_activate_mirror(start_event.span_id)

    # Push the new span ID onto the context stack
    _push_span_id(start_event.span_id)

    return SpanStub(start_event.span_id)


@contextmanager
def span_context(
    name: str,
    attributes: dict | None = None,
    events: list[Event] | None = None,
    status: Status | None = None,
):
    span_stub = start_span(name, attributes, events, status)
    try:
        yield span_stub
        span = span_stub.span
        if span is not None and span.status.code == 'UNSET':
            span_stub.update(status=Status(code='OK'))
    except (Exception, asyncio.CancelledError) as e:
        span_stub.update(status=Status(code='ERROR', message=str(e)))
        raise
    finally:
        span_stub.finish()


def _close_children_to(target_span_id: str, *, error: BaseException | None = None) -> None:
    """Close all spans on ``_active_span_stack`` above *target_span_id*.

    Walks from the stack top downward.  For each span that is **not**
    *target_span_id*: marks it OK (or ERROR if *error* is set) and calls
    :func:`end_span`.  Stops as soon as *target_span_id* is reached or the
    stack is exhausted.  *target_span_id* itself is **not** closed.
    """
    stack = _active_span_stack.get()
    if not stack:
        return

    while stack and stack[-1] != target_span_id:
        child_id = stack[-1]
        try:
            if error is not None:
                update_span(child_id, status=Status(code='ERROR', message=str(error)))
            else:
                update_span(child_id, status=Status(code='OK'))
        except Exception:
            logger.debug('Failed to update status of orphaned span %s', child_id, exc_info=True)
        end_span(child_id)
        # Re-read stack after end_span (which pops)
        stack = _active_span_stack.get() or []


def observe(
    name: str | None = None,
    attributes: dict | None = None,
    events: list[Event] | None = None,
    status: Status | None = None,
    ignore_inputs: bool = False,
    self_attrs: list[str] | None = None,
):
    def decorator(func):
        nonlocal name
        if name is None:
            name = func.__name__
        # Ensure type checker knows name is str
        span_name = cast(str, name)

        def prepare_inputs(*args, **kwargs):
            enhanced_attributes = attributes.copy() if attributes else {}

            # Extract specified attributes from self (first positional arg)
            if self_attrs and args:
                self_obj = args[0]
                for attr in self_attrs:
                    val = getattr(self_obj, attr, None)
                    if val is not None:
                        enhanced_attributes[attr] = val

            if ignore_inputs:
                return enhanced_attributes

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = dict(bound.arguments)

            params = list(sig.parameters.keys())
            if params and params[0] == 'self' and 'self' in inputs:
                # Remove self parameter for class methods
                inputs.pop('self')

            enhanced_attributes['inputs'] = serialize_object(inputs)
            return enhanced_attributes

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                enhanced_attributes = prepare_inputs(*args, **kwargs)
                span_stub = start_span(span_name, enhanced_attributes, events, status)
                caught_error: BaseException | None = None
                try:
                    result = await func(*args, **kwargs)
                    if result is not None:
                        span_stub.update(attributes={'outputs': serialize_object(result)})
                    span_stub.update(status=Status(code='OK'))
                    return result
                except (Exception, asyncio.CancelledError) as e:
                    caught_error = e
                    span_stub.update(status=Status(code='ERROR', message=str(e)))
                    raise
                finally:
                    _close_children_to(span_stub.span_id, error=caught_error)
                    span_stub.finish()

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            enhanced_attributes = prepare_inputs(*args, **kwargs)
            span_stub = start_span(span_name, enhanced_attributes, events, status)
            caught_error: BaseException | None = None
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    span_stub.update(attributes={'outputs': serialize_object(result)})
                span_stub.update(status=Status(code='OK'))
                return result
            except (Exception, asyncio.CancelledError) as e:
                caught_error = e
                span_stub.update(status=Status(code='ERROR', message=str(e)))
                raise
            finally:
                _close_children_to(span_stub.span_id, error=caught_error)
                span_stub.finish()

        return wrapper

    return decorator


def update_span(
    span_id: str,
    attributes: dict | None = None,
    events: list[Event] | None = None,
    status: Status | None = None,
) -> str:
    span_controller: SpanController = SpanController.get_instance_sync()
    update_event = SpanUpdateEvent(
        span_id=span_id,
        attributes=attributes,
        events=events,
        status=status,
    )
    span_controller.add_event(update_event)
    return span_id


def end_span(span_id: str) -> str:
    """End a span.

    Args:
        span_id: The ID of the span to end

    Returns:
        The span ID that was ended
    """
    span_controller: SpanController = SpanController.get_instance_sync()
    current_span_id = get_current_span_id()
    mirror_deactivated = False
    if current_span_id == span_id:
        mirror_deactivated = span_controller.otel_bridge.try_deactivate_mirror(span_id)

    end_event = SpanEndEvent(span_id=span_id)
    try:
        span_controller.add_event(end_event)
    except BaseException:
        if mirror_deactivated:
            span_controller.otel_bridge.try_activate_mirror(span_id)
        raise

    # Remove the span from the Nexus context stack only after END succeeds.
    if current_span_id == span_id:
        _pop_span_id()

    return end_event.span_id


def add_span_processor(processor: SpanProcessor):
    span_controller: SpanController = SpanController.get_instance_sync()
    span_controller.add_processor(processor)


def remove_span_processor(processor: SpanProcessor):
    span_controller: SpanController = SpanController.get_instance_sync()
    span_controller.remove_processor(processor)


@contextmanager
def processor_context(processors: list[SpanProcessor], finalize_on_exit: bool = True):
    """Context manager to temporarily register span processors for the current context.

    This allows you to add processors that are only active within a specific context,
    without affecting the global processor list or other concurrent contexts.
    The processors will be automatically removed when exiting the context.

    This implementation uses ContextVar to ensure thread and coroutine isolation,
    so processors registered in one thread/coroutine won't affect others.

    Example:
        with processor_context([JsonLineSpanProcessor("trace.jsonl")]) as ctx_processors:
            # Any spans created here will be processed by the JsonLineSpanProcessor
            with span_context("my_operation") as span:
                # do something
                pass
        # JsonLineSpanProcessor is no longer active here, finalize() has been called

    Args:
        processors: List of SpanProcessor instances to register for this context
        finalize_on_exit: If True, call finalize() on each processor when exiting.
            Set to False if you want to manage finalization manually. Note: when
            False, the background dispatcher queue is NOT drained on exit, so call
            ``SpanController.get_instance_sync().background_dispatcher.barrier()`` before your
            manual finalize() to ensure all queued hooks have run.

    Yields:
        The list of processors that were registered
    """
    if not processors:
        yield []
        return

    for processor in processors:
        validate_processor_scopes(getattr(processor, 'processor_scopes', frozenset({DEFAULT_SCOPE})))

    # Get current context processors
    current_processors = _context_processors.get() or []

    # Create new list with added processors
    new_processors = current_processors + processors

    # Set the new processor list in context
    token = _context_processors.set(new_processors)

    try:
        yield processors
    finally:
        # Restore the previous context
        _context_processors.reset(token)

        # Finalize processors if requested
        if finalize_on_exit:
            # Drain queued hooks so no tail events are lost before finalize().
            # The barrier drains the entire FIFO queue; since the context token
            # was already reset above, no further events for these processors can
            # be enqueued, so a global drain guarantees all of this context's
            # hooks have run before the barrier returns.
            span_controller: SpanController = SpanController.get_instance_sync()
            if not span_controller.background_dispatcher.barrier(timeout=5.0):
                logger.warning('processor_context: dispatcher barrier timed out; finalizing anyway')

            for processor in processors:
                try:
                    processor.finalize()
                except Exception as e:
                    logger.error(f'Error finalizing processor {processor}: {e}')


def replay_span_events(events: Iterable[dict[str, Any]], processors: list[SpanProcessor] | None = None) -> None:
    """Replay span events from an iterator.

    This function takes an iterator of span event dictionaries and replays them through
    the specified processors. This is useful for analyzing recorded traces,
    debugging, or reprocessing span data from various sources (database, memory, etc).

    Args:
        events: Iterator of span event dictionaries. Each dict should have:
               - 'span_id': span identifier
               - 'trace_id': trace identifier (for START events)
               - 'parent_span_id': parent span id (for START events)
               - 'event_type': 'start', 'update', or 'end' (lowercase)
               - 'event_data': JSON string or dict containing event details
               - 'time_unix_nano': timestamp in nanoseconds
        processors: List of processors to handle the replayed events.
                   If None, uses the currently registered processors.

    Raises:
        ValueError: If events contain invalid data
    """
    span_controller: SpanController = SpanController.get_instance_sync()

    original_processors = None
    if processors is not None:
        original_processors = span_controller.processors.copy()
        span_controller.processors.clear()
        for processor in processors:
            span_controller.add_processor(processor)

    try:
        for event in events:
            try:
                # Parse event_data if it's a string
                event_data_str = event.get('event_data', '{}')
                event_data = json.loads(event_data_str) if isinstance(event_data_str, str) else event_data_str

                # Convert database event format to SpanEvent format
                event_type = event.get('event_type', '').upper()
                span_id = event.get('span_id')
                time_unix_nano = event.get('time_unix_nano')

                if not span_id or time_unix_nano is None:
                    continue

                # Build the appropriate event type
                if event_type == 'START':
                    span_event = SpanStartEvent(
                        span_id=span_id,
                        time_unix_nano=time_unix_nano,
                        parent_span_id=event.get('parent_span_id', ''),
                        trace_id=event.get('trace_id', ''),
                        name=event_data.get('name', 'Unknown'),
                        attributes=event_data.get('attributes', {}),
                        events=event_data.get('events', []),
                        status=event_data.get('status', {}),
                    )
                elif event_type == 'UPDATE':
                    span_event = SpanUpdateEvent(
                        span_id=span_id,
                        time_unix_nano=time_unix_nano,
                        attributes=event_data.get('attributes'),
                        events=event_data.get('events'),
                        status=event_data.get('status'),
                    )
                elif event_type == 'END':
                    span_event = SpanEndEvent(
                        span_id=span_id,
                        time_unix_nano=time_unix_nano,
                    )
                else:
                    continue

                span_controller.add_event(span_event, origin='replay')

            except Exception as e:
                logger.debug(f'Error replaying span event: {e}')
                continue

    finally:
        # Drain so replayed events are fully processed before we return /
        # restore the original processors (preserves synchronous semantics).
        if not span_controller.background_dispatcher.barrier(timeout=5.0):
            logger.warning('replay_span_events: dispatcher barrier timed out; restoring processors anyway')
        if original_processors is not None:
            span_controller.processors.clear()
            span_controller.processors.extend(original_processors)
