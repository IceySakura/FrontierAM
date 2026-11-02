"""Tests for nexus.trace.phases: flat, context-manager, and close-only usage."""

from __future__ import annotations

from nexus.trace import processor_context
from nexus.trace.phases import phase
from nexus.trace.processors.print import PrintSpanProcessor
from nexus.trace.tracer import _active_span_stack, observe
from nexus.trace.types import Status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stack_names() -> list[str]:
    """Return the names of all spans on the active stack (bottom-to-top)."""
    from nexus.trace.controller import SpanController

    stack = _active_span_stack.get() or []
    controller = SpanController.get_instance_sync()
    names: list[str] = []
    for sid in stack:
        try:
            names.append(controller.get_span(sid).name)
        except KeyError:
            names.append(f'<missing:{sid}>')
    return names


# ---------------------------------------------------------------------------
# Flat usage
# ---------------------------------------------------------------------------


class TestPhaseFlat:
    def test_open_and_close(self):
        """phase('x') opens; phase() closes."""
        with processor_context([]):
            phase('task.init')
            assert 'task.init' in _stack_names()

            phase()
            assert 'task.init' not in _stack_names()

    def test_sibling_replacement(self):
        """Opening a sibling closes the previous phase."""
        with processor_context([]):
            phase('task.init')
            phase('task.agent')
            names = _stack_names()
            assert 'task.init' not in names
            assert 'task.agent' in names
            phase()

    def test_child_nesting(self):
        """A child name keeps the parent open."""
        with processor_context([]):
            phase('task.init')
            phase('task.init.get_runtime')
            names = _stack_names()
            assert 'task.init' in names
            assert 'task.init.get_runtime' in names

            # Close child, parent remains
            phase('task.init.setup')
            names = _stack_names()
            assert 'task.init' in names
            assert 'task.init.get_runtime' not in names
            assert 'task.init.setup' in names
            phase()


# ---------------------------------------------------------------------------
# Context manager usage
# ---------------------------------------------------------------------------


class TestPhaseContextManager:
    def test_basic_with(self):
        """with phase('x') opens on enter, closes on exit."""
        with processor_context([]):
            phase('task.init')
            with phase('task.init.get_runtime'):
                assert 'task.init.get_runtime' in _stack_names()
            assert 'task.init.get_runtime' not in _stack_names()
            assert 'task.init' in _stack_names()
            phase()

    def test_exception_marks_error(self):
        """Phase span is marked ERROR when exception propagates through __exit__."""
        from nexus.trace.events import SpanEndEvent
        from nexus.trace.processors.base import SpanProcessor

        ended_spans: list[tuple[str, dict]] = []

        class _Collector(SpanProcessor):
            def on_end(self, span, event: SpanEndEvent) -> None:
                ended_spans.append((span.name, dict(span.status) if span.status else {}))

        with processor_context([_Collector()]):
            phase('task.init')
            try:
                with phase('task.init.setup'):
                    raise ValueError('test error')
            except ValueError:
                pass

            # The setup span should have been ended with ERROR status
            setup_ends = [(n, s) for n, s in ended_spans if n == 'task.init.setup']
            assert len(setup_ends) == 1
            assert setup_ends[0][1].get('code') == 'ERROR'
            phase()


# ---------------------------------------------------------------------------
# Close-only
# ---------------------------------------------------------------------------


class TestPhaseCloseOnly:
    def test_close_all_phases(self):
        """phase() with no args closes all task_phase spans on the stack."""
        with processor_context([]):
            phase('task.init')
            phase('task.init.get_runtime')
            phase()
            # All task_phase spans should be closed
            names = _stack_names()
            assert 'task.init' not in names
            assert 'task.init.get_runtime' not in names


# ---------------------------------------------------------------------------
# Interaction with @observe
# ---------------------------------------------------------------------------


class TestPhaseWithObserve:
    async def test_observe_cleans_up_orphaned_phases(self):
        """@observe safety net closes phase spans left open when the function exits."""

        @observe(name='task.test_task', attributes={'type': 'task', 'name': 'test_task'})
        async def fake_task():
            phase('task.init')
            phase('task.agent')
            # Don't close — @observe should clean up

        with processor_context([]):
            await fake_task()
            # After fake_task returns, the @observe cleanup should have closed all children
            names = _stack_names()
            assert 'task.init' not in names
            assert 'task.agent' not in names
