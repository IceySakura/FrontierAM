"""Registered hooks for a single runner invocation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nexus.core.registry import Registry


if TYPE_CHECKING:
    from nexus.core.context import ExecutionContext
    from nexus.runner.single import RunInput, RunOutput


class RunHook:
    """One instance per run; ready, output and closing run inside ExecutionContext.

    Ready and output callbacks run in configuration order. Closing runs in reverse
    order for every hook whose ready callback was entered, including failed starts.
    Hooks must tolerate closing after partial initialization.
    """

    required: bool = True

    async def on_initialization_error(self, run_input: RunInput, error: Exception, output: RunOutput) -> None:
        """Override to receive setup errors as failed outputs, even before context creation."""

    async def on_context_ready(self, ctx: ExecutionContext) -> None:
        """Prepare before the task starts."""

    async def on_run_output(self, ctx: ExecutionContext, output: RunOutput) -> None:
        """Finalize task artifacts before output callbacks and sandbox hold."""

    async def on_context_closing(self, ctx: ExecutionContext) -> None:
        """Clean up before runtime release, also on cancellation."""

    def get_result(self) -> Any:
        """Return JSON-serializable data for RunOutput.hook_results."""
        return None


def validate_run_hook(obj: Any, _name: str, _meta: dict) -> None:
    if not isinstance(obj, type) or not issubclass(obj, RunHook):
        raise TypeError('Run hook must be a RunHook subclass')


run_hook_registry = Registry[type[RunHook]]('run_hook', validate_run_hook)
