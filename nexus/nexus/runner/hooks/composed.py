"""Compose configured hooks with the runner's existing callback interface."""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, Literal

from nexus.runner.hooks.base import RunHook
from nexus.runner.single import RunHooks
from nexus.utils.common import maybe_await


if TYPE_CHECKING:
    from nexus.core.context import ExecutionContext
    from nexus.runner.single import RunInput, RunOutput


logger = logging.getLogger(__name__)


class ComposedHook(RunHooks):
    """Per-run hook composition, including lifecycle state and result collection."""

    def __init__(self, instances: dict[str, RunHook], *, hooks: RunHooks | None = None) -> None:
        self._original = hooks or RunHooks()
        self._instances = instances
        self._entered: dict[str, RunHook] = {}
        self._initialization_hooks = {
            key: hook
            for key, hook in instances.items()
            if getattr(hook.on_initialization_error, '__func__', None) is not RunHook.on_initialization_error
        }
        self._reported_initialization: dict[str, RunHook] = {}
        self._errors: dict[str, str] = {}
        self._output: RunOutput | None = None
        super().__init__(
            task_input_overrides=self._original.task_input_overrides,
            agent_input_overrides=self._original.agent_input_overrides,
            span_processors=self._original.span_processors,
            on_context_ready=self._on_context_ready,
            on_run_output=self._on_run_output,
            on_context_closing=self._on_context_closing,
            on_initialization_error=self._on_initialization_error
            if self._initialization_hooks or self._original.on_initialization_error is not None
            else None,
        )
        # The caller must still observe failures through its original RunHooks.
        self.hook_failures = self._original.hook_failures

    async def _on_initialization_error(self, run_input: RunInput, error: Exception, output: RunOutput) -> None:
        self._output = output
        first_error: Exception | None = None
        for key, hook in self._initialization_hooks.items():
            self._reported_initialization[key] = hook
            try:
                await self._invoke(key, hook, 'on_initialization_error', run_input, error, output)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        try:
            self._collect_results(output)
        except Exception as exc:
            if first_error is None:
                first_error = exc
        if self._original.on_initialization_error is not None:
            try:
                await maybe_await(self._original.on_initialization_error(run_input, error, output))
            except Exception as exc:
                output.hook_errors['embedding.on_initialization_error'] = f'{type(exc).__name__}: {exc}'
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    async def _on_context_ready(self, ctx: ExecutionContext) -> None:
        if self._original.on_context_ready is not None:
            await maybe_await(self._original.on_context_ready(ctx))
        for key, hook in self._instances.items():
            self._entered[key] = hook
            await self._invoke(key, hook, 'on_context_ready', ctx)

    async def _on_run_output(self, ctx: ExecutionContext, output: RunOutput) -> None:
        self._output = output
        try:
            for key, hook in self._entered.items():
                await self._invoke(key, hook, 'on_run_output', ctx, output)
        except Exception:
            output.hook_errors = dict(self._errors)
            raise
        self._collect_results(output)
        if self._original.on_run_output is not None:
            await maybe_await(self._original.on_run_output(ctx, output))

    async def _on_context_closing(self, ctx: ExecutionContext) -> None:
        first_error: BaseException | None = None
        for key, hook in reversed(self._entered.items()):
            try:
                await self._invoke(key, hook, 'on_context_closing', ctx)
            except BaseException as exc:
                if first_error is None or not isinstance(exc, Exception):
                    first_error = exc
        if self._original.on_context_closing is not None:
            try:
                await maybe_await(self._original.on_context_closing(ctx))
            except BaseException as exc:
                self._errors['embedding.on_context_closing'] = f'{type(exc).__name__}: {exc}'
                if first_error is None or not isinstance(exc, Exception):
                    first_error = exc
        try:
            if self._output is not None:
                self._collect_results(self._output)
        except BaseException as exc:
            if first_error is None or not isinstance(exc, Exception):
                first_error = exc
        if first_error is not None:
            raise first_error

    async def _invoke(
        self,
        key: str,
        hook: RunHook,
        stage: Literal['on_initialization_error', 'on_context_ready', 'on_run_output', 'on_context_closing'],
        *args: Any,
    ) -> None:
        try:
            await getattr(hook, stage)(*args)
        except Exception as exc:
            self._errors[f'{key}.{stage}'] = f'{type(exc).__name__}: {exc}'
            logger.exception('Run hook %s.%s failed', key, stage)
            if hook.required:
                raise

    def _collect_results(self, output: RunOutput) -> None:
        first_error: Exception | None = None
        try:
            for key, hook in (self._entered | self._reported_initialization).items():
                try:
                    output.hook_results[key] = hook.get_result()
                except Exception as exc:
                    self._errors[f'{key}.get_result'] = f'{type(exc).__name__}: {exc}'
                    logger.exception('Run hook %s.get_result failed', key)
                    if hook.required and first_error is None:
                        first_error = exc
        finally:
            output.hook_errors.update(self._errors)
        if first_error is not None:
            raise first_error
