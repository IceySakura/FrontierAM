"""Single task runner for nexus framework.

Handles running a single task with:
- ExecutionContext setup (runtime provider, OpenAI clients)
- Agent factory function building
- Task input construction and execution
- Span processor management
- Cache management for deterministic replay
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import os
import signal
import sys
import traceback

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_type_hints

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, PrivateAttr, SerializerFunctionWrapHandler, field_validator, model_serializer
from typing_extensions import Self

from nexus.core import (
    AgentRunOutput,
    ExecutionContext,
    OpenAIConfig,
    Registry,
    WithAgentRunner,
    WorkflowState,
    agent_registry,
    get_agent_gateway_scaffold_policy,
    openai_client_registry,
    task_registry,
)
from nexus.dataset.item import TaskFilesSpec, resolve_task_files_input
from nexus.dataset.preparation import TaskFilesMaterialization, TaskFilesMaterializer
from nexus.dataset.task_files import task_files_from_spec
from nexus.extensions.openai_clients.cache_task_id import observe_client_init
from nexus.gateway.observers import TrajectoryObserver, create_trajectory_observers, trajectory_observer_context
from nexus.runner.hooks import RunHook, run_hook_registry
from nexus.runtime import runtime_provider_registry
from nexus.runtime.models import RemoteInvokeError
from nexus.runtime.trace_drain import TraceDrainErrorPolicy
from nexus.trace import processor_context
from nexus.trace.processors import SpanProcessor, span_processor_registry
from nexus.utils.cache import cache_context, get_runtime_stats, load_cache, save_cache
from nexus.utils.cache_task_id import get_cache_task_id, get_input_cache_task_id
from nexus.utils.common import escape_non_oc_interpolations, maybe_await
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)
ENV_CONFIG_SUFFIX = '__env'


class CacheConfig(BaseModel):
    """Configuration for cache and replay functionality.

    The cache system operates in two stages:
    1. REPLAY: Read from cache until all entries are exhausted
    2. CACHE: Write to cache after replay is complete

    Example YAML config:
        runner:
          cache:
            enabled: true
            file_path: /tmp/cache/trajectory.pkl
            ignore_args_hash: false
            overwrite_tags: ['tool']  # Re-execute tools but keep LLM cache
    """

    enabled: bool = True  # Whether caching is enabled (set False to temporarily disable)
    file_path: str | None = None  # Path to cache file (pickle format)
    ignore_args_hash: bool = False  # If True, match by function name only
    overwrite_tags: list[str] | None = None  # Tags to overwrite during replay


class HookConfig(BaseModel):
    """Configuration for a registered runner hook."""

    name: str
    required: bool = True
    args: dict[str, Any] = Field(default_factory=dict)


class RunnerConfig(BaseModel):
    """Configuration for runner behavior.

    Controls how the runner executes the task, not the task itself.

    Example YAML config:
        runner:
          timeout: 3600
          hold_after_task: 1800
          cache:
            file_path: /tmp/cache.pkl
    """

    timeout: float | None = None  # Task timeout in seconds, None means no timeout
    hold_after_task: float | None = None  # Seconds to hold sandbox alive after task, None means no hold
    hold_signal_handlers: bool = True  # Whether hold should install SIGINT/SIGTERM handlers
    early_output_path: str | None = None  # Optional RunOutput JSON path written before hold_after_task waits
    cache: CacheConfig | None = None  # Cache config, None means no caching
    use_gateway: bool = False  # Wrap main as a gateway client and inject gateway_trajectory
    hooks: dict[str, HookConfig] = Field(default_factory=dict)
    trace_drain_error_policy: TraceDrainErrorPolicy = 'warn'
    task_files_materialization: TaskFilesMaterialization = Field(default_factory=TaskFilesMaterialization)


class ModuleObjectConfig(BaseModel):
    """Configuration for a registered module object."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ProcessorConfig(BaseModel):
    """Configuration for a span processor."""

    name: str  # Registered processor name
    args: dict[str, Any] = {}  # Arguments passed to processor __init__


class _RunAnnotations(BaseModel):
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator('extra')
    @classmethod
    def _validate_extra(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        except (TypeError, ValueError) as exc:
            raise ValueError('extra must contain finite JSON values') from exc
        if len(encoded) > 2 * 1024 * 1024:
            raise ValueError('extra exceeds 2 MiB size limit')
        return value

    @model_serializer(mode='wrap')
    def _serialize_annotations(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = handler(self)
        if not self.extra:
            value.pop('extra', None)
        return value


class RunInput(_RunAnnotations):
    """Input for single task runner.

    All fields must be serializable for distributed scheduling.
    """

    # Module imports - loaded before task execution
    imports: list[str] = []

    # Task configuration
    task_name: str
    task_config: dict[str, Any] = {}

    # Tool configuration
    tool_configs: dict[str, Any] = {}  # Tool-specific configurations (e.g., API keys)

    # Dataset task files
    task_files: TaskFilesSpec | None = None

    # Run identification (host-unique identifier)
    run_id: str | None = None  # If None, will be auto-generated

    # Runtime provider configuration
    runtime_provider_name: str | None = None
    runtime_provider_config: dict[str, Any] = {}
    runtime_configs: dict[str, dict[str, Any]] = {}  # Per-runtime configs, e.g. {'main': {...}, 'eval': {...}}

    # Agent configuration
    agent_name: str | None = None  # None means task doesn't need agent
    agent_config: dict[str, Any] = {}

    # LLM configuration - supports multiple clients
    # Format: {'main': {'name': 'openai', 'config': {...}}, 'eval': {...}}
    openai_clients: dict[str, dict[str, Any]] = {}

    # Span processor configuration
    # Key is used to identify processor results in RunOutput.processor_results
    processors: dict[str, ProcessorConfig] = {}

    # Gateway Trajectory Tree observers. These are read-only sidecars and do
    # not contribute values to RunOutput.processor_results.
    gateway_trajectory_observers: dict[str, ModuleObjectConfig] = Field(default_factory=dict)

    # Runner behavior configuration
    runner: RunnerConfig = RunnerConfig()

    # Resumable workflow progress, keyed by stable execution scope
    workflow_states: dict[str, WorkflowState] = Field(default_factory=dict)

    # Context management: externally constructed CompactionManager instance
    # Not serializable – set programmatically by the runner before calling run().
    compaction_manager: Any = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        """Create RunInput from a dictionary with OmegaConf resolution.

        This method enables environment variable interpolation using OmegaConf syntax,
        e.g., ${oc.env:API_KEY} will be replaced with the value of the API_KEY env var.

        Non-OmegaConf interpolations (e.g., ${var} in JavaScript code) are automatically
        escaped to prevent parsing errors.

        Args:
            config: Configuration dictionary (may contain OmegaConf interpolations)

        Returns:
            RunInput instance with resolved configuration
        """
        config_copy: dict[str, Any] = copy.deepcopy(config)

        # Workflow outputs are opaque state, not runner configuration. Keep
        # interpolation-like strings inside them byte-for-byte unchanged.
        workflow_states = config_copy.pop('workflow_states', None)
        extra = config_copy.pop('extra', {})

        # Remove free-text fields before OmegaConf processing to avoid
        # interpolation errors from code patches / diffs / prompts that
        # contain ${...} patterns (including ${oc.env:...} which
        # escape_non_oc_interpolations intentionally preserves).
        saved_fields: list[tuple[str, str, Any]] = []  # (parent_key, field_key, value)

        # Extract free-text fields from any config section that may contain
        # arbitrary text with ${...} patterns.
        # NOTE(petersheng): prompt and reward hold dataset text (code, LaTeX, templates) for every
        # task, so a literal ${oc.env:...} in them is data and must not resolve to an env value.
        _free_text_fields = {
            'task_config': ('instance', 'user_prompt', 'system_prompt', 'prompt', 'reward'),
            'agent_config': ('user_prompt', 'system_prompt', 'prompt'),
        }
        for section_key, field_keys in _free_text_fields.items():
            section = config_copy.get(section_key)
            if isinstance(section, dict):
                for fk in field_keys:
                    if fk in section:
                        saved_fields.append((section_key, fk, section.pop(fk)))

        escaped_config = escape_non_oc_interpolations(config_copy)
        omega_config = OmegaConf.create(escaped_config)
        resolved = OmegaConf.to_container(omega_config, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError(f'Config must resolve to a dict, got: {type(resolved).__name__}')

        # Restore saved free-text fields after OmegaConf resolution
        for parent_key, field_key, value in saved_fields:
            resolved.setdefault(parent_key, {})[field_key] = value

        if workflow_states is not None:
            resolved['workflow_states'] = workflow_states
        resolved['extra'] = extra

        return cls.model_validate(resolved)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Create RunInput from a YAML/JSON configuration file.

        Loads the file and resolves OmegaConf interpolations (e.g., ${oc.env:API_KEY}).
        Non-OmegaConf interpolations (e.g., ${var} in JavaScript code) are preserved.

        Args:
            path: Path to the YAML or JSON configuration file

        Returns:
            RunInput instance with resolved configuration

        Raises:
            FileNotFoundError: If the config file does not exist
            TypeError: If the config file does not contain a mapping
        """
        import json

        import yaml

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')

        # Load as raw dict to avoid OmegaConf parsing invalid ${...} patterns
        with open(config_path) as f:
            config = json.load(f) if config_path.suffix == '.json' else yaml.safe_load(f)

        if not isinstance(config, dict):
            raise TypeError(f'Config file must contain a mapping, got: {type(config).__name__}')

        if 'task_files' in config:
            config['task_files'] = resolve_task_files_input(config['task_files'], base_dir=config_path.parent)

        return cls.from_dict(config)


class ErrorInfo(BaseModel):
    """Error information when task execution fails."""

    type: str  # Exception type name (e.g., "TimeoutError", "RuntimeError")
    message: str  # Error message
    traceback: str | None = None  # Full traceback
    _exception: BaseException | None = PrivateAttr(default=None)

    @property
    def exception(self) -> BaseException | None:
        """Return the original exception in this process, without serializing it."""
        return self._exception


def _resolve_env_suffix_config(value: Any, *, path: str) -> Any:
    """Resolve ``foo__env: ENV_NAME`` mappings without mutating the source config."""
    if isinstance(value, dict):
        resolved: dict[Any, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            child_path = f'{path}.{key_text}' if path else key_text
            if key_text.endswith(ENV_CONFIG_SUFFIX):
                target_key = key_text.removesuffix(ENV_CONFIG_SUFFIX)
                if not target_key:
                    raise ValueError(f'Invalid env config key {child_path!r}: target key is empty')
                target_path = f'{path}.{target_key}' if path else target_key
                if target_key in value:
                    raise ValueError(f'Config defines both {target_path!r} and {child_path!r}; remove one')
                if not isinstance(child, str) or not child.strip():
                    raise ValueError(
                        f'Env config key {child_path!r} must contain a non-empty environment variable name'
                    )
                env_name = child.strip()
                if env_name not in os.environ:
                    raise ValueError(f'Environment variable {env_name!r} required by {child_path!r} is not set')
                resolved[target_key] = os.environ[env_name]
                logger.info(
                    'Resolved env config key %s -> %s from environment variable %s',
                    child_path,
                    target_path,
                    env_name,
                )
                continue
            resolved[key] = _resolve_env_suffix_config(child, path=child_path)
        return resolved
    if isinstance(value, list):
        return [_resolve_env_suffix_config(item, path=f'{path}[{index}]') for index, item in enumerate(value)]
    return value


def _resolve_run_input_env_suffixes(run_input: RunInput) -> RunInput:
    """Create a runtime-only RunInput copy with ``__env`` mappings resolved."""
    config = run_input.model_dump(mode='python')
    workflow_states = config.pop('workflow_states')
    extra = config.pop('extra', {})
    resolved = _resolve_env_suffix_config(config, path='run_input')
    resolved['workflow_states'] = workflow_states
    resolved['extra'] = extra
    return RunInput.model_validate(resolved)


class RunOutput(_RunAnnotations):
    """Output from single task runner.

    Always returned, even when task execution fails. Check `success` field
    to determine if the task completed successfully.
    """

    task_output: Any = None  # May be None if task failed before producing output
    agent_output: Any | None = None  # AgentRunOutput from the agent, if run_agent was called
    judge_output: Any | None = None  # Judge status, when the task reports a grading result
    processor_results: dict[str, Any] = {}  # Results from processors, keyed by processor key
    hook_results: dict[str, Any] = Field(default_factory=dict)
    hook_errors: dict[str, str] = Field(default_factory=dict)  # Keys are "<instance>.<lifecycle method>"
    cache_stats: dict[str, Any] | None = None  # Cache statistics if caching was enabled
    input_cache_task_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    input_cache_task_ids: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)
    cache_task_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    cache_task_ids: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)
    workflow_states: dict[str, WorkflowState] = Field(default_factory=dict)

    success: bool = True  # False if task execution failed
    error: ErrorInfo | None = None  # Only set when success=False

    @model_serializer(mode='wrap')
    def _serialize_annotations(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        value = super()._serialize_annotations(handler)
        # Preserve existing output hashes for tasks that do not report a judge.
        if self.judge_output is None:
            value.pop('judge_output', None)
        return value


@dataclass(frozen=True)
class RunHookFailure:
    """Failure raised by an embedding hook during error reporting or teardown."""

    stage: Literal['initialization_error', 'run_output', 'context_closing']
    error: Exception


class RunHookError(RuntimeError):
    """One or more embedding hooks failed."""

    def __init__(self, failures: list[RunHookFailure]) -> None:
        self.failures = tuple(failures)
        details = '; '.join(f'{failure.stage}: {failure.error}' for failure in failures)
        super().__init__(details)


@dataclass
class RunHooks:
    """In-process hooks for embedding the single runner.

    Hooks are deliberately not part of ``RunInput`` because they are callbacks
    and runtime objects, not serializable scheduler configuration.

    An initialization-error callback opts into failed outputs for setup errors
    that would otherwise propagate to the caller.
    """

    task_input_overrides: dict[str, Any] = field(default_factory=dict)
    agent_input_overrides: dict[str, Any] = field(default_factory=dict)
    span_processors: list[SpanProcessor] = field(default_factory=list)
    on_context_ready: Callable[[ExecutionContext], Any | Awaitable[Any]] | None = None
    on_run_output: Callable[[ExecutionContext, RunOutput], Any | Awaitable[Any]] | None = None
    on_context_closing: Callable[[ExecutionContext], Any | Awaitable[Any]] | None = None
    on_initialization_error: Callable[[RunInput, Exception, RunOutput], Any | Awaitable[Any]] | None = None
    hook_failures: list[RunHookFailure] = field(default_factory=list, init=False)

    def raise_if_failed(self) -> None:
        if self.hook_failures:
            raise RunHookError(self.hook_failures) from self.hook_failures[0].error


async def _invoke_post_run_hook(
    hooks: RunHooks,
    stage: Literal['initialization_error', 'run_output', 'context_closing'],
    callback: Callable[..., Any],
    *args: Any,
) -> Exception | None:
    try:
        await maybe_await(callback(*args))
    except Exception as exc:
        hooks.hook_failures.append(RunHookFailure(stage=stage, error=exc))
        logger.exception('Run %s hook failed', stage)
        return exc
    return None


@asynccontextmanager
async def _runtime_hold_scope(runtime_provider: Any, hold_after_task: float | None) -> AsyncIterator[None]:
    """Restore runtime release before context cleanup on every exit path."""
    enabled = hold_after_task is not None and runtime_provider is not None
    if enabled:
        runtime_provider._suppress_release = True
    try:
        yield
    finally:
        if enabled:
            runtime_provider._suppress_release = False


@asynccontextmanager
async def _processor_scope(processors: list[SpanProcessor]) -> AsyncIterator[None]:
    """Register ``processors`` for the run, then await the teardown they cannot run.

    ``processor_context`` finalizes synchronously, so a processor whose teardown must
    await — draining publications, reading a store, releasing records — has nowhere to
    run it. The hook is optional and discovered by name, like ``dispatch_mode``.

    Every exit leaves through here, so a cancelled run still finalizes; the teardown is
    shielded because a caller that was cancelled mid-write must not strand its records.
    """
    try:
        with processor_context(processors, finalize_on_exit=True):
            yield
    finally:
        for processor in processors:
            afinalize = getattr(processor, 'afinalize', None)
            if afinalize is None:
                continue
            teardown = asyncio.ensure_future(afinalize())
            try:
                await asyncio.shield(teardown)
            except asyncio.CancelledError:
                await asyncio.shield(teardown)
                raise


@asynccontextmanager
async def _trajectory_observation_scope(
    ctx: ExecutionContext,
    hooks: RunHooks | None,
    trajectory_observers: tuple[TrajectoryObserver, ...],
) -> AsyncIterator[None]:
    observer_context = trajectory_observer_context(trajectory_observers) if trajectory_observers else nullcontext()
    with observer_context:
        try:
            yield
        finally:
            if hooks is not None and hooks.on_context_closing is not None:
                await _invoke_post_run_hook(hooks, 'context_closing', hooks.on_context_closing, ctx)


def _get_input_type(func: Any) -> type[BaseModel]:
    """Extract the input type from a function signature."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if not params:
        raise TypeError(f'Function {func} has no parameters')
    return get_type_hints(func).get(params[0].name, params[0].annotation)


def _build_input(input_type: type[BaseModel], available_kwargs: dict[str, Any]) -> Any:
    """Build an input instance from available kwargs.

    Only passes kwargs that are valid fields for the input type.
    """
    valid_fields = set(input_type.model_fields.keys())
    filtered_kwargs = {k: v for k, v in available_kwargs.items() if k in valid_fields}
    return input_type(**filtered_kwargs)


def _create_processors(processor_configs: dict[str, ProcessorConfig]) -> dict[str, SpanProcessor]:
    """Create processor instances from configurations.

    Args:
        processor_configs: Dict mapping keys to processor configurations

    Returns:
        Dict mapping keys to processor instances
    """
    processors: dict[str, SpanProcessor] = {}
    for key, config in processor_configs.items():
        processor_cls = span_processor_registry.get(config.name)
        processors[key] = processor_cls(**config.args)
        logger.debug(f'Created processor: {config.name} as {key}')
    return processors


def _create_runtime_provider(run_input: RunInput) -> Any | None:
    """Create runtime provider from configuration.

    Args:
        run_input: RunInput with runtime provider configuration

    Returns:
        Runtime provider instance or None if not configured
    """
    if not run_input.runtime_provider_name:
        return None

    provider_cls = runtime_provider_registry.get(run_input.runtime_provider_name)
    runtime_provider = provider_cls(**run_input.runtime_provider_config)

    # Pre-configure per-runtime params
    for runtime_name, runtime_config in run_input.runtime_configs.items():
        runtime_provider.configure(runtime_name, **runtime_config)

    logger.debug(f'Created runtime provider: {run_input.runtime_provider_name}')
    return runtime_provider


class AgentOutputCapture:
    """Container to capture agent output from run_agent calls."""

    def __init__(self):
        self.output: AgentRunOutput | None = None


def _build_run_agent_factory(
    run_input: RunInput, task_needs_agent: bool, agent_input_overrides: dict[str, Any] | None = None
) -> tuple[Any | None, AgentOutputCapture | None]:
    """Build run_agent factory function.

    Args:
        run_input: RunInput with agent configuration
        task_needs_agent: Whether the task requires an agent (WithAgentRunner)

    Returns:
        Tuple of (run_agent async function, AgentOutputCapture container)
        Returns (None, None) if task doesn't need agent
    """
    # If task doesn't need agent at all, return None
    if not task_needs_agent:
        return None, None

    # If task needs agent but no agent configured, return a placeholder that raises if called
    if not run_input.agent_name:
        capture = AgentOutputCapture()

        async def run_agent(**_task_kwargs: Any) -> AgentRunOutput:
            raise RuntimeError(
                'Task requires an agent (WithAgentRunner) but no agent_name was configured. '
                'Please specify agent_name in RunInput to use this task.'
            )

        return run_agent, capture

    # Normal case: agent is configured
    agent_fn = agent_registry.get(run_input.agent_name)
    agent_input_type = _get_input_type(agent_fn)
    capture = AgentOutputCapture()

    async def run_agent(**task_kwargs: Any) -> AgentRunOutput:
        """Run the agent with task-provided parameters.

        Agent accesses OpenAI client via ExecutionContext.
        Task provides: prompt and other task-specific params.
        """
        merged_kwargs = {
            **run_input.agent_config,
            **task_kwargs,
            **(agent_input_overrides or {}),
        }
        agent_input = _build_input(agent_input_type, merged_kwargs)
        logger.debug(f'Running agent: {run_input.agent_name}')
        result = await agent_fn(agent_input)
        # Capture the agent output for later retrieval
        capture.output = result
        return result

    run_agent.nexus_meta = getattr(agent_fn, 'nexus_meta', {})  # type: ignore[attr-defined]
    logger.debug(f'Created run_agent factory for agent: {run_input.agent_name}')
    return run_agent, capture


def _print_hold_info(runtime_provider: Any, timeout: float) -> None:
    """Print runtime info for hold mode."""
    pid = os.getpid()
    print('\n' + '═' * 54)
    print('  HOLD MODE — sandbox(es) kept alive for diagnosis')
    print('═' * 54)
    for name, runtime in runtime_provider._active_runtimes.items():
        base_url = getattr(runtime, 'base_url', 'unknown')
        print(f'  Runtime: {name}')
        print(f'    base_url:  {base_url}')
    print(f'\n  PID: {pid}')
    print(f'  Timeout: {timeout:.0f}s (auto-release)')
    print(f'  Exit: kill {pid}  OR  Ctrl+C')
    print('\n  Diagnose:')
    for _name, runtime in runtime_provider._active_runtimes.items():
        base_url = getattr(runtime, 'base_url', 'unknown')
        print(f'    nexus sandbox exec {base_url} "ls /workspace"')
    print('═' * 54 + '\n')
    sys.stdout.flush()


async def _hold_sandbox(runtime_provider: Any, timeout: float, *, install_signal_handlers: bool = True) -> None:
    """Hold sandbox alive: print info, wait for signal or timeout, then return."""
    _print_hold_info(runtime_provider, timeout)

    event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []

    if install_signal_handlers:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, event.set)
                installed_signals.append(sig)
            except (NotImplementedError, RuntimeError):
                logger.debug('Hold mode signal handlers are not available on this event loop', exc_info=True)
                break

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        logger.info('Hold interrupted by signal, releasing...')
    except asyncio.TimeoutError:
        logger.info('Hold timeout reached, releasing...')
    finally:
        for sig in installed_signals:
            with suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(sig)


def cache_task_id_for(*parts: str) -> str:
    """Derive a stable gateway cache-affinity ID from identity parts.

    Single source of truth for litellm ``cache_task_id`` derivation, shared by the
    composed run path here and the legacy Harbor generator. Requests whose parts all
    match land on the same gateway node; NUL-joining keeps distinct part tuples from
    colliding. Passing a single part reproduces the historical run_id-only hash.
    """
    return hashlib.md5('\0'.join(parts).encode(), usedforsecurity=False).hexdigest()


def _resolve_dynamic_client_args(
    run_input: RunInput,
    *,
    client_name: str,
    config_dict: dict[str, Any],
    run_id_provided: bool,
) -> dict[str, Any]:
    """Resolve per-run client args after Dataset Items and Run Parts have been composed."""
    client_args = dict(config_dict.get('client_args', {}))
    derive_cache_id = client_args.pop('cache_task_id_from_run_id', False)
    cache_task_id_by_run_id = client_args.pop('cache_task_id_by_run_id', None)
    derive_trace_id = client_args.pop('nexus_trace_id_from_run_id', False)
    if derive_cache_id is False and derive_trace_id is False and cache_task_id_by_run_id is None:
        return client_args
    if derive_cache_id is not True and derive_cache_id is not False:
        raise ValueError('cache_task_id_from_run_id must be a boolean')
    if derive_trace_id is not True and derive_trace_id is not False:
        raise ValueError('nexus_trace_id_from_run_id must be a boolean')
    if cache_task_id_by_run_id is not None and not isinstance(cache_task_id_by_run_id, dict):
        raise ValueError('cache_task_id_by_run_id must be a mapping from run_id to cache_task_id')
    cache_affinity_clients = {'litellm', 'openai_responses'}
    if (derive_cache_id or cache_task_id_by_run_id is not None) and client_name not in cache_affinity_clients:
        supported = ', '.join(sorted(cache_affinity_clients))
        raise ValueError(f'run-id-derived gateway cache affinity is supported only by: {supported}')
    if derive_trace_id and client_name != 'litellm':
        raise ValueError('run-id-derived Nexus trace routing is supported only by the litellm client')
    if not run_input.run_id:
        raise ValueError('run-id-derived gateway affinity requires a non-empty RunInput.run_id')
    if cache_task_id_by_run_id is not None:
        mapped_cache_task_id = cache_task_id_by_run_id.get(run_input.run_id)
        if not isinstance(mapped_cache_task_id, str) or not mapped_cache_task_id:
            raise ValueError(f'cache_task_id_by_run_id has no non-empty value for run_id {run_input.run_id!r}')
        client_args.setdefault('cache_task_id', mapped_cache_task_id)
    if not run_id_provided:
        logger.warning(
            'Deriving gateway affinity from the auto-generated run_id %s: requests inside this run '
            'still share one gateway node, but a later run of the same task will not. Set an '
            'explicit run_id (batch items and dataset items already do) for affinity that '
            'survives across runs.',
            run_input.run_id,
        )
    request_args = config_dict.get('request_args', {})
    model = str(request_args.get('model') or '')
    batch_id = str(run_input.runtime_provider_config.get('batch_id') or '')
    cache_affinity_id = (
        cache_task_id_for(batch_id, run_input.run_id, model) if batch_id else cache_task_id_for(run_input.run_id, model)
    )
    trace_affinity_id = cache_task_id_for(run_input.run_id, model)
    if derive_cache_id and not batch_id:
        logger.warning(
            'Deriving gateway cache affinity without runtime_provider_config.batch_id: requests inside '
            'this run remain sticky, but a later rollout of the same task can reuse the same affinity. '
            'Set a globally unique batch_id for fresh-trajectory isolation.'
        )
    if derive_cache_id:
        client_args.setdefault('cache_task_id', cache_affinity_id)
    if derive_trace_id:
        client_args.setdefault('nexus_trace_id', trace_affinity_id)
        client_args.setdefault('nexus_llm_lb_strategy', 'ConsistentHash')
    return client_args


def write_run_output(output_path: str | Path, result: RunOutput) -> None:
    """Write a run output JSON file atomically."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    tmp_path.write_text(result.model_dump_json(indent=2), encoding='utf-8')
    tmp_path.replace(path)


def _create_configured_hooks(
    run_input: RunInput,
    instances: dict[str, RunHook],
    failures: dict[str, Exception],
    *,
    stop_on_error: bool = True,
) -> None:
    """Build hooks once; failure recovery only constructs error reporters."""
    for key, config in run_input.runner.hooks.items():
        if key in instances or key in failures:
            continue
        try:
            hook_type = run_hook_registry.get(config.name)
            if not stop_on_error and hook_type.on_initialization_error is RunHook.on_initialization_error:
                continue
            args = _resolve_env_suffix_config(config.args, path=f'run_input.runner.hooks.{key}.args')
            hook = hook_type(**args)
            hook.required = config.required
            instances[key] = hook
        except Exception as exc:
            failures[key] = exc
            if stop_on_error:
                raise
            logger.exception('Run hook %s initialization failed', key)


async def _initialization_failure_output(
    run_input: RunInput, error: Exception, hooks: RunHooks | None = None
) -> RunOutput:
    error_info = ErrorInfo(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    error_info._exception = error
    logger.error('Initialization failed: %s: %s', error_info.type, error_info.message)
    output = RunOutput(
        extra=copy.deepcopy(run_input.extra),
        task_output=None,
        agent_output=None,
        processor_results={},
        cache_stats=None,
        workflow_states=copy.deepcopy(run_input.workflow_states),
        success=False,
        error=error_info,
    )
    if hooks is not None and hooks.on_initialization_error is not None:
        hook_error = await _invoke_post_run_hook(
            hooks, 'initialization_error', hooks.on_initialization_error, run_input, error, output
        )
        if hook_error is not None:
            output.hook_errors.setdefault(
                'runner.on_initialization_error', f'{type(hook_error).__name__}: {hook_error}'
            )
        output.success = False
        output.error = error_info
    return output


async def run(run_input: RunInput, hooks: RunHooks | None = None) -> RunOutput:
    """Run a single task.

    Orchestrates:
    0. Load required modules via imports
    1. ExecutionContext setup (runtime provider, OpenAI clients)
    2. run_agent factory function creation
    3. Task input construction
    4. Span processor setup
    5. Cache setup (if enabled)
    6. Task execution (within ExecutionContext, processor_context, and cache_context)
    """
    # Generate run_id if not provided
    run_id_provided = run_input.run_id is not None
    if run_input.run_id is None:
        import uuid

        run_input.run_id = uuid.uuid4().hex[:12]  # 12-character random hex string
        logger.debug(f'Auto-generated run_id: {run_input.run_id}')

    logger.debug(f'Starting runner for task: {run_input.task_name} (run_id: {run_input.run_id})')

    original_hooks = hooks
    configured_hooks: dict[str, RunHook] = {}
    hook_initialization_errors: dict[str, Exception] = {}
    try:
        # 0. Load required modules. This has to stay inside the try: a failing import
        # (missing dependency, typo in a Profile's imports) is an initialization error
        # like any other, and callers that batch many runs rely on getting a failed
        # RunOutput rather than an exception escaping run().
        for module_path in run_input.imports:
            logger.debug(f'Importing module: {module_path}')
            Registry.import_module(module_path)

        run_input = _resolve_run_input_env_suffixes(run_input)
        task_fn = task_registry.get(run_input.task_name)
        task_input_type = _get_input_type(task_fn)
        _create_configured_hooks(run_input, configured_hooks, hook_initialization_errors)
        if configured_hooks:
            from nexus.runner.hooks.composed import ComposedHook

            hooks = ComposedHook(configured_hooks, hooks=hooks)

        # 1. Setup RuntimeProvider (if configured)
        runtime_provider = _create_runtime_provider(run_input)
    except Exception as e:
        # Available hooks can report setup failures even when an import or another hook failed.
        _create_configured_hooks(run_input, configured_hooks, hook_initialization_errors, stop_on_error=False)
        if configured_hooks:
            from nexus.runner.hooks.composed import ComposedHook

            hooks = ComposedHook(configured_hooks, hooks=original_hooks)
        output = await _initialization_failure_output(run_input, e, hooks)
        output.hook_errors.update({
            f'{key}.initialize': f'{type(exc).__name__}: {exc}' for key, exc in hook_initialization_errors.items()
        })
        return output

    try:
        # Fill gateway trajectory ids, and honor runner.use_gateway, before clients exist.
        from nexus.runner.gateway_binding import apply_gateway_run_binding

        apply_gateway_run_binding(run_input)

        # 2. Setup OpenAI Clients
        openai_clients: dict[str, Any] = {}
        # if run_input.agent_name and issubclass(task_input_type, WithAgentRunner):
        for client_key, client_spec in run_input.openai_clients.items():
            # Support pre-built client instances (e.g., GatewayOpenAIClient built by the caller)
            prebuilt = client_spec.get('instance')
            if prebuilt is not None:
                openai_clients[client_key] = prebuilt
                openai_clients[client_key]._client_key = client_key
                logger.debug(f'Using pre-built OpenAI client "{client_key}": {type(prebuilt).__name__}')
                continue

            name = client_spec.get('name', 'openai')
            config_dict = client_spec.get('config', {})

            client_cls = openai_client_registry.get(name)
            client_args = _resolve_dynamic_client_args(
                run_input,
                client_name=name,
                config_dict=config_dict,
                run_id_provided=run_id_provided,
            )
            if name == 'gateway':
                configured_policy = client_args.get('scaffold_policy')
                if configured_policy is None or isinstance(configured_policy, str):
                    client_args['scaffold_policy'] = get_agent_gateway_scaffold_policy(
                        run_input.agent_name,
                        configured_policy,
                    )
            openai_config = OpenAIConfig(
                client_args=client_args,
                request_args=config_dict.get('request_args', {}),
                enforced_request_args=config_dict.get('enforced_request_args', {}),
            )
            openai_clients[client_key] = client_cls(openai_config)
            # Attach client_key to the client instance for processor filtering (e.g., TokenCounterProcessor)
            openai_clients[client_key]._client_key = client_key
            logger.debug(f'Created OpenAI client "{client_key}": {name}')

        # 3. Build run_agent factory function
        task_needs_agent = issubclass(task_input_type, WithAgentRunner)
        agent_input_overrides = hooks.agent_input_overrides if hooks is not None else None
        run_agent, agent_output_capture = _build_run_agent_factory(run_input, task_needs_agent, agent_input_overrides)

        # 4. Build Task Input
        task_kwargs = {**run_input.task_config}
        if run_agent is not None:
            task_kwargs['run_agent'] = run_agent
        if hooks is not None:
            task_kwargs.update(hooks.task_input_overrides)

        task_input = _build_input(task_input_type, task_kwargs)
        logger.debug(f'Built task input for task: {run_input.task_name}')

        # 5. Setup runtime sidecars
        processor_instances = _create_processors(run_input.processors)
        hytuner_processor = processor_instances.get('hytuner')
        if hytuner_processor is not None:
            for client in openai_clients.values():
                bind_observer = getattr(client, 'set_training_data_observer', None)
                if callable(bind_observer):
                    bind_observer(hytuner_processor)

        trajectory_observers = tuple(create_trajectory_observers(run_input.gateway_trajectory_observers).values())

        # 6. Setup Cache (if configured)
        cache_config = run_input.runner.cache
        cache_dict: dict[str, Any] | None = None
        cache_enabled = cache_config is not None and cache_config.enabled and cache_config.file_path is not None
        if cache_enabled:
            cache_dict = load_cache(cache_config.file_path)  # type: ignore[union-attr]
            logger.debug(f'Cache enabled with {len(cache_dict.get("entries", []))} existing entries')

        # 7. Execute Task within ExecutionContext, processor_context, and cache_context
        ctx = ExecutionContext(
            runtime_provider=runtime_provider,
            runtime_configs=run_input.runtime_configs,
            openai_clients=openai_clients,
            task_name=run_input.task_name,
            agent_name=run_input.agent_name,
            agent_config={**run_input.agent_config, **(agent_input_overrides or {})},
            task_config=run_input.task_config,
            tool_configs=run_input.tool_configs,
            task_files=task_files_from_spec(run_input.task_files, run_id=run_input.run_id),
            run_id=run_input.run_id,
            compaction_manager=run_input.compaction_manager,
            workflow_states=copy.deepcopy(run_input.workflow_states),
            task_files_materialization=run_input.runner.task_files_materialization,
        )
    except Exception as e:
        if hooks is None or hooks.on_initialization_error is None:
            raise
        return await _initialization_failure_output(run_input, e, hooks)

    # Initialize result variables
    task_output = None
    success = True
    error_info: ErrorInfo | None = None
    cache_stats: dict[str, Any] | None = None

    hold_after_task = run_input.runner.hold_after_task

    task_started = False
    try:
        async with (
            ctx,
            _trajectory_observation_scope(ctx, hooks, trajectory_observers),
            _runtime_hold_scope(runtime_provider, hold_after_task),
        ):
            # Use context-local sidecars so concurrent runs do not observe one another.
            hook_processors = hooks.span_processors if hooks is not None else []
            async with _processor_scope([*processor_instances.values(), *hook_processors]):
                if hooks is not None and hooks.on_context_ready is not None:
                    await maybe_await(hooks.on_context_ready(ctx))

                input_cache_task_ids = {
                    key: cache_task_id
                    for key, client in openai_clients.items()
                    if (cache_task_id := get_input_cache_task_id(client)) is not None
                }
                for client in openai_clients.values():
                    observe_client_init(client)

                # Use cache_context if caching is enabled
                cache_cm = (
                    cache_context(
                        cache_dict,
                        ignore_args_hash=cache_config.ignore_args_hash if cache_config else False,
                        overwrite_tags=cache_config.overwrite_tags if cache_config else None,
                    )
                    if cache_enabled
                    else nullcontext()
                )

                with cache_cm as cache_data:
                    logger.debug(f'Executing task: {run_input.task_name}')
                    try:
                        async with async_timeout(run_input.runner.timeout) as timeout_handle:
                            ctx.timeout_handle = timeout_handle
                            policy = run_input.runner.task_files_materialization
                            if policy.mode == 'cwd' and policy.target == 'host':
                                if ctx.task_files is None:
                                    raise ValueError('files_to_cwd requires task files')
                                ctx._task_files_lease = await asyncio.wait_for(
                                    TaskFilesMaterializer().prepare(ctx.task_files, ctx.task_files.spec, Path.cwd()),
                                    timeout=policy.timeout,
                                )
                            task_started = True
                            task_output = await task_fn(task_input)
                        logger.info(f'Task completed: {run_input.task_name}')
                    except (asyncio.TimeoutError, TimeoutError) as e:
                        if not task_started and hooks is not None and hooks.on_initialization_error is not None:
                            raise
                        success = False
                        error_info = ErrorInfo(
                            type='TimeoutError',
                            message=f'Task execution timed out after {run_input.runner.timeout} seconds',
                            traceback=None,
                        )
                        error_info._exception = e
                        logger.error(f'Task timed out: {run_input.task_name} after {run_input.runner.timeout}s')
                    except Exception as e:
                        if not task_started and hooks is not None and hooks.on_initialization_error is not None:
                            raise
                        success = False
                        error_traceback = traceback.format_exc()
                        if isinstance(e, RemoteInvokeError) and e.remote_traceback:
                            error_traceback = f'{error_traceback}\nRemote traceback:\n{e.remote_traceback}'
                        error_info = ErrorInfo(
                            type=type(e).__name__,
                            message=str(e),
                            traceback=error_traceback,
                        )
                        error_info._exception = e
                        logger.error(f'Task failed: {run_input.task_name} - {error_info.type}: {error_info.message}')
                        if isinstance(e, RemoteInvokeError) and e.remote_traceback:
                            logger.error('Remote task traceback:\n%s', e.remote_traceback.rstrip())
                    finally:
                        # Save cache if enabled (even on failure, partial cache may be useful)
                        if cache_enabled and cache_config and cache_data:
                            save_cache(cache_data, cache_config.file_path)  # type: ignore[arg-type]
                            from nexus.utils.cache import get_cache_stats

                            runtime_stats = get_runtime_stats()
                            cache_stats = get_cache_stats(cache_data)
                            cache_stats['replay_hits'] = runtime_stats['replay_hits']
                            cache_stats['new_cached'] = runtime_stats['new_cached']

            # Collect processor results after processor_context has finalized
            # processors, but before hold keeps the runtime alive for diagnosis.
            processor_results = {key: processor.get_result() for key, processor in processor_instances.items()}
            agent_output = agent_output_capture.output if agent_output_capture else None
            cache_task_ids = {
                key: cache_task_id
                for key, client in openai_clients.items()
                if (cache_task_id := get_cache_task_id(client)) is not None
            }
            run_output = RunOutput(
                extra=copy.deepcopy(run_input.extra),
                task_output=task_output,
                agent_output=agent_output,
                processor_results=processor_results,
                cache_stats=cache_stats,
                input_cache_task_id=next(iter(input_cache_task_ids.values()))
                if len(input_cache_task_ids) == 1
                else None,
                input_cache_task_ids=input_cache_task_ids if len(input_cache_task_ids) > 1 else {},
                cache_task_id=next(iter(cache_task_ids.values())) if len(cache_task_ids) == 1 else None,
                cache_task_ids=cache_task_ids if len(cache_task_ids) > 1 else {},
                workflow_states=copy.deepcopy(ctx.workflow_states),
                success=success,
                error=error_info,
            )

            if hooks is not None and hooks.on_run_output is not None:
                await _invoke_post_run_hook(hooks, 'run_output', hooks.on_run_output, ctx, run_output)

            if hold_after_task is not None and run_input.runner.early_output_path:
                try:
                    write_run_output(run_input.runner.early_output_path, run_output)
                    logger.info('Early output saved before hold: %s', run_input.runner.early_output_path)
                except Exception:
                    logger.exception('Failed to save early output before hold: %s', run_input.runner.early_output_path)

            # Hold sandbox alive after task completes
            if hold_after_task is not None and runtime_provider is not None:
                await _hold_sandbox(
                    runtime_provider,
                    hold_after_task,
                    install_signal_handlers=run_input.runner.hold_signal_handlers,
                )
    except Exception as e:
        if task_started or hooks is None or hooks.on_initialization_error is None:
            raise
        return await _initialization_failure_output(run_input, e, hooks)

    return run_output


__all__ = [
    'CacheConfig',
    'ErrorInfo',
    'HookConfig',
    'ModuleObjectConfig',
    'ProcessorConfig',
    'RunInput',
    'RunHooks',
    'RunnerConfig',
    'RunOutput',
    'run',
    'write_run_output',
]
