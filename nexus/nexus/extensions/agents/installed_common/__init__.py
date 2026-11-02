"""Shared models and helpers for installed agents."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import shlex
import shutil
import socket

from collections.abc import Callable
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal

from nexus.core.agent import AgentRunInput, AgentRunOutput, ContentPartParam, WithUserPrompt, WithWorkingDir
from nexus.utils.paths import AGENTS_DIR
from nexus.utils.platform import is_windows


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from nexus.extensions.agents.installed_common.native_bridge import (
        call_native_bridge_runtime_agent,
    )
    from nexus.trace import RemoteTraceContext


logger = logging.getLogger(__name__)

DEFAULT_WORKING_DIR = '/workspace'
DEFAULT_AGENT_INSTALL_ROOT = str(AGENTS_DIR)
DEFAULT_TIMEOUT = 3600.0
_CHANNEL_IDLE_TIMEOUT = 30.0
DEFAULT_RUNTIME_BIN_DIR = '/nix/runtime/nix-env/bin'
DEFAULT_NPM_REGISTRY = 'http://mirrors.tencent.com/npm/'
DEFAULT_PROXY_API_KEY = 'dummy-key-for-proxy'
DEFAULT_CLI_STREAM_READER_LIMIT = 16 * 1024 * 1024


def splice_toml_root_keys(config_lines: list[str], extra_config: str | None) -> list[str]:
    """Port aca1dc8b's TOML scope fix: extra root keys precede generated tables.

    Appending model_reasoning_effort after [model_providers.nexus] silently
    scopes it into that table; Codex never reads the intended root option.
    """
    extra = (extra_config or '').strip()
    if not extra:
        return list(config_lines)
    root: list[str] = []
    tables: list[str] = []
    sink = root
    for line in extra.splitlines():
        if line.lstrip().startswith('['):
            sink = tables
        sink.append(line)
    merged = list(config_lines)
    if root:
        header = next((i for i, line in enumerate(merged) if line.lstrip().startswith('[')), len(merged))
        merged[header:header] = [*root, '']
    if tables:
        merged.extend(['', *tables])
    return merged


class InstalledAgentInput(AgentRunInput, WithUserPrompt, WithWorkingDir):
    """Common input fields for installed agents."""

    timeout: float = DEFAULT_TIMEOUT
    """Execution timeout in seconds."""


class InstalledNodeCliInput(InstalledAgentInput):
    """Common input fields for installed agents installed through npm."""

    npm_registry: str | None = DEFAULT_NPM_REGISTRY
    """npm registry used for runtime installs."""


class InstalledAgentOutput(AgentRunOutput):
    """Common output fields returned by installed agents."""

    success: bool
    """Whether the agent finished successfully."""

    output: str
    """Primary stdout-style output captured from the agent."""

    stderr: str
    """Standard error output or runtime setup failure message."""

    exit_code: int
    """Process exit code reported by the runtime."""

    execution_time: float
    """Total execution time in seconds."""

    failure_kind: Literal['agent', 'infrastructure'] | None = None
    """Classify a failed run without forcing task layers to parse stderr."""

    failure_stage: str | None = None
    """Machine-readable stage such as install, proxy, pre_start, or process."""


def coerce_user_prompt(user_prompt: str | list[ContentPartParam]) -> str:
    """Normalize structured prompts into the plain string interface used by installed CLIs."""
    if isinstance(user_prompt, str):
        return user_prompt
    return json.dumps(user_prompt, ensure_ascii=False)


def find_free_port() -> int:
    """Find an available loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        sock.listen(1)
        return sock.getsockname()[1]


def find_runtime_binary(name: str) -> str | None:
    """Find a binary from PATH or the default runtime bin directory."""
    resolved = shutil.which(name)
    if resolved:
        return resolved
    fallback = Path(DEFAULT_RUNTIME_BIN_DIR) / name
    if fallback.exists():
        return str(fallback)
    return None


def find_nexus_runtime_binary(name: str) -> str | None:
    """Find a binary in the nexus runtime bin dir, falling back to PATH when it is missing."""
    runtime_binary = Path(DEFAULT_RUNTIME_BIN_DIR) / name
    if runtime_binary.is_file():
        return str(runtime_binary)
    return find_runtime_binary(name)


def resolve_npm_bin_entry(install_dir: str | PathLike[str], package_name: str, bin_name: str) -> Path | None:
    """Resolve the JS entry point for an npm bin command from the package.json ``bin`` field."""
    package_dir = Path(install_dir) / 'node_modules' / package_name
    package_json = package_dir / 'package.json'
    if not package_json.exists():
        return None
    try:
        data = json.loads(package_json.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    bin_field = data.get('bin')
    if isinstance(bin_field, str):
        return package_dir / bin_field
    if isinstance(bin_field, dict):
        entry = bin_field.get(bin_name)
        if entry:
            return package_dir / entry
    return None


def build_runtime_path(*prepend_paths: str | PathLike[str]) -> str:
    """Build a PATH value from *prepend_paths* and a sanitised inherited PATH.

    Nix runtime paths are stripped from the inherited ``PATH`` so that
    agent subprocesses see the user container's native tools.
    """
    from nexus.utils.shell import get_clean_sandbox_env

    parts = [str(path) for path in prepend_paths if path]
    parts.append(get_clean_sandbox_env().get('PATH', ''))
    return os.pathsep.join(part for part in parts if part)


def build_runtime_command(argv: list[str]) -> str:
    """Quote argv for run_command's PowerShell (Windows) or Bash (POSIX)."""
    if is_windows():
        return '& ' + ' '.join("'" + arg.replace("'", "''") + "'" for arg in argv)
    return shlex.join(argv)


def resolve_npm_registry(npm_registry: str | None) -> str:
    """Resolve the registry used for npm-based installed agents."""
    registry = (npm_registry or DEFAULT_NPM_REGISTRY).strip()
    return registry or DEFAULT_NPM_REGISTRY


PUBLIC_NPM_REGISTRY = 'https://registry.npmjs.org'


def registry_candidates(npm_registry: str | None) -> list[str]:
    """Registries to try when downloading a package tarball directly.

    The configured registry comes first, but a sandbox that reaches the public
    internet and not an internal mirror still has to resolve the package, so
    fall back to npmjs.org. (Measured on the Gym WAA image: the internal mirror
    refuses connections while registry.npmjs.org answers.)
    """
    candidates = [resolve_npm_registry(npm_registry).rstrip('/'), PUBLIC_NPM_REGISTRY]
    ordered: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return ordered


def npm_install_timeout() -> float:
    """Wall-clock budget for installing a CLI agent inside a sandbox."""
    return 1800.0


def build_npm_env(
    *,
    npm_registry: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a stable npm environment for runtime installs."""
    registry = resolve_npm_registry(npm_registry)
    env = {
        'NPM_CONFIG_REGISTRY': registry,
        'npm_config_registry': registry,
        'npm_config_audit': 'false',
        'npm_config_fund': 'false',
        'npm_config_update_notifier': 'false',
        'npm_config_yes': 'true',
        'PATH': build_runtime_path(DEFAULT_RUNTIME_BIN_DIR),
        'NO_COLOR': '1',
        'CI': '1',
    }
    if extra_env:
        env.update(extra_env)
    return env


def unpack_command_result(
    result: Any,
    *,
    timeout_message: str,
    error_message: str,
) -> tuple[str, str, int]:
    """Normalize runtime command executor results into stdout/stderr/exit_code."""
    from nexus.runtime.executors.command import CommandRunStatus

    stdout = result.stdout or ''
    if result.status == CommandRunStatus.TimeLimitExceeded:
        return stdout, result.stderr or timeout_message, -1
    if result.status == CommandRunStatus.Error:
        return stdout, result.stderr or error_message, -1
    return stdout, result.stderr or '', result.return_code or 0


def classify_command_failure(
    result: Any,
    *,
    process_failed: bool,
) -> tuple[Literal['agent', 'infrastructure'] | None, str | None]:
    """Distinguish Runtime executor failures from a process that ran and failed."""
    from nexus.runtime.executors.command import CommandRunStatus

    if result.status == CommandRunStatus.Error:
        return 'infrastructure', 'runtime_executor'
    if process_failed:
        return 'agent', 'process'
    return None, None


def unbind_llm_channel(runtime: Any, channel_id: str, handler: Any | None = None) -> None:
    """Release *channel_id*; no-op when *handler* is not its owner."""
    from nexus.runtime.server.services.llm_handler import LLMRequestRouter

    LLMRequestRouter.for_runtime(runtime).unbind(channel_id, handler)


async def call_runtime_agent(
    target: str | Callable[..., Any],
    /,
    no_thinking_request_args: dict[str, Any] | None = None,
    no_thinking_request_args_mode: Literal['replace', 'inherit'] = 'replace',
    haiku_model: str | None = None,
    preserve_thinking: bool = True,
    on_tool_call: Callable[[str], None] | None = None,
    runtime_trace_context: RemoteTraceContext | None = None,
    channel_id: str | None = None,
    persistent_channel: bool = False,
    openai_client_name: str = 'main',
    disable_invoke_wait_timeout: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a runtime-side installed-agent setup function with the shared LLM bridge.

    ``on_tool_call`` is an optional host-side observer invoked with each tool_call
    name as LLM responses come back through the bridge. Used by kilo to detect
    'suggest' early and short-circuit the otherwise long-running invoke.

    ``channel_id`` binds this call to one LLM channel instead of the runtime's
    default route, so several concurrent sessions on one runtime each keep their
    own host-side handler (model client, thinking cache) and do not answer each
    other's requests.

    ``persistent_channel`` keeps that handler bound after this invoke. A later
    call with the same channel reuses its thinking/reasoning cache while
    refreshing the caller's turn-local trace processor context. The session
    owner must eventually call :func:`unbind_llm_channel`.

    The returned dict carries the bridge's per-request outcomes so a caller can tell a model
    call refused on token budget from a real crash — the runtime-side agent only ever sees the
    former as an HTTP 500 and reports a generic failure:

    * ``llm_request_outcomes`` — ``{request_id: error_code | None}`` in completion order, covering
      only the calls settled during this invoke on a persistent channel.
    * ``last_llm_error_code`` — code of the last call to settle, or None if it succeeded.
    * ``last_llm_finish_reason`` — ``finish_reason`` of the last call to settle, or None.
    * ``last_llm_unparsed_tool_call`` — whether the last call to settle began a tool call the
      parser could not extract.

    A caller classifying termination should prefer evidence from the agent's own transcript,
    which names the call it died on, and fall back to ``last_llm_error_code`` only when the
    transcript says nothing usable.

    ``disable_invoke_wait_timeout`` delegates the cumulative invoke budget to the
    caller's active-time deadline. Individual HTTP operations remain bounded.
    """
    from nexus.core import require_context
    from nexus.runtime.models import InvokeRequest
    from nexus.runtime.server.services.llm_handler import LLMRequestHandler, LLMRequestRouter

    ctx = require_context()
    runtime = await ctx.get_runtime()
    await runtime.ws_connect()
    openai_client = ctx.openai_client if openai_client_name == 'main' else ctx.get_openai_client(openai_client_name)

    from nexus.extensions.agents.installed_common.execution import get_agent_execution_target

    execution_target = get_agent_execution_target()
    if execution_target is not None:
        target_callable = target if callable(target) else None
        if target_callable is None or 'execution_target' not in inspect.signature(target_callable).parameters:
            raise ValueError(f'Installed Agent runtime entry {target!r} does not support the selected execution target')
        kwargs['execution_target'] = execution_target.model_dump(mode='json')

    router = LLMRequestRouter.for_runtime(runtime)
    handler = router.get(channel_id) if channel_id and persistent_channel else None
    if handler is not None:
        if handler.openai_client is not openai_client:
            raise RuntimeError(f'Persistent LLM channel {channel_id!r} was created with another OpenAI client')
        await handler.wait_until_idle(timeout=_CHANNEL_IDLE_TIMEOUT)
        handler.refresh_request_context()
    else:
        handler = LLMRequestHandler(
            runtime=runtime,
            openai_client=openai_client,
            no_thinking_request_args=no_thinking_request_args,
            no_thinking_request_args_mode=no_thinking_request_args_mode,
            haiku_model=haiku_model,
            preserve_thinking=preserve_thinking,
            on_tool_call=on_tool_call,
        )
        if channel_id:
            router.bind(channel_id, handler)
        else:
            handler.register()
    settled_before = len(handler.request_outcomes) if persistent_channel else 0
    try:
        request = InvokeRequest.build(
            target,
            venv_id=runtime.default_venv_id,
            trace_context=runtime_trace_context,
            disable_invoke_wait_timeout=disable_invoke_wait_timeout,
            **kwargs,
        )
        result = await runtime.invoke(request)
        if persistent_channel:
            try:
                await handler.wait_until_idle(timeout=_CHANNEL_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                # The runtime already finished this turn; a model call the agent left behind cannot
                # change its result. The next turn on this channel still waits for it to settle.
                logger.warning(
                    'LLM channel %s still has model calls in flight %.0fs after the invoke returned; '
                    'keeping the invoke result',
                    channel_id,
                    _CHANNEL_IDLE_TIMEOUT,
                )
        if isinstance(result, dict):
            outcomes = dict(list(handler.request_outcomes.items())[settled_before:])
            result['llm_request_outcomes'] = outcomes
            result['last_llm_error_code'] = next(reversed(outcomes.values())) if outcomes else None
            last_request_id = next(reversed(outcomes), None)
            if last_request_id is None:
                result['last_llm_finish_reason'] = None
                result['last_llm_unparsed_tool_call'] = False
            else:
                result['last_llm_finish_reason'] = handler.request_finish_reasons.get(last_request_id)
                result['last_llm_unparsed_tool_call'] = handler.request_unparsed_tool_calls.get(last_request_id, False)
        return result
    finally:
        if channel_id and not persistent_channel:
            router.unbind(channel_id, handler)
        elif not channel_id:
            handler.unregister()


def open_cli_tee_files(
    tee_log_dir: str | None,
    agent_key: str,
) -> tuple[IO[str] | None, IO[str] | None]:
    """Open append-mode, line-buffered tee files for CLI stdout/stderr.

    Returns (stdout_fh, stderr_fh). Both are ``None`` when ``tee_log_dir`` is falsy or when
    opening either file fails — logging is a debug aid and never fails the run.
    """
    if not tee_log_dir:
        return None, None
    try:
        directory = Path(tee_log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d')
        stdout_fh = (directory / f'{agent_key}.{stamp}.stdout.log').open('a', encoding='utf-8', buffering=1)
        try:
            stderr_fh = (directory / f'{agent_key}.{stamp}.stderr.log').open('a', encoding='utf-8', buffering=1)
        except Exception:
            stdout_fh.close()
            raise
    except Exception as exc:
        logger.warning('Failed to open tee log files under %s for %s: %s', tee_log_dir, agent_key, exc)
        return None, None
    return stdout_fh, stderr_fh


def write_tee_line(fh: IO[str] | None, line: str) -> None:
    """Append a line (newline added) to a tee file handle."""
    if fh is None:
        return
    with contextlib.suppress(Exception):
        fh.write(line + '\n')


def close_tee_file(fh: IO[str] | None) -> None:
    """Close a tee file handle."""
    if fh is None:
        return
    with contextlib.suppress(Exception):
        fh.close()


def encode_stdin_prompt(prompt: str) -> bytes:
    """Encode a prompt exactly as it would have been encoded as a command-line argument.

    Call it before spawning the child: an unencodable prompt must fail the launch, not reach
    the child as an empty stdin.
    """
    return prompt.encode('utf-8', 'surrogateescape')


async def write_and_close_stdin(stream: asyncio.StreamWriter, data: bytes) -> None:
    """Feed a one-shot prompt to a child and send EOF.

    Run it alongside the stdout/stderr readers: a prompt larger than the pipe buffer blocks
    in ``drain()`` until the child reads it. A child that already exited surfaces through
    its exit code and stderr, so a broken pipe here is ignored.
    """
    try:
        stream.write(data)
        await stream.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        with contextlib.suppress(Exception):
            stream.close()


def __getattr__(name: str) -> Any:
    if name == 'call_native_bridge_runtime_agent':
        from nexus.extensions.agents.installed_common import native_bridge

        return getattr(native_bridge, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'DEFAULT_NPM_REGISTRY',
    'DEFAULT_PROXY_API_KEY',
    'DEFAULT_RUNTIME_BIN_DIR',
    'DEFAULT_TIMEOUT',
    'DEFAULT_WORKING_DIR',
    'PUBLIC_NPM_REGISTRY',
    'InstalledAgentInput',
    'InstalledAgentOutput',
    'InstalledNodeCliInput',
    'build_npm_env',
    'build_runtime_path',
    'build_runtime_command',
    'call_native_bridge_runtime_agent',
    'call_runtime_agent',
    'classify_command_failure',
    'close_tee_file',
    'coerce_user_prompt',
    'encode_stdin_prompt',
    'find_free_port',
    'find_nexus_runtime_binary',
    'find_runtime_binary',
    'open_cli_tee_files',
    'npm_install_timeout',
    'registry_candidates',
    'resolve_npm_bin_entry',
    'resolve_npm_registry',
    'unbind_llm_channel',
    'unpack_command_result',
    'write_and_close_stdin',
    'write_tee_line',
]
