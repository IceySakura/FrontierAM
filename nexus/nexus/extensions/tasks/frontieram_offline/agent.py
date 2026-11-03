"""Nexus agent for a host Codex controller with only an offline workspace tool."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import time

from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

from pydantic import ConfigDict, Field

from nexus.core import AgentRunInput, AgentRunOutput, OpenAIConfig, agent_registry, require_context

from .client import BoundedOpenAIClient
from .proxy import OfflineResponsesProxy


if TYPE_CHECKING:
    from .runtime import OfflineDockerRuntime


DISABLED_FEATURES = (
    'apps',
    'plugins',
    'shell_tool',
    'unified_exec',
    'view_image',
    'multi_agent',
    'multi_agent_v2',
    'browser_use',
    'computer_use',
    'image_generation',
    'code_mode',
    'shell_snapshot',
    'hooks',
    'enable_request_compression',
    'unbounded_connection_retries',
    'workspace_dependencies',
    'skill_search',
    'skill_mcp_dependency_install',
    'goals',
    'daemon_auto_start',
    'tool_suggest',
    'api_key_model_discovery',
)


class OfflineCodexInput(AgentRunInput):
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    user_prompt: str
    run_dir: str
    model: str = 'claude-opus-5-5'
    base_url: str
    key_file: str | None = None
    allow_live_endpoint: bool = False
    development_seconds: float = Field(default=7200, gt=0, le=7200)
    max_http_attempts: int = Field(default=800, ge=1, le=800)
    max_output_tokens: int = Field(default=64000, ge=128, le=64000)
    auto_compact_token_limit: int = Field(default=110000, ge=1024, lt=131072)


class OfflineCodexOutput(AgentRunOutput):
    exit_status: str
    exit_code: int
    evaluation_eligible: bool
    stop_reason: str
    source_frozen: bool
    development_seconds: float
    http_attempts: int
    http_retries: int
    cache_task_id: str | None = None
    model: str
    evidence_dir: str
    error_categories: list[str] = Field(default_factory=list)


def codex_settings(
    model: str, base_url: str, mcp: dict[str, Any], seconds: float, compact_limit: int = 110000
) -> dict[str, Any]:
    """The benchmark's narrow tool policy, independent of the user's Codex home."""
    values: dict[str, Any] = {
        'model': model,
        'model_provider': 'nexus_offline',
        'model_providers.nexus_offline.name': 'nexus_offline',
        'model_providers.nexus_offline.base_url': base_url,
        'model_providers.nexus_offline.wire_api': 'responses',
        'model_providers.nexus_offline.env_key': 'FRONTIERAM_PROXY_TOKEN',
        'model_providers.nexus_offline.requires_openai_auth': False,
        'model_providers.nexus_offline.supports_websockets': False,
        'model_providers.nexus_offline.request_max_retries': 0,
        'model_providers.nexus_offline.stream_max_retries': 0,
        'model_providers.nexus_offline.stream_idle_timeout_ms': int((seconds + 30) * 1000),
        'web_search': 'disabled',
        'project_doc_max_bytes': 0,
        'skills.max_context_tokens': 1,
        'skills.include_instructions': False,
        'skills.bundled.enabled': False,
        'check_for_update_on_startup': False,
        'analytics.enabled': False,
        'model_context_window': 131072,
        'model_auto_compact_token_limit': compact_limit,
        'mcp_servers.workspace.command': mcp['command'],
        'mcp_servers.workspace.args': mcp['args'],
        'mcp_servers.workspace.required': True,
        'mcp_servers.workspace.enabled_tools': ['exec'],
        'mcp_servers.workspace.startup_timeout_sec': 20,
        'mcp_servers.workspace.tool_timeout_sec': 130,
        'mcp_servers.workspace.default_tools_approval_mode': 'approve',
        'developer_instructions': (
            'Only workspace.exec is available. All file work and tests must use this offline Docker tool. '
            'Host tools, network/search, external connectors, questions and subagents are unavailable.'
        ),
    }
    for name, setting in mcp.get('env', {}).items():
        values[f'mcp_servers.workspace.env.{name}'] = setting
    values.update({f'features.{name}': False for name in DISABLED_FEATURES})
    return values


async def _capture(stream, path: Path, finished: asyncio.Event, errors: list[str], *, events: bool) -> None:
    total = 0
    try:
        with path.open('wb') as output:
            while line := await stream.readline():
                total += len(line)
                if total > 8 * 1024 * 1024:
                    errors.append('controller_log_limit')
                    finished.set()
                    return
                output.write(line)
                output.flush()
                if events:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError('Controller event must be an object')
                    if event.get('type') == 'turn.completed':
                        finished.set()
                    elif event.get('type') == 'turn.failed':
                        errors.append('codex_turn_failed')
                        finished.set()
    except asyncio.CancelledError:
        raise
    except Exception:
        # A broken reader must stop development, not silently become an empty
        # error list that could later qualify as a normal deadline submission.
        errors.append('controller_event_stream_error' if events else 'controller_stderr_stream_error')
        finished.set()


async def _drain_readers(readers: list[asyncio.Task], errors: list[str], *, timeout: float = 2.0) -> None:
    """Consume buffered final events without waiting forever for inherited FDs."""
    if not readers:
        return
    done, pending = await asyncio.wait(readers, timeout=timeout)
    if pending:
        errors.append('controller_log_drain_timeout')
        for reader in pending:
            reader.cancel()
        # These are our own cancellations of the known _capture coroutine,
        # whose cancellation handler always propagates immediately.
        await asyncio.gather(*pending, return_exceptions=True)
    for reader in done:
        if reader.cancelled():
            raise asyncio.CancelledError
        if reader.exception() is not None:
            errors.append('controller_reader_task_error')


async def _settle_cleanup(awaitable) -> None:
    """Complete bounded teardown, then propagate any caller cancellation."""
    cleanup = asyncio.create_task(awaitable)
    owner = asyncio.current_task()
    cancelled = bool(owner and owner.cancelling())
    while True:
        try:
            await asyncio.shield(cleanup)
            break
        except asyncio.CancelledError:
            if cleanup.cancelled():
                raise
            cancelled = True
        except Exception as error:
            if cancelled:
                raise asyncio.CancelledError from error
            raise
    if cancelled:
        raise asyncio.CancelledError


def _classify_stop(
    stop_reason: str,
    *,
    final_seen: bool,
    returncode: int | None,
    errors: list[str],
    error_events: list[dict[str, Any]],
    deadline: float,
    frozen: bool,
) -> tuple[str, bool]:
    """Classify only after buffered controller events have been consumed."""
    normal_final = (
        stop_reason in {'turn_completed', 'codex_process_exited'} and final_seen and returncode == 0 and not errors
    )
    if normal_final:
        return 'turn_completed', frozen
    expected_codes = {0, -signal.SIGTERM, -signal.SIGKILL, 128 + signal.SIGTERM, 128 + signal.SIGKILL}
    boundary_errors = {'deadline', 'stopped', 'cancelled'}
    errors_at_boundary = all(
        isinstance(event.get('at_monotonic'), (int, float))
        and event['at_monotonic'] >= deadline - 0.001
        and event.get('category') in boundary_errors
        for event in error_events
    )
    if error_events and errors_at_boundary:
        expected_codes.add(1)
    normal_deadline = (
        stop_reason == 'development_deadline'
        and returncode in expected_codes
        and all(category in boundary_errors | {'codex_turn_failed'} for category in errors)
        and (not errors or bool(error_events))
        and errors_at_boundary
    )
    return stop_reason, frozen and normal_deadline


@agent_registry.register('frontieram_codex_offline')
async def offline_codex_agent(value: OfflineCodexInput) -> OfflineCodexOutput:
    """Use Nexus's client/protocol conversion while keeping all candidate code offline."""
    runtime = cast('OfflineDockerRuntime', await require_context().get_runtime())
    assets = runtime.assets
    run_dir = Path(value.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ('home', 'codex_home', 'empty_host_cwd'):
        (run_dir / name).mkdir()
    parsed = urlparse(value.base_url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Model endpoint must not contain credentials or query parameters')
    is_fixture = parsed.hostname in {'127.0.0.1', 'localhost', '::1'}
    if not is_fixture and (
        not value.allow_live_endpoint or value.base_url.rstrip('/') != 'http://api.aihub.woa.com/standard/v1'
    ):
        raise ValueError('Live model calls require an explicit live profile and the approved endpoint')
    client_args: dict[str, Any] = {'base_url': value.base_url}
    if is_fixture:
        if value.key_file:
            raise ValueError('Local fixtures must not read a real credential')
        api_key = 'local-fixture-only'
    else:
        # AIHub routes by cache_task_id: one value per attempt keeps the whole conversation,
        # including retries and compaction turns, on one upstream account.
        client_args['cache_task_id'] = 'frontieram-' + secrets.token_hex(16)
        if not value.key_file:
            raise ValueError('A private credential file is required for a live profile')
        key_path = Path(value.key_file).expanduser()
        if key_path.stat().st_mode & 0o077:
            raise ValueError('Credential file must be private to its owner')
        api_key = key_path.read_text().strip()
    metadata_dir = assets.source_root / 'harness_recovery/metadata'
    protocol = json.loads((metadata_dir / f'{value.model}.protocol.json').read_text())['data']
    parameters = {p['name']: p for p in protocol['standard_protocol']['openai_chat']['parameters']}
    output_parameter = 'max_completion_tokens' if 'max_completion_tokens' in parameters else 'max_tokens'
    provider_timeout = 600 if value.model == 'deepseek-v4-pro' else (None if value.model == 'hy4-preview' else 3600)
    retry_policy = json.loads((metadata_dir / 'retry_policies.json').read_text())
    started = time.monotonic()
    deadline = started + value.development_seconds
    deadline_unix = time.time() + value.development_seconds
    mcp = runtime.mcp_config(deadline_monotonic=deadline, deadline_unix=deadline_unix)
    client = BoundedOpenAIClient(
        OpenAIConfig(client_args={'api_key': api_key, **client_args}, request_args={'model': value.model}),
        deadline=deadline,
        retry_policy=retry_policy,
        max_http_attempts=value.max_http_attempts,
        audit_path=run_dir / 'http_audit.jsonl',
        model=value.model,
        max_output_tokens=value.max_output_tokens,
        output_parameter=output_parameter,
        provider_timeout_max_seconds=provider_timeout,
    )
    client._client_key = 'main'
    token = secrets.token_urlsafe(32)
    proxy = OfflineResponsesProxy(client, token=token)
    process = None
    readers: list[asyncio.Task] = []
    waiters: list[asyncio.Task] = []
    observed_errors: list[str] = []
    stop_reason = 'controller_error'
    stopped_at = None
    frozen_at = None
    frozen = False
    finished = asyncio.Event()
    try:
        await proxy.start()
        settings = codex_settings(
            value.model, proxy.base_url, mcp, value.development_seconds, value.auto_compact_token_limit
        )
        command = [
            str(assets.codex_path),
            '--no-daemon',
            '-a',
            'never',
            'exec',
            '--ignore-user-config',
            '--ignore-rules',
            '--ephemeral',
            '--skip-git-repo-check',
            '--strict-config',
            '--json',
            '--sandbox',
            'read-only',
            '--color',
            'never',
        ]
        for key, setting in settings.items():
            command.extend(['-c', key + '=' + json.dumps(setting, ensure_ascii=False)])
        command.append(value.user_prompt)
        environment = {
            'PATH': '/usr/bin:/bin:/usr/local/bin',
            'HOME': str(run_dir / 'home'),
            'CODEX_HOME': str(run_dir / 'codex_home'),
            'FRONTIERAM_PROXY_TOKEN': token,
            'RUST_LOG': 'error',
            'TERM': 'dumb',
        }
        # Credentials remain solely in the host-side client, never in this child or its tools.
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=run_dir / 'empty_host_cwd',
            env=environment,
            start_new_session=True,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,
        )
        readers = [
            asyncio.create_task(
                _capture(process.stdout, run_dir / 'codex_events.jsonl', finished, observed_errors, events=True)
            ),
            asyncio.create_task(
                _capture(process.stderr, run_dir / 'codex_stderr.log', finished, observed_errors, events=False)
            ),
        ]
        waiters = [asyncio.create_task(process.wait()), asyncio.create_task(finished.wait())]
        await asyncio.wait(waiters, timeout=max(0, deadline - time.monotonic()), return_when=asyncio.FIRST_COMPLETED)
        stopped_at = time.monotonic()
        if stopped_at >= deadline:
            stop_reason = 'development_deadline'
        elif finished.is_set() and not observed_errors:
            stop_reason = 'turn_completed'
        else:
            stop_reason = 'codex_process_exited'
    except asyncio.CancelledError:
        stop_reason = 'runner_cancelled'
        raise
    except Exception as error:
        observed_errors.append(type(error).__name__)
    finally:
        if stopped_at is None:
            stopped_at = time.monotonic()
        client.stop()

        async def cleanup() -> None:
            nonlocal frozen, frozen_at
            try:
                await runtime.pause()
                frozen = True
                frozen_at = time.monotonic()
            finally:
                try:
                    if process and process.returncode is None:
                        if stop_reason == 'turn_completed':
                            with suppress(asyncio.TimeoutError):
                                await asyncio.wait_for(process.wait(), timeout=1)
                        if process.returncode is None:
                            with suppress(ProcessLookupError):
                                os.killpg(process.pid, signal.SIGTERM)
                            try:
                                await asyncio.wait_for(process.wait(), timeout=2)
                            except asyncio.TimeoutError:
                                with suppress(ProcessLookupError):
                                    os.killpg(process.pid, signal.SIGKILL)
                                try:
                                    await asyncio.wait_for(process.wait(), timeout=2)
                                except asyncio.TimeoutError:
                                    observed_errors.append('controller_process_reap_timeout')
                finally:
                    for waiter in waiters:
                        waiter.cancel()
                    await asyncio.gather(*waiters, return_exceptions=True)
                    try:
                        await _drain_readers(readers, observed_errors)
                    finally:
                        try:
                            await proxy.stop()
                        finally:
                            await client.aclose()

        await _settle_cleanup(cleanup())
    errors = [*observed_errors]
    if client.errors:
        errors.extend(
            str(e.get('category', e.get('kind', 'model_request_error'))) if isinstance(e, dict) else type(e).__name__
            for e in client.errors
        )
    proxy_errors = getattr(proxy, 'errors', None) or []
    if proxy_errors:
        errors.extend(
            str(e.get('category', 'protocol_error')) if isinstance(e, dict) else type(e).__name__ for e in proxy_errors
        )
    error_events = [*client.errors, *proxy_errors]
    stop_reason, eligible = _classify_stop(
        stop_reason,
        final_seen=finished.is_set(),
        returncode=process.returncode if process is not None else None,
        errors=errors,
        error_events=error_events,
        deadline=deadline,
        frozen=frozen,
    )
    result = OfflineCodexOutput(
        exit_status='finished' if eligible else ('inference_error' if errors else 'runtime_error_by_env'),
        exit_code=0 if eligible else -1,
        evaluation_eligible=eligible,
        stop_reason=stop_reason,
        source_frozen=frozen,
        development_seconds=stopped_at - started,
        http_attempts=client.http_attempts,
        http_retries=client.retry_count,
        cache_task_id=client_args.get('cache_task_id'),
        model=value.model,
        evidence_dir=str(run_dir),
        error_categories=errors,
    )
    (run_dir / 'agent_result.json').write_text(result.model_dump_json(indent=2) + '\n')
    (run_dir / 'boundary.json').write_text(
        json.dumps(
            {
                'deadline_monotonic': deadline,
                'deadline_unix': deadline_unix,
                'stop_observed_monotonic': stopped_at,
                'workspace_frozen_monotonic': frozen_at,
                'cleanup_elapsed_seconds': time.monotonic() - stopped_at,
                'never_extended': True,
                'host_tools_disabled': True,
                'isolated_home': True,
                'actual_model_service_calls': 0 if is_fixture else client.http_attempts,
                'framework': 'Nexus Task/Agent, OpenAIClient, ResponsesProxy, native converters and trace',
                'offline_profile': 'dedicated host Codex agent; not the built-in runtime-server codex agent',
            },
            indent=2,
        )
        + '\n'
    )
    return result
