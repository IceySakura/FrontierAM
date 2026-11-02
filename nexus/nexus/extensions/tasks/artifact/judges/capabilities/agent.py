"""Run Claude Code on a supplied runtime and collect its reward and trajectory.

Use runtime events when supported, otherwise a local LLM bridge.
Callers retain ownership of supplied runtimes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict


if TYPE_CHECKING:
    from nexus.core.openai_client import OpenAICompatibleClient
    from nexus.extensions.agents.claude_code.agent import ClaudeCodeInput
    from nexus.extensions.tasks.artifact.judges.base import JudgeInput


logger = logging.getLogger(__name__)


class _BridgeRunKwargs(TypedDict):
    working_dir: str
    timeout: float
    extra_cli_args: list[str]
    extra_env: dict[str, str]
    version: str
    npm_registry: str | None
    thinking_preset: str
    conversion_log_level: str
    pre_start_scripts: list[dict[str, Any]]
    config_file_patches: list[dict[str, Any]]
    mcp_servers: list[dict[str, Any]]


def runtime_supports_events(runtime: Any) -> bool:
    """Probe event subscription and detach the probe; method presence alone is insufficient."""
    on = getattr(runtime, 'on', None)
    if on is None:
        return False

    def _probe(*_args: Any, **_kwargs: Any) -> None:
        """Never invoked — detached before any event can fire."""

    try:
        unsubscribe = on('llm_request', _probe)
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001
        # An unexpected failure to subscribe is still "cannot deliver events".
        logger.debug('runtime event probe failed; treating as event-less', exc_info=True)
        return False

    if callable(unsubscribe):
        with contextlib.suppress(Exception):
            unsubscribe()
    else:
        off = getattr(runtime, 'off', None)
        if off is not None:
            with contextlib.suppress(Exception):
                off('llm_request', _probe)
    return True


@dataclass
class ClaudeCodeResult:
    """Structured result from a ``run_claude_code_on_runtime`` call."""

    output: str = ''
    stderr: str = ''
    error: str | None = None
    runtime: Any = None
    runtime_name: str = ''
    execution_time: float = 0.0
    trajectories: dict[str, Any] = field(default_factory=dict)


async def _provision_runtime(runtime_name: str, image: str) -> Any:
    """Configure and acquire a runtime, forwarding the worker venv and caller exclusions."""
    from nexus.core.context import require_context

    ctx = require_context()
    if image:
        provider = ctx.runtime_provider
        assert provider is not None, 'runtime provider is not configured'
        main_params = provider.get_params('main')
        extra_args = main_params.get('extra_args', [])
        provider.configure(runtime_name, image=image, extra_args=extra_args)
    # Forward the main runtime worker venv to evaluation containers.
    main_venv = ctx.runtime_configs.get('main', {}).get('worker_venv')
    if main_venv and runtime_name != 'main':
        # Normalize to a dict so we can inject exclude_patterns without
        # clobbering caller intent.
        venv_cfg: dict[str, Any] = dict(main_venv) if isinstance(main_venv, dict) else {}
        # Merge exclude_patterns (dedup) — never drop what the caller specified.
        excludes = list(venv_cfg.get('exclude_patterns') or [])
        for pat in ('.agents/**', '.venv/**', '.claude/**', 'workspace/**', '*.pyc', '__pycache__/**'):
            if pat not in excludes:
                excludes.append(pat)
        venv_cfg['exclude_patterns'] = excludes
        ctx.runtime_configs.setdefault(runtime_name, {})['worker_venv'] = venv_cfg
    runtime = await ctx.get_runtime(runtime_name)
    await runtime.ws_connect()  # idempotent on reuse
    return runtime


async def run_claude_code_on_prepared_runtime(
    runtime: Any,
    runtime_name: str,
    *,
    task_prompt: str,
    openai_client: OpenAICompatibleClient,
    claude_input: ClaudeCodeInput,
    reward_file: str | None = None,
    pre_run_setup: Callable[..., Any] | None = None,
) -> ClaudeCodeResult:
    """Run one Claude Code pass without releasing the caller-owned runtime."""
    from nexus.extensions.agents.claude_code.runtime_setup import setup_and_run_claude_code
    from nexus.runtime.server.services.llm_handler import LLMRequestHandler
    from nexus.trace import processor_context
    from nexus.trace.processors.chat_tracker import ChatTrackerProcessor

    result = ClaudeCodeResult(runtime_name=runtime_name, runtime=runtime)
    # Use task_prompt; omit None fields so runtime defaults still apply.
    invoke_kwargs = claude_input.model_dump(exclude={'user_prompt'}, exclude_none=True)
    use_events = runtime_supports_events(runtime)

    chat_tracker = ChatTrackerProcessor()
    with processor_context([chat_tracker], finalize_on_exit=True):
        # Reconnect inside processor_context so the socket read loop inherits tracing context.
        # Local runtimes need no reconnect.
        if use_events:
            try:
                await runtime.ws_disconnect()
                await runtime.ws_connect()
            except Exception as e:  # noqa: BLE001
                logger.warning('ws reconnect before agent run failed (continuing): %s', e)

        # Preserve signed thinking blocks across request turns.
        handler = LLMRequestHandler(
            runtime=runtime,
            openai_client=openai_client,
            preserve_thinking=True,
            no_thinking_request_args=claude_input.no_thinking_request_args,
            no_thinking_request_args_mode=claude_input.no_thinking_request_args_mode,
            haiku_model=claude_input.haiku_model,
        )
        if use_events:
            handler.register()
        try:
            # Clear reward files after preparation so restored files cannot revive stale verdicts.
            if pre_run_setup is not None:
                setup_ret = pre_run_setup(runtime)
                if asyncio.iscoroutine(setup_ret) or asyncio.isfuture(setup_ret):
                    await setup_ret
            if reward_file:
                await runtime.call_executor(
                    'run_command',
                    {'command': f'rm -f {reward_file}', 'timeout': 10},
                )
            try:
                if use_events:
                    import inspect

                    runtime_params = inspect.signature(setup_and_run_claude_code).parameters
                    runtime_kwargs = {k: v for k, v in invoke_kwargs.items() if k in runtime_params}
                    cc_result = await runtime.invoke(
                        setup_and_run_claude_code,
                        task=task_prompt,
                        **runtime_kwargs,
                    )
                else:
                    cc_result = await _run_claude_code_via_bridge(
                        task_prompt=task_prompt,
                        openai_client=openai_client,
                        invoke_kwargs=invoke_kwargs,
                    )
                result.output = cc_result.get('output', '')
                result.stderr = cc_result.get('stderr', '')
                result.execution_time = cc_result.get('execution_time', 0.0)
                if not cc_result.get('success', False):
                    result.error = f'claude-code exit_code={cc_result.get("exit_code", -1)}'
            except Exception as e:
                result.error = f'{type(e).__name__}: {e}'
                logger.exception('run_claude_code_on_prepared_runtime: claude-code raised an error')
        finally:
            if use_events:
                handler.unregister()

    result.trajectories = chat_tracker.get_result()
    turns = len(result.trajectories.get('trajectories', []))
    logger.info(
        'claude-code [%s]: execution_time=%.1fs, turns=%d, error=%s',
        runtime_name,
        result.execution_time,
        turns,
        result.error,
    )
    return result


async def _run_claude_code_via_bridge(
    *,
    task_prompt: str,
    openai_client: OpenAICompatibleClient,
    invoke_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run locally through the shared LLM handler.

    Forward supported launch settings and warn on unsupported non-default options.
    """
    from nexus.extensions.tasks.artifact.judges.capabilities.agent_direct_gateway import (
        DEFAULT_THINKING_PRESET,
        LLMHandlerBridge,
        run_claude_code_with_bridge,
    )

    # Forwarded — the bridge runner accepts these directly.
    forwarded: _BridgeRunKwargs = {
        'working_dir': invoke_kwargs.get('working_dir') or '/workspace',
        'timeout': float(invoke_kwargs.get('timeout') or 900.0),
        'extra_cli_args': list(invoke_kwargs.get('extra_cli_args') or []),
        'extra_env': dict(invoke_kwargs.get('extra_env') or {}),
        'version': str(invoke_kwargs.get('version') or ''),
        'npm_registry': invoke_kwargs.get('npm_registry') or None,
        'thinking_preset': str(invoke_kwargs.get('thinking_preset') or '') or DEFAULT_THINKING_PRESET,
        'conversion_log_level': invoke_kwargs.get('conversion_log_level') or 'ERROR',
        # Skills are installed through pre_start_scripts, not a ClaudeCodeInput field.
        'pre_start_scripts': list(invoke_kwargs.get('pre_start_scripts') or []),
        'config_file_patches': list(invoke_kwargs.get('config_file_patches') or []),
        'mcp_servers': list(invoke_kwargs.get('mcp_servers') or []),
    }

    # Bridge support: npm installation, CLI turn limits, and additive MCP servers.
    # Non-default strict MCP and unattended-hook settings below remain unsupported.
    unsupported_with_values = {
        k: invoke_kwargs[k]
        for k in ('mcp_config_servers', 'mcp_servers_mode', 'mcp_config_mode', 'enable_unattended_interaction_hooks')
        if invoke_kwargs.get(k) not in (None, '', {}, [], False, 'additive', 'cli', 'replace')
    }
    if unsupported_with_values:
        # Loud, because a judge asking for strict MCP replacement and silently
        # getting additive is a scoring difference, not a cosmetic one.
        logger.warning(
            'bridge path ignores non-default claude_input fields %s — '
            'extend run_claude_code_with_bridge if a judge depends on them',
            unsupported_with_values,
        )

    async with LLMHandlerBridge(
        openai_client,
        preserve_thinking=True,
        no_thinking_request_args=invoke_kwargs.get('no_thinking_request_args'),
        no_thinking_request_args_mode=invoke_kwargs.get('no_thinking_request_args_mode', 'replace'),
        haiku_model=invoke_kwargs.get('haiku_model'),
    ) as bridge:
        return await run_claude_code_with_bridge(
            task_prompt=task_prompt,
            llm_bridge=bridge,
            **forwarded,
        )


async def run_claude_code_on_runtime(
    image: str,
    task_prompt: str,
    openai_client: OpenAICompatibleClient,
    claude_input: ClaudeCodeInput,
    pre_run_setup: Callable[..., Any] | None = None,
    reward_file: str | None = None,
) -> ClaudeCodeResult:
    """Provision a runtime and leave it alive for reward collection; the caller must release it."""
    job_id = uuid.uuid4().hex[:8]
    runtime_name = f'eval_web_app_{job_id}'
    result = ClaudeCodeResult(runtime_name=runtime_name)
    try:
        runtime = await _provision_runtime(runtime_name, image)
        result = await run_claude_code_on_prepared_runtime(
            runtime,
            runtime_name,
            task_prompt=task_prompt,
            openai_client=openai_client,
            claude_input=claude_input,
            reward_file=reward_file,
            pre_run_setup=pre_run_setup,
        )
    except Exception as e:
        result.error = f'{type(e).__name__}: {e}'
        logger.exception('run_claude_code_on_runtime: failed to provision/run')
    return result


@dataclass
class RewardRun:
    """Agent output and reward data with separate execution and read/parse errors."""

    cc: ClaudeCodeResult
    raw: str = ''
    parsed: dict[str, Any] = field(default_factory=dict)
    agent_error: str | None = None
    read_error: str | None = None


async def run_agent_and_read_reward(
    *,
    image: str,
    task_prompt: str,
    openai_client: OpenAICompatibleClient,
    claude_input: ClaudeCodeInput,
    reward_file: str,
    parser: Callable[[str], dict[str, Any]] | None = None,
    pre_run_setup: Callable[..., Any] | None = None,
    prepared_runtime: Any = None,
    prepared_runtime_name: str = '',
) -> RewardRun:
    """Run the agent and read its reward; release only runtimes created by this call."""
    run = RewardRun(cc=ClaudeCodeResult())
    try:
        if prepared_runtime is not None:
            run.cc = await run_claude_code_on_prepared_runtime(
                prepared_runtime,
                runtime_name=prepared_runtime_name or 'prepared',
                task_prompt=task_prompt,
                openai_client=openai_client,
                claude_input=claude_input,
                reward_file=reward_file,
                pre_run_setup=pre_run_setup,
            )
        else:
            run.cc = await run_claude_code_on_runtime(
                image=image,
                task_prompt=task_prompt,
                openai_client=openai_client,
                claude_input=claude_input,
                reward_file=reward_file,
                pre_run_setup=pre_run_setup,
            )
        run.agent_error = run.cc.error
        if run.cc.runtime is not None:
            try:
                run.raw, run.parsed = await read_reward_file(run.cc.runtime, reward_file, parser=parser)
            except Exception as e:
                run.read_error = f'read_reward_file failed: {e}'
                logger.error('run_agent_and_read_reward: failed to read/parse result: %s', e)
    except Exception as e:
        run.agent_error = f'{type(e).__name__}: {e}'
        logger.exception('run_agent_and_read_reward: failed to run eval')
    finally:
        # Only release runtimes WE provisioned. A caller-supplied prepared_runtime
        # is owned by the caller (e.g. the pipeline releases it after all judges).
        if prepared_runtime is None and run.cc.runtime_name:
            await release_runtime(run.cc.runtime_name)
    return run


async def read_reward_file(
    runtime: Any,
    file_path: str,
    parser: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Download a file from *runtime* and optionally parse it.

    Returns ``(raw_text, parsed_data)``.
    """
    from nexus.runtime.executors.file import FileItem

    # Best-effort sync before reading the reward file.
    try:
        await runtime.call_executor('run_command', {'command': 'sync', 'timeout': 10})
    except Exception as e:  # noqa: BLE001
        logger.debug('read_reward_file: sync before read failed (continuing): %s', e)

    dl_resp = await runtime.download_file([FileItem(path=file_path)])
    raw = ''
    if dl_resp.files:
        raw = (dl_resp.files[0].content or '').strip()
    if dl_resp.errors:
        logger.warning('read_reward_file download errors: %s', dl_resp.errors)
    # Fall back to cat when the download returns no content.
    if not raw:
        try:
            res = await runtime.call_executor('run_command', {'command': f'cat {file_path}', 'timeout': 15})
            out = getattr(res, 'stdout', None) or (res.get('stdout') if isinstance(res, dict) else '') or ''
            if out.strip():
                raw = out.strip()
                logger.info('read_reward_file: recovered %d chars via cat fallback', len(raw))
        except Exception as e:  # noqa: BLE001
            logger.debug('read_reward_file: cat fallback failed: %s', e)
    logger.debug('read_reward_file raw (%s):\n%s', file_path, raw[:500])

    parsed = parser(raw) if parser else {}
    return raw, parsed


async def release_runtime(runtime_name: str) -> None:
    """Safely release a runtime by name. Logs warnings on failure."""
    try:
        from nexus.core.context import require_context

        provider = require_context().runtime_provider
        assert provider is not None, 'runtime provider is not configured'
        await provider.release(name=runtime_name)
        logger.debug('Released runtime %s', runtime_name)
    except Exception as e:
        logger.warning('Failed to release runtime %s: %s', runtime_name, e)


async def call_judge_agent(
    inp: JudgeInput,
    *,
    prompt: str,
    reward_file: str,
    parser: Callable[[str], dict[str, Any]] | None = None,
    client_role: str = 'agent',
    with_playwright: bool = True,
    with_skill: bool = True,
    timeout: float | None = None,
    max_turns: int | None = None,
    pre_run_setup: Callable[..., Any] | None = None,
) -> RewardRun:
    """Resolve a client, run the agent, and read its reward using the supplied runtime.

    Raise structured no_client, agent_failed, or parse_failed errors.
    """
    from nexus.extensions.tasks.artifact.agent_setup.claude_config import default_judge_claude_input
    from nexus.extensions.tasks.artifact.judges.base import JudgeExecutionError
    from nexus.extensions.tasks.artifact.judges.registry import resolve_client_role

    client = resolve_client_role(inp.openai_clients, client_role)
    if client is None:
        raise JudgeExecutionError(
            'no_client',
            f'call_judge_agent: no client for role {client_role!r}; available = {sorted(inp.openai_clients)}',
        )

    claude_input = default_judge_claude_input(
        timeout=timeout,
        max_turns=max_turns,
        with_playwright=with_playwright,
        with_skill=with_skill,
    )
    model_config = inp.model_configs.get(client_role)
    if model_config is not None:
        claude_input = model_config.apply_agent(claude_input)

    try:
        run = await run_agent_and_read_reward(
            image=inp.image,
            task_prompt=prompt,
            openai_client=client,
            claude_input=claude_input,
            reward_file=reward_file,
            parser=parser,
            pre_run_setup=pre_run_setup,
            prepared_runtime=inp.runtime,
            prepared_runtime_name=inp.instance_id,
        )
    except Exception as exc:
        raise JudgeExecutionError(
            'agent_failed',
            f'call_judge_agent: {type(exc).__name__}: {exc}',
        ) from exc

    # Both error channels are distinct on ``RewardRun`` on purpose (see the
    # RewardRun docstring). Preserve the distinction when mapping upward.
    if run.agent_error:
        raise JudgeExecutionError('agent_failed', run.agent_error)
    if run.read_error:
        raise JudgeExecutionError('parse_failed', run.read_error)
    return run


__all__ = [
    'ClaudeCodeResult',
    'RewardRun',
    '_provision_runtime',
    'call_judge_agent',
    'run_claude_code_on_prepared_runtime',
    'run_claude_code_on_runtime',
    'run_agent_and_read_reward',
    'read_reward_file',
    'release_runtime',
]
