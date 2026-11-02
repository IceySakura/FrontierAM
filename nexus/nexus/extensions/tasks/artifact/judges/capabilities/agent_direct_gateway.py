"""Run Claude Code locally through AnthropicProxy and a Nexus OpenAI client.

Reuse the shared response handler for message conversion, signed thinking,
and configured model routing without a socket.io event channel.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
import uuid

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from nexus.core.openai_client import OpenAICompatibleClient


logger = logging.getLogger(__name__)

# The thinking preset the infer agent runs with. A judge that read signed
# thinking differently from the agent would be scoring a different transcript.
DEFAULT_THINKING_PRESET = 'bedrock-claude'


class LLMHandlerBridge:
    """Serve OpenAI-shaped requests through the shared LLMRequestHandler."""

    # Per-run totals; reset before each judge run.
    _run_totals: dict[str, int] = {}

    def __init__(
        self,
        openai_client: OpenAICompatibleClient,
        *,
        preserve_thinking: bool = True,
        no_thinking_request_args: dict[str, Any] | None = None,
        no_thinking_request_args_mode: Literal['replace', 'inherit'] = 'replace',
        haiku_model: str | None = None,
    ) -> None:
        # Import here so this module can be imported for type-checking without
        # eagerly pulling the runtime server tree.
        from nexus.runtime.server.services.llm_handler import LLMRequestHandler

        # Response handling does not require a runtime; never register events on this bridge.
        self._handler = LLMRequestHandler(
            runtime=None,  # type: ignore[arg-type]
            openai_client=openai_client,
            preserve_thinking=preserve_thinking,
            no_thinking_request_args=no_thinking_request_args,
            no_thinking_request_args_mode=no_thinking_request_args_mode,
            haiku_model=haiku_model,
        )
        # Per-bridge totals mirror the class-level dict so an operator wanting
        # to see one agent's usage in isolation can inspect ``bridge.usage``.
        self.usage: dict[str, int] = {
            'calls': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'cache_read_input_tokens': 0,
            'cache_creation_input_tokens': 0,
        }

    async def __aenter__(self) -> LLMHandlerBridge:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        pass

    async def request(self, openai_params: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
        """Handle one request without socket.io heartbeats.

        The timeout argument is unused; the underlying client owns its timeout.
        """
        del timeout  # underlying client owns this
        request_id = str(uuid.uuid4())
        response = await self._handler._produce_response(request_id, dict(openai_params), rsio_mode=False)

        # Accumulate normalized usage for this bridge and the current judge run.
        usage = response.get('usage') if isinstance(response, dict) else None
        if isinstance(usage, dict):
            details = usage.get('prompt_tokens_details') or {}
            cache_read = int(usage.get('cache_read_input_tokens') or details.get('cached_tokens') or 0)
            cache_create = int(usage.get('cache_creation_input_tokens') or 0)
            prompt = int(usage.get('prompt_tokens') or 0)
            completion = int(usage.get('completion_tokens') or 0)
            for target in (self.usage, LLMHandlerBridge._run_totals):
                target['calls'] = target.get('calls', 0) + 1
                target['prompt_tokens'] = target.get('prompt_tokens', 0) + prompt
                target['completion_tokens'] = target.get('completion_tokens', 0) + completion
                target['cache_read_input_tokens'] = target.get('cache_read_input_tokens', 0) + cache_read
                target['cache_creation_input_tokens'] = target.get('cache_creation_input_tokens', 0) + cache_create
            logger.debug(
                'llm_call rid=%s prompt=%s cache_read=%s cache_create=%s completion=%s',
                request_id[:8],
                prompt,
                cache_read,
                cache_create,
                completion,
            )
        return response


def reset_run_totals() -> None:
    """Clear ``LLMHandlerBridge._run_totals``. Call once per judge run."""
    LLMHandlerBridge._run_totals = {}


def install_skills(skill_dirs: list[str] | None) -> None:
    """Copy bundled skill directories into ~/.claude/skills before CLI startup."""
    import shutil

    for source in skill_dirs or []:
        name = os.path.basename(str(source).rstrip('/'))
        if not name:
            continue
        target = Path(os.path.expanduser(f'~/.claude/skills/{name}'))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)
        logger.debug('installed skill %s', name)


async def run_claude_code_with_bridge(
    *,
    task_prompt: str,
    llm_bridge: Any,
    working_dir: str = '/workspace',
    timeout: float = 900.0,
    extra_cli_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    skill_dirs: list[str] | None = None,
    version: str = '',
    npm_registry: str | None = None,
    thinking_preset: str = DEFAULT_THINKING_PRESET,
    conversion_log_level: str = 'ERROR',
    pre_start_scripts: list[dict[str, Any]] | None = None,
    config_file_patches: list[dict[str, Any]] | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one local Claude Code pass and return output, status, and timing.

    Apply config patches, pre-start scripts, and MCP registration in that order.
    Use shared installer and streaming helpers; pass the prompt through stdin.
    """
    from nexus.extensions.agents.claude_code.installer import install_claude_code_via_npm
    from nexus.extensions.agents.claude_code.runtime_setup import (
        DEFAULT_INSTALL_DIR,
        _build_single_run_command,
        _run_claude_streaming,
    )
    from nexus.extensions.agents.installed_common import (
        DEFAULT_NPM_REGISTRY,
        DEFAULT_PROXY_API_KEY,
        build_runtime_path,
        find_free_port,
    )
    from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy

    started = time.monotonic()

    def failure(stderr: str) -> dict[str, Any]:
        return {
            'success': False,
            'output': '',
            'stderr': stderr,
            'exit_code': -1,
            'execution_time': time.monotonic() - started,
        }

    try:
        claude_path = await install_claude_code_via_npm(
            install_dir=DEFAULT_INSTALL_DIR,
            version=version or 'latest',
            npm_registry=npm_registry or DEFAULT_NPM_REGISTRY,
        )
    except Exception as exc:
        logger.exception('claude-code install failed')
        return failure(f'claude-code install failed: {type(exc).__name__}: {exc}')

    install_skills(skill_dirs)

    port = find_free_port()
    # Preassign the bridge before starting the proxy to avoid socket.io setup.
    proxy = AnthropicProxy(
        sio=None,  # type: ignore[arg-type]
        listen_port=port,
        listen_host='127.0.0.1',
        request_timeout=timeout,
        thinking_config=thinking_preset,
        conversion_log_level=conversion_log_level,
    )
    proxy.bridge = llm_bridge
    await proxy.start()
    try:
        # Use the shared runtime environment with a local proxy endpoint.
        claude_env: dict[str, str] = {
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{port}',
            'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
            'HOME': os.environ.get('HOME', '/root'),
            'PATH': build_runtime_path(DEFAULT_INSTALL_DIR),
            'IS_SANDBOX': '1',
            # Disable the CLI idle watchdog while long backend requests emit only SSE comments.
            'CLAUDE_ENABLE_STREAM_WATCHDOG': '0',
            **(extra_env or {}),
        }
        Path(working_dir).mkdir(parents=True, exist_ok=True)

        # Prepare config patches, pre-start scripts, then MCP before launching the CLI.
        from nexus.extensions.agents.claude_code.mcp import register_mcp_servers
        from nexus.extensions.agents.claude_code.runtime_setup import (
            _apply_json_patch,
            _run_pre_start_script,
        )

        for patch in config_file_patches or []:
            _apply_json_patch(patch['path'], patch.get('content') or {})

        for index, script in enumerate(pre_start_scripts or []):
            try:
                await _run_pre_start_script(dict(script), working_dir, claude_env)
            except Exception as exc:
                # Do not score without the required preparation.
                logger.exception('pre-start script %d failed', index)
                return failure(f'pre_start_scripts[{index}] failed: {type(exc).__name__}: {exc}')

        if mcp_servers:
            # Without this an agent judge reached its prompt's "browse the
            # page" instruction with no browser tool at all.
            await register_mcp_servers(
                claude_path,
                working_dir=working_dir,
                claude_env=claude_env,
                mcp_servers=mcp_servers,
            )

        cmd = _build_single_run_command(claude_path, extra_cli_args=extra_cli_args)
        logger.debug('running claude-code against direct bridge: timeout=%.0fs port=%s', timeout, port)
        exit_code, stdout, stderr = await _run_claude_streaming(
            cmd=cmd,
            cwd=working_dir,
            env=claude_env,
            timeout=timeout,
            stdin_input=task_prompt,
        )
    except Exception as exc:
        logger.exception('claude-code run failed')
        return failure(f'{type(exc).__name__}: {exc}')
    finally:
        with contextlib.suppress(Exception):
            await proxy.stop()

    return {
        'success': exit_code == 0,
        'output': stdout,
        'stderr': stderr,
        'exit_code': exit_code,
        'execution_time': time.monotonic() - started,
    }


__all__ = [
    'DEFAULT_THINKING_PRESET',
    'LLMHandlerBridge',
    'install_skills',
    'reset_run_totals',
    'run_claude_code_with_bridge',
]
