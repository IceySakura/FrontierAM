"""OpenCC Agent.

Runs OpenCC (OpenClaude) inside a runtime sandbox. The host-side code
uploads a standalone binary (bun-compiled, no Node.js dependency) to the
runtime, starts an Anthropic-to-OpenAI proxy, and runs
``opencc-bin --print`` with the task prompt.  LLM calls flow through the
host-side OpenAI client via Socket.IO.
"""

import logging

from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.core import agent_registry
from nexus.extensions.agents.installed_common import (
    InstalledAgentInput,
    InstalledAgentOutput,
    coerce_user_prompt,
)


logger = logging.getLogger(__name__)


class PreStartScript(BaseModel):
    """A script to execute inside the runtime before OpenCC starts."""

    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    timeout: float = 300.0
    cwd: str | None = None
    env: dict[str, str] | None = None
    fail_on_error: bool = True


class OpenCCInput(InstalledAgentInput):
    """Input for OpenCC agent."""

    extra_env: dict[str, str] | None = None
    """Extra environment variables for the OpenCC process."""

    extra_cli_args: list[str] = Field(default_factory=list)
    """Additional CLI arguments appended before the task prompt."""

    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)
    """Scripts executed in order before OpenCC starts."""

    conversion_log_level: str = 'INFO'
    """Log level for anthropic<->openai conversion dumps."""

    thinking_preset: str = 'hunyuan'
    """Thinking format preset for the anthropic<->openai converter."""

    no_thinking_request_args: dict[str, Any] | None = None
    """Override ``request_args`` for LLM calls when OpenCC does NOT request thinking."""

    haiku_model: str | None = None
    """Model name to use when OpenCC requests a haiku-class model."""


class OpenCCOutput(InstalledAgentOutput):
    """Output from OpenCC agent."""


async def _upload_binaries_to_runtime(runtime: Any) -> tuple[str, str]:
    """Upload both glibc and musl OpenCC binaries to the sandbox.

    Returns (remote_glibc_path, remote_musl_path).
    """
    import base64

    from nexus.runtime.executors.models.file import FileItem

    from .installer import BIN_GLIBC, BIN_MUSL, BIN_PATH_GLIBC, BIN_PATH_MUSL

    files: list[FileItem] = []
    remote_glibc = f'/tmp/{BIN_GLIBC}'
    remote_musl = f'/tmp/{BIN_MUSL}'

    for local, remote, label in [
        (BIN_PATH_GLIBC, remote_glibc, 'glibc'),
        (BIN_PATH_MUSL, remote_musl, 'musl'),
    ]:
        if not local.exists():
            raise FileNotFoundError(f'OpenCC {label} binary not found at {local}')
        content_b64 = base64.b64encode(local.read_bytes()).decode('ascii')
        files.append(FileItem(path=remote, content=content_b64, encoding='base64'))
        logger.debug('Queued %s binary for upload (%d bytes)', label, local.stat().st_size)

    await runtime.upload_file(files, overwrite=True)
    logger.debug('Uploaded glibc+musl binaries to sandbox')
    return remote_glibc, remote_musl


@agent_registry.register('opencc')
async def run_opencc(input_data: OpenCCInput) -> OpenCCOutput:
    """Run OpenCC agent inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)
    extra_env = dict(input_data.extra_env or {})

    logger.debug('Starting OpenCC agent for task: %s...', task[:100])

    from nexus.core import require_context
    from nexus.runtime.server.services.llm_handler import LLMRequestHandler

    from .runtime_setup import setup_and_run_opencc

    ctx = require_context()
    runtime = await ctx.get_runtime()
    await runtime.ws_connect()

    # Upload both glibc and musl binaries to the runtime sandbox.
    remote_glibc, remote_musl = await _upload_binaries_to_runtime(runtime)

    handler = LLMRequestHandler(
        runtime=runtime,
        openai_client=ctx.openai_client,
        no_thinking_request_args=input_data.no_thinking_request_args,
        haiku_model=input_data.haiku_model,
    )
    handler.register()
    try:
        result = await runtime.invoke(
            setup_and_run_opencc,
            task=task,
            working_dir=input_data.working_dir,
            timeout=input_data.timeout,
            remote_glibc_path=remote_glibc,
            remote_musl_path=remote_musl,
            extra_env=extra_env,
            extra_cli_args=list(input_data.extra_cli_args),
            pre_start_scripts=[s.model_dump() for s in input_data.pre_start_scripts],
            conversion_log_level=input_data.conversion_log_level,
            thinking_preset=input_data.thinking_preset,
        )
    finally:
        handler.unregister()

    logger.info('OpenCC execution completed: success=%s', result.get('success'))
    return OpenCCOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
    )


__all__ = [
    'OpenCCInput',
    'OpenCCOutput',
    'PreStartScript',
    'run_opencc',
]
