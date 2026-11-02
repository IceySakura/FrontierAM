"""WorkBuddy Agent.

Installs the private ``@tencent/nexus-wb-cli`` npm package in the runtime and
runs it through the shared OpenAI proxy. Generic packages and host tarballs are
kept as explicit fallback paths.
"""

from __future__ import annotations

import base64
import logging
import os

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from nexus.core import agent_registry, require_context
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)
from nexus.runtime.executors.models.file import FileItem

from .compat import DEFAULT_COMPAT_PERMISSION_MODE_PATCH, build_compatibility_patches
from .installer import (
    DEFAULT_WORKBUDDY_VERSION,
    REMOTE_TARBALL,
    iter_file_chunks,
    remote_part_path,
    resolve_tarball_path,
)
from .runtime_setup import DEFAULT_MODEL_ID, DEFAULT_PRESET


logger = logging.getLogger(__name__)


class PreStartScript(BaseModel):
    interpreter: Literal['bash', 'python'] = 'bash'
    code: str
    timeout: float = 300.0
    cwd: str | None = None
    env: dict[str, str] | None = None
    fail_on_error: bool = True


class ConfigFilePatch(BaseModel):
    path: str
    content: dict[str, Any]


class WorkBuddyInput(InstalledNodeCliInput):
    """Input for the WorkBuddy ``wb`` CLI."""

    version: str = DEFAULT_WORKBUDDY_VERSION
    """npm version or dist-tag. A dist-tag (default ``latest``) resolves at install time; an
    exact version is reused from cache and validated against the host-tarball fallback."""

    npm_username: str | None = None
    """Registry username. Falls back to ``WORKBUDDY_NPM_USERNAME``."""

    npm_token: str | None = None
    """Registry token. Falls back to ``WORKBUDDY_NPM_TOKEN``. Not a bearer token."""

    package_url: str | None = None
    """Generic repository URL for the standard npm ``.tgz`` package."""

    package_sha256: str | None = Field(default=None, pattern=r'^[0-9a-fA-F]{64}$')
    """Expected SHA-256 for ``package_url``."""

    tarball_path: str | None = None
    """Development fallback: host path to an uncompressed platform tarball.

    When omitted, installation uses ``package_url`` or the configured private npm
    registry without uploading package bytes from the host.
    """

    model_id: str = DEFAULT_MODEL_ID
    """Bare id written into models.json. 5.9 ``--model`` uses ``custom-local:<id>``."""

    effort: str | None = 'xhigh'
    """``wb --effort``: minimal / low / medium / high / xhigh / max. None omits the flag."""

    thinking: bool = False
    """Declare the model reasoning-capable and enable wb thinking.

    Only then does wb send ``reasoning_effort`` and keep ``reasoning_content`` out of the
    reply. Backends that split reasoning only when an effort is requested otherwise return
    it inline in ``content``, and wb passes it on, including into a child agent's result.
    Off by default because wb then also sends ``verbosity``, which models without reasoning
    support may reject.
    """

    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Merged into ``extra_body.chat_template_kwargs`` of every completion request.

    Mirrors HyTuner's ``data.apply_chat_template_kwargs``; for HY models use
    ``{'reasoning_effort': 'no_think' | 'low' | 'high' | 'max'}``. A ``reasoning_effort`` here
    replaces the top-level one wb derives from ``effort``. Empty sends no template kwargs.
    """

    preset: str | None = DEFAULT_PRESET
    """``wb --preset``: ``code`` / ``work`` / ``design``. Keep this set.

    ``--preset`` picks the WorkBuddy prompt template; ``--mode`` only selects
    fragments *within* one. With no preset the CLI falls back to a short CodeBuddy
    Code prompt (~15k chars, "You are CodeBuddy Code") where ``--mode`` barely
    changes the system text at all -- the four modes are byte-identical apart from
    the model name, and ``ask``/``quick`` still ship Write/Edit/Bash. Setting this
    to None therefore does not get you WorkBuddy behaviour.

    A selected preset can still fall back to CodeBuddy when its compose plugins
    have not loaded. ``rollout_mode`` registers the installed plugins and checks
    the actual model requests; the preset flag alone is not an identity check.
    """

    mode: str | None = None
    """``wb --mode``: ``craft`` / ``ask`` / ``plan`` / ``quick``. None keeps the CLI default (craft).

    Only meaningful together with ``preset``; see that field. Note ``plan`` still
    receives the write tools in its prompt -- its read-only property comes from
    runtime enforcement, not from the tool list, so do not rely on the prompt alone.
    """

    json_output: bool = True
    """Emit JSON Lines on stdout (``--json``)."""

    rollout_mode: bool = False
    """Load the installed prompt plugins, pass ``--features rollout``, and reject a CodeBuddy system prompt.

    Shared configs turn this on with ``WORKBUDDY_ROLLOUT_MODE`` (default true). Left
    off, a missing plugin is silent and the main request can be the CodeBuddy prompt.
    """

    extra_env: dict[str, str] | None = None
    extra_cli_args: list[str] = Field(default_factory=list)
    request_capture_dir: str | None = None
    """Runtime directory for opt-in HTTP body/header metadata capture before host sanitization."""
    config_file_patches: list[ConfigFilePatch] = Field(default_factory=list)
    pre_start_scripts: list[PreStartScript] = Field(default_factory=list)
    permission_mode: str | None = None
    compat_permission_mode_patch: bool = DEFAULT_COMPAT_PERMISSION_MODE_PATCH

    @model_validator(mode='after')
    def validate_package_source(self) -> WorkBuddyInput:
        if self.package_url and not self.package_sha256:
            raise ValueError('package_sha256 is required with package_url')
        if self.compat_permission_mode_patch and not self.permission_mode:
            raise ValueError('permission_mode is required when compat_permission_mode_patch is enabled')
        return self


class WorkBuddyOutput(InstalledAgentOutput):
    """Output from WorkBuddy agent."""

    status: str | None = None
    """Turn status derived from the ``--json`` result line: completed / partial /
    refused / cancelled / timeout / truncation / error."""

    task_outcome: str | None = None
    """Raw ``_meta`` taskOutcome, e.g. SUCCESS / PARTIAL_SUCCESS / FAILED_TOOL_EXECUTION."""

    stop_reason: str | None = None
    """ACP stopReason for the final turn, e.g. end_turn / max_tokens / refusal."""

    assistant_text: str | None = None
    """The agent's visible reply, with thoughts and tool traffic stripped."""

    transcript_path: str | None = None
    """Runtime-side path to the full turn transcript (JSONL)."""

    conversation_id: str | None = None
    """Conversation id, usable with ``--resume``."""

    prompt_identity: dict[str, Any] | None = None
    """Rollout request counts and prompt hashes. The prompt text itself is not included."""


async def _upload_tarball_parts(runtime: Any, tarball_path: str | None) -> list[str]:
    local = resolve_tarball_path(tarball_path)
    logger.debug('Uploading WorkBuddy tarball %s (%d bytes)', local, local.stat().st_size)
    part_paths: list[str] = []
    for index, chunk in enumerate(iter_file_chunks(local)):
        remote = remote_part_path(index, REMOTE_TARBALL)
        response = await runtime.upload_file(
            [FileItem(path=remote, content=base64.b64encode(chunk).decode('ascii'), encoding='base64')],
            overwrite=True,
        )
        for result in getattr(response, 'results', None) or []:
            if not getattr(result, 'success', True):
                raise RuntimeError(f'Failed to upload WorkBuddy tarball part {remote}: {result.message}')
        part_paths.append(remote)
        logger.debug('Uploaded tarball part %d -> %s (%d bytes)', index, remote, len(chunk))
    if not part_paths:
        raise RuntimeError(f'WorkBuddy tarball is empty: {local}')
    return part_paths


@agent_registry.register('workbuddy', meta={'gateway': {'scaffold_policy': 'workbuddy'}})
async def run_workbuddy(input_data: WorkBuddyInput) -> WorkBuddyOutput:
    """Run WorkBuddy inside the runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)
    # CODEBUDDY_API_KEY and the other routing vars are enforced runtime-side; anything
    # passed here for them is dropped with a warning.
    extra_env = dict(input_data.extra_env or {})
    extra_cli_args = list(input_data.extra_cli_args)
    config_file_patches = [patch.model_dump() for patch in input_data.config_file_patches]
    pre_start_scripts = [script.model_dump() for script in input_data.pre_start_scripts]
    if input_data.compat_permission_mode_patch:
        compat = build_compatibility_patches(input_data.version, input_data.permission_mode or '')
        extra_cli_args.extend(compat['extra_cli_args'])
        config_file_patches.extend(compat['config_file_patches'])
        pre_start_scripts.extend(compat['pre_start_scripts'])

    logger.debug('Starting WorkBuddy agent for task: %s...', task[:100])

    npm_username = (input_data.npm_username or os.getenv('WORKBUDDY_NPM_USERNAME') or '').strip()
    npm_token = (input_data.npm_token or os.getenv('WORKBUDDY_NPM_TOKEN') or '').strip()
    if bool(npm_username) != bool(npm_token):
        raise RuntimeError('WORKBUDDY_NPM_USERNAME and WORKBUDDY_NPM_TOKEN must both be set')

    part_paths: list[str] | None = None
    if not npm_username:
        ctx = require_context()
        runtime = await ctx.get_runtime()
        await runtime.ws_connect()
        part_paths = (
            await _upload_tarball_parts(runtime, input_data.tarball_path)
            if input_data.tarball_path and not input_data.package_url
            else []
        )

    from .runtime_setup import setup_and_run_workbuddy

    result = await call_runtime_agent(
        setup_and_run_workbuddy,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        npm_registry=input_data.npm_registry,
        npm_token=npm_token or None,
        npm_username=npm_username or None,
        version=input_data.version,
        package_url=input_data.package_url,
        package_sha256=input_data.package_sha256,
        model_id=input_data.model_id,
        effort=input_data.effort,
        thinking=input_data.thinking,
        chat_template_kwargs=input_data.chat_template_kwargs or None,
        preset=input_data.preset,
        mode=input_data.mode,
        json_output=input_data.json_output,
        rollout_mode=input_data.rollout_mode,
        extra_env=extra_env,
        extra_cli_args=extra_cli_args,
        request_capture_dir=input_data.request_capture_dir,
        config_file_patches=config_file_patches,
        pre_start_scripts=pre_start_scripts,
        remote_part_paths=part_paths,
    )

    success = result.get('success', False)
    failure_kind = result.get('failure_kind')
    failure_stage = result.get('failure_stage')
    if not success and failure_kind is None:
        # A runtime that never reached the classifier still has to be attributable.
        failure_kind = 'agent' if result.get('exit_code') not in (None, -1) else 'infrastructure'
        failure_stage = 'process' if failure_kind == 'agent' else 'runtime_protocol'

    logger.info(
        'WorkBuddy execution completed: success=%s status=%s taskOutcome=%s',
        success,
        result.get('status'),
        result.get('task_outcome'),
    )
    return WorkBuddyOutput(
        success=success,
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        failure_kind=failure_kind,
        failure_stage=failure_stage,
        status=result.get('status'),
        task_outcome=result.get('task_outcome'),
        stop_reason=result.get('stop_reason'),
        assistant_text=result.get('assistant_text'),
        transcript_path=result.get('transcript_path'),
        conversation_id=result.get('conversation_id'),
        prompt_identity=result.get('prompt_identity'),
    )


__all__ = [
    'ConfigFilePatch',
    'PreStartScript',
    'WorkBuddyInput',
    'WorkBuddyOutput',
    'run_workbuddy',
]
