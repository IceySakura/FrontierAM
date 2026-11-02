"""Run the packaged Handy CLI as a Nexus installed agent."""

import base64
import hashlib
import logging

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nexus.core import agent_registry
from nexus.extensions.agents.installed_common import (
    InstalledAgentOutput,
    InstalledNodeCliInput,
    call_runtime_agent,
    coerce_user_prompt,
)

from .installer import DEFAULT_HANDY_VERSION
from .runtime_setup import DEFAULT_HANDY_MAX_OUTPUT_TOKENS, DEFAULT_HANDY_MODEL


logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_DIR = '/tmp/nexus/handy-artifacts'
"""Sandbox directory Handy writes deliveries to. Outside the run's temp session root,
which is deleted before the host could fetch anything from it."""

MAX_ARTIFACT_FETCH_BYTES = 64 * 1024 * 1024
"""Per-file ceiling for host-side fetches.

``download_file`` inlines the whole file as base64 in one RPC response, and the
Socket.IO server caps a message at 100 MB, so base64's 4/3 expansion puts the hard
limit near 75 MB. Stay below it and report oversized files instead of failing the run
— Handy itself allows up to 1 GiB, so this mismatch is expected, not exceptional.
"""


class HandyArtifact(BaseModel):
    """A file Handy delivered through ``submit_file_artifact``."""

    artifact_id: str
    filename: str
    content_type: str
    size: int
    sha256: str
    title: str | None = None
    created_at: str | None = None
    sandbox_path: str
    local_path: str | None = None
    """Where the file was written on the host, or None if it was not fetched."""

    fetch_error: str | None = None
    """Why the file was not fetched (too large, checksum mismatch, download failure)."""


class HandyInput(InstalledNodeCliInput):
    """Input for a single complete Handy CLI run."""

    version: str = DEFAULT_HANDY_VERSION
    """Version of ``@tencent/handy-agent`` to install in the runtime."""

    model: str = Field(default=DEFAULT_HANDY_MODEL, min_length=1)
    """Model identity exposed to Handy; Nexus remains authoritative for routing."""

    max_output_tokens: int = Field(default=DEFAULT_HANDY_MAX_OUTPUT_TOKENS, gt=0)
    """Maximum output token budget requested by Handy."""

    extra_env: dict[str, str] | None = None
    """Web, media, and artifact configuration passed to Handy."""

    conversion_log_level: str = 'INFO'
    """Log level for Anthropic protocol conversion diagnostics."""

    thinking_preset: str = 'hunyuan'
    """Thinking-format preset used by the Nexus Anthropic proxy."""

    no_thinking_request_args: dict[str, Any] | None = None
    """Host request overrides for Handy calls that do not request thinking."""

    haiku_model: str | None = None
    """Host model override for Handy requests targeting a haiku-class model."""

    collect_artifacts: bool = False
    """Keep Handy's ``submit_file_artifact`` deliveries on disk and fetch them to the host."""

    artifact_dir: str = DEFAULT_ARTIFACT_DIR
    """Sandbox directory Handy writes deliveries to when *collect_artifacts* is set."""

    artifact_output_dir: str | None = None
    """Host directory to write fetched artifacts into. Metadata only when unset."""

    plugin_dirs: list[str] | None = None
    """Session plugin directories, passed through as repeated ``--plugin-dir`` flags.

    Handy merges these with the ``plugins`` list in its packaged ``app.yaml`` rather
    than replacing it, so this only needs to name plugins that list omits. Handy
    ships some that way — ``deep-research-worker`` and ``ui-ux-pro-max`` carry the
    optional ``research-worker`` and ``artifact-worker`` subagents, which stay
    unavailable until mounted here.

    Paths resolve inside the sandbox after the npm install completes; a relative path
    is taken against the installed ``@tencent/handy-agent`` directory, so callers can
    write ``plugins/ui-ux-pro-max`` without knowing the install prefix.
    """


class HandyOutput(InstalledAgentOutput):
    """Output from a single Handy CLI run."""

    artifacts: list[HandyArtifact] = Field(default_factory=list)
    """Files Handy delivered through ``submit_file_artifact``."""


async def _fetch_artifacts(
    manifest: list[dict[str, Any]],
    output_dir: str | None,
) -> list[HandyArtifact]:
    """Fetch each delivered file to the host, verifying size and checksum.

    A single bad artifact is recorded on that artifact and skipped rather than
    failing the whole run: the agent's work is already done by this point, and a
    partial set of deliveries is more useful than none.
    """
    from nexus.core import require_context
    from nexus.runtime.executors.models.file import FileItem

    artifacts = [
        HandyArtifact(
            artifact_id=entry.get('artifactId', ''),
            filename=entry.get('filename', ''),
            content_type=entry.get('contentType', 'application/octet-stream'),
            size=int(entry.get('size') or 0),
            sha256=entry.get('sha256', ''),
            title=entry.get('title'),
            created_at=entry.get('createdAt'),
            sandbox_path=entry.get('sandbox_path', ''),
        )
        for entry in manifest
    ]
    if not artifacts:
        return []

    fetchable = []
    for artifact in artifacts:
        if artifact.size > MAX_ARTIFACT_FETCH_BYTES:
            artifact.fetch_error = (
                f'{artifact.size} bytes exceeds the {MAX_ARTIFACT_FETCH_BYTES} byte transport limit; '
                'left in the sandbox'
            )
            logger.warning('Skipping oversized Handy artifact %s: %s', artifact.filename, artifact.fetch_error)
        elif not artifact.sandbox_path:
            artifact.fetch_error = 'metadata carried no sandbox path'
        else:
            fetchable.append(artifact)
    if not fetchable:
        return artifacts

    runtime = await require_context().get_runtime()
    response = await runtime.download_file(
        files=[FileItem(path=artifact.sandbox_path, encoding='base64') for artifact in fetchable]
    )
    downloaded = {item.path: item for item in response.files}
    for result in response.errors or []:
        logger.warning('Failed to download Handy artifact %s: %s', result.path, result.message)

    target_dir = Path(output_dir) if output_dir else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)

    for artifact in fetchable:
        item = downloaded.get(artifact.sandbox_path)
        if item is None or item.content is None:
            artifact.fetch_error = 'file was not returned by download_file'
            continue
        payload = base64.b64decode(item.content)
        digest = hashlib.sha256(payload).hexdigest()
        if artifact.sha256 and digest != artifact.sha256:
            artifact.fetch_error = f'sha256 mismatch: expected {artifact.sha256}, got {digest}'
            logger.warning('Handy artifact %s failed checksum verification', artifact.filename)
            continue
        if target_dir is None:
            continue
        # Namespace by artifact_id: Handy only sanitises the filename, so two
        # deliveries can legitimately share one.
        destination = target_dir / artifact.artifact_id / artifact.filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        artifact.local_path = str(destination)
        logger.debug('Fetched Handy artifact %s -> %s (%d bytes)', artifact.filename, destination, len(payload))

    return artifacts


@agent_registry.register('handy')
async def run_handy(input_data: HandyInput) -> HandyOutput:
    """Install and run Handy inside the Nexus runtime sandbox."""
    task = coerce_user_prompt(input_data.user_prompt)
    logger.debug('Starting Handy for task: %s...', task[:100])

    from .runtime_setup import setup_and_run_handy

    result = await call_runtime_agent(
        setup_and_run_handy,
        no_thinking_request_args=input_data.no_thinking_request_args,
        haiku_model=input_data.haiku_model,
        task=task,
        working_dir=input_data.working_dir,
        timeout=input_data.timeout,
        version=input_data.version,
        model=input_data.model,
        max_output_tokens=input_data.max_output_tokens,
        npm_registry=input_data.npm_registry,
        extra_env=dict(input_data.extra_env or {}),
        conversion_log_level=input_data.conversion_log_level,
        thinking_preset=input_data.thinking_preset,
        artifact_dir=input_data.artifact_dir if input_data.collect_artifacts else None,
        plugin_dirs=list(input_data.plugin_dirs or []),
    )
    logger.info('Handy execution completed: success=%s', result.get('success'))

    artifacts: list[HandyArtifact] = []
    if input_data.collect_artifacts:
        manifest = result.get('artifacts') or []
        try:
            artifacts = await _fetch_artifacts(manifest, input_data.artifact_output_dir)
        except Exception:
            # The run itself succeeded; surface the failure without discarding its output.
            logger.exception('Failed to fetch Handy artifacts')

    return HandyOutput(
        success=result.get('success', False),
        output=result.get('output', ''),
        stderr=result.get('stderr', ''),
        exit_code=result.get('exit_code', -1),
        execution_time=result.get('execution_time', 0),
        artifacts=artifacts,
    )


__all__ = ['HandyArtifact', 'HandyInput', 'HandyOutput', 'run_handy']
