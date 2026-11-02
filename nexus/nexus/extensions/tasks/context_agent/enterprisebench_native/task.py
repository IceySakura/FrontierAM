"""EnterpriseBench task implemented directly on Nexus Runtime APIs.

This task is agent-agnostic — it works with ANY nexus agent (terminus2, claude_code,
codex, swe_agent, etc.). The agent is configured via RunInput.agent_name, NOT here.

The task's responsibility is:
1. Initialize the runtime environment
2. Delegate to the configured agent (via run_agent)
3. Upload and execute the task's verifier directly in the Runtime
4. Collect RewardKit/OpenCode results and Nexus evidence

The supported task-file subset matches the 300 SyntaraOffice tasks. Unsupported
Harbor-only features are rejected explicitly instead of being silently ignored.
"""

import asyncio
import copy
import json
import logging
import math
import posixpath
import shlex
import shutil
import tempfile
import time
import uuid

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from nexus.core import (
    OpenAICompatibleClient,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.context_agent.bad_patterns import ToolCallBadPatternPolicy, apply_bad_pattern_score
from nexus.extensions.tasks.context_agent.enterprisebench_native.errors import (
    EnterpriseAgentTimeoutError,
    EnterpriseJudgeError,
    EnterpriseJudgeLengthError,
    EnterpriseJudgeTimeoutError,
    EnterpriseRuntimeRPCError,
    EnterpriseRuntimeStartupTimeoutError,
    EnterpriseStageError,
    enterprise_stage,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.evidence import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_FILES,
    persist_run_records,
    persist_workspace_evidence,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.host_judge_capture import ResponsesCapture
from nexus.extensions.tasks.context_agent.enterprisebench_native.judge_monitor import JudgeMonitor
from nexus.extensions.tasks.context_agent.enterprisebench_native.verifier_config import (
    VERIFIER_ENV,
    normalize_judge_config,
    render_judge_toml,
    render_opencode_config,
    render_prompt_template,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.workspace_permissions import checked_workspace_root
from nexus.extensions.tasks.context_agent.output import agent_termination_reason as _agent_termination_reason
from nexus.extensions.tasks.context_agent.output import apply_tool_name_resolution_score, judge_skip_reason
from nexus.extensions.tasks.context_agent.task_files_auth import bind_task_wedata_key
from nexus.extensions.tasks.context_agent.termination import report_task_termination
from nexus.runtime.executors import FileItem
from nexus.trace import get_current_span_id, update_span
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)


RUNTIME_TESTS_DIR = '/tests'
RUNTIME_VERIFIER_DIR = '/logs/verifier'
RUNTIME_ARTIFACTS_DIR = '/logs/artifacts'
RUNTIME_TEST_STDOUT = f'{RUNTIME_VERIFIER_DIR}/test-stdout.txt'


class _YamlAuthoritativeJudgeClient:
    """Let the configured Judge request args override OpenCode defaults."""

    def __init__(
        self,
        client: Any,
        deadline: float | None = None,
        monitor: JudgeMonitor | None = None,
        api_records: ResponsesCapture | None = None,
    ):
        from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient

        self._monitor = monitor or JudgeMonitor()
        self._client = client
        # Clone for Task-local request capture, retaining the transport and retry policy.
        # Never mutate a client shared with another task or policy rollout.
        if isinstance(client, OpenAIResponsesClient):
            self._client = copy.copy(client)
            from nexus.extensions.tasks.context_agent.enterprisebench_native.host_judge_capture import attach_capture

            attach_capture(self._client, api_records if api_records is not None else ResponsesCapture(), self._monitor)
        self.config = client.config
        self._deadline = deadline
        self._pending: set[asyncio.Task] = set()

    def cancel_pending(self) -> None:
        for task in self._pending:
            task.cancel()

    async def create_completion(
        self,
        messages: list[Any],
        model: str | Any = None,
        **kwargs: Any,
    ) -> Any:
        controls = kwargs.pop('_judge_response_controls', {})
        if not isinstance(controls, dict) or set(controls) - {'store', 'metadata', 'service_tier'}:
            raise ValueError('Invalid Judge response controls')
        kwargs = {**controls, **kwargs}
        configured = getattr(self.config, 'request_args', {}) or {}
        for key in configured:
            if key not in {'messages', 'input', 'tools'}:
                kwargs.pop(key, None)
        if 'max_output_tokens' in configured or 'max_tokens' in configured:
            kwargs.pop('max_output_tokens', None)
            kwargs.pop('max_tokens', None)
        if 'reasoning' in configured or 'reasoning_effort' in configured:
            kwargs.pop('reasoning', None)
            kwargs.pop('reasoning_effort', None)
        if configured.get('model') is not None:
            model = None
        request = self._monitor.start_request() if self._monitor is not None else None
        task = asyncio.current_task()
        if task is not None:
            self._pending.add(task)
        error: BaseException | None = None
        try:
            if self._deadline is None:
                return await self._client.create_completion(messages=messages, model=model, **kwargs)
            async with enterprise_stage(
                'judge.model',
                max(0, self._deadline - time.monotonic()),
                EnterpriseJudgeTimeoutError,
            ):
                return await self._client.create_completion(messages=messages, model=model, **kwargs)
        except BaseException as exc:
            from nexus.core import ContextLengthExceededError

            error = exc
            if isinstance(exc, ContextLengthExceededError):
                raise self._monitor.length_failure('context_length_exceeded') from None
            if self._monitor.failure is not None:
                raise self._monitor.failure from None
            raise
        finally:
            if request is not None and self._monitor is not None:
                self._monitor.finish_request(*request, error)
            if task is not None:
                self._pending.discard(task)


def _judge_launch_settings(judge_client: Any | None) -> tuple[int, str | None]:
    """Return OpenCode settings derived from the authoritative Judge YAML."""
    if judge_client is None:
        return 32768, None
    request_args = getattr(getattr(judge_client, 'config', None), 'request_args', {}) or {}
    output_limit = request_args.get('max_output_tokens', request_args.get('max_tokens', 32768))
    try:
        output_limit = int(output_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Judge max output tokens must be an integer, got {output_limit!r}') from exc
    if output_limit <= 0:
        raise ValueError('Judge max output tokens must be positive')

    reasoning = request_args.get('reasoning')
    effort = reasoning.get('effort') if isinstance(reasoning, Mapping) else request_args.get('reasoning_effort')
    return output_limit, str(effort).strip() if effort else None


def _positive_int(value: Any, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} must be an integer, got {value!r}') from exc
    if parsed <= 0:
        raise ValueError(f'{field_name} must be positive')
    return parsed


@dataclass(frozen=True)
class EnterpriseTaskSpec:
    """Nexus-normalized task metadata plus paths to the materialized payload."""

    name: str
    instruction: str
    tests_dir: Path
    solution_dir: Path


class EnvironmentInitError(RuntimeError):
    """Environment boot/healthcheck failed before the agent could run.

    Raised by ``_boot_image_services`` on launch timeout or failed readiness
    polling. The task reports ``INITIALIZATION_ERROR`` through tracing so the trainer can mask these
    samples separately from genuine ``UNHANDLED_ERROR`` (model/code) failures.
    Subclasses RuntimeError for backward compatibility with existing
    ``except RuntimeError`` handlers.
    """


class EnterpriseBenchNativeTaskInput(TaskRunInput, WithAgentRunner):
    """Input for an EnterpriseBench rollout and its in-sandbox Judge.

    Agent-specific configuration remains in ``agent_config``. This task owns
    the EnterpriseBench environment, evidence, and verifier lifecycle.
    """

    # EnterpriseBench task identification.
    #
    # Empty is legal when the run carries dataset task_files: remote workers do
    # not share the filesystem where the dataset was authored, so the task
    # directory must be materialized from the item's files blob.
    task_dir: str = ''  # Path to the materialized task directory
    instance_id: str = ''  # Task instance ID for tracking
    benchmark_task_name: str = ''
    task_description: str = ''

    # Task configuration (passed to agent as user_prompt)
    instruction: str = ''  # Override instruction (if empty, read from task dir)
    workdir: str = '/app'  # Working directory in container

    # Execution parameters
    agent_timeout: int = 7200  # Agent execution timeout (seconds)
    judge_timeout: int = Field(default=2400, gt=0)  # Total verifier deadline, including preparation and retries.
    judge_max_consecutive_failures: int = Field(default=3, ge=0)  # 0 disables this guard.
    judge_max_consecutive_rate_limits: int = Field(default=3, ge=0)
    judge_no_progress_timeout: float = Field(default=600, ge=0)  # 0 disables; judge_timeout still applies.
    test_timeout: float = Field(default=7200.0, gt=0)  # Legacy command cap; cannot extend judge_timeout.
    rewardkit_max_concurrent_llm: int = 4
    rewardkit_max_concurrent_agent: int = 2
    reward_mode: Literal['sample_acc', 'rubric_acc'] = 'sample_acc'
    zero_score_on_tool_name_resolution_error: bool = False
    tool_call_bad_pattern_policy: ToolCallBadPatternPolicy | None = None

    # Execution switches. Current SyntaraOffice tasks support shared verification only.
    run_oracle: bool = False  # Run oracle solution instead of agent
    run_nop: bool = False  # Skip both oracle and agent; verify pristine post-boot state
    verifier_mode: Literal['shared', 'separate'] = 'shared'
    verifier_tests_prebaked: bool = False
    skip_tests_upload: bool = False  # Skip uploading tests (pre-baked in image)
    skip_environment_upload: bool = True  # Skip default environment/workspace overlay onto workdir
    environment_uploads: list[dict[str, Any]] = []  # Items from [environment.upload] to copy into the sandbox
    expose_workdir_files: bool = False  # Pass pre-agent workdir files to agents that accept context_files
    workdir_context_max_files: int = 500
    judge_execution_mode: Literal['host', 'runtime_direct'] = 'host'
    judge_client_name: str = ''  # Host OpenAI client exposed to an in-runtime Responses judge proxy
    judge_config: dict[str, Any] = Field(default_factory=dict)
    judge_mcp_url: str = 'http://localhost:8900/mcp'
    WEDATA_KEY: str | dict[str, Any] | None = Field(default=None, exclude=True, repr=False)
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    artifact_max_files: int = DEFAULT_MAX_FILES
    artifact_max_bytes: int = DEFAULT_MAX_BYTES
    network_isolation_enabled: bool = False  # Retained in the input contract for result reproducibility.
    verifier_network_isolation_enabled: bool = False  # Retained for compatibility with existing Taskfiles.

    # Image boot hook: gongfeng replaces the image ENTRYPOINT with nexus-runtime, so
    # any in-image init (e.g. Canyon's /omnibus/start_all.sh that brings up the MCP
    # HTTP servers) never runs on its own. When boot_command is set we launch it in
    # the background (the script may block forever), inject boot_env (e.g. ENABLED_APPS
    # from task.toml [environment.env], which the bypassed ENV directives no longer
    # provide), then poll readiness derived from [environment.healthcheck] before the
    # agent runs.
    boot_command: str = ''  # e.g. "/omnibus/start_all.sh"
    boot_env: dict[str, str] = {}  # env for the boot process (ENABLED_APPS, DOJO_FAKE_NOW, ...)
    boot_timeout: float = 7200.0  # Wall-clock timeout for the detached image-service launch.
    healthcheck_command: str = ''  # from [environment.healthcheck].command; rc 0 means ready
    healthcheck_url: str = ''  # inferred from healthcheck.command URL for curl fallback; empty = no wait
    healthcheck_retries: int = 180  # from healthcheck.retries; max readiness attempts
    healthcheck_interval_sec: float = 3.0  # from healthcheck.interval_sec; poll interval seconds
    healthcheck_timeout_sec: float = 3.0  # from healthcheck.timeout_sec; per-probe timeout seconds
    healthcheck_start_period_sec: float = 0.0  # from healthcheck.start_period_sec; initial grace period


class EnterpriseBenchNativeTaskOutput(TaskRunOutput, WithScore):
    """EnterpriseBench result with binary reward and audit evidence."""

    extra_fields: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None
    reward: float = 0.0
    rewards: dict[str, float] = Field(default_factory=dict)
    raw_reward: float | None = None
    sample_accuracy: float = 0.0
    rubric_accuracy: float | None = None
    reward_metadata: dict[str, Any] = Field(default_factory=dict)
    test_stdout: str = ''
    verifier_error: str = ''
    grading_style: str | None = 'rubric'
    judge_unavailable: bool = False
    evaluation_skipped: bool = False
    rationale: str | None = None
    requirement_status: list[str] = Field(default_factory=list)
    graded_on: str | None = 'agent_deliverables_and_world_state'
    criteria_passed: int | None = None
    criteria_total: int | None = None
    judge_rounds: int | None = None
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    workspace_artifact: dict[str, Any] | None = None
    evidence_artifact: dict[str, Any] | None = None
    external_services_diff: str | None = None
    judge_details: dict[str, Any] = Field(default_factory=dict)
    judge_trajectories: list[dict[str, Any]] = Field(default_factory=list)
    termination_reason: str = 'ENV_DONE'


def _skipped_judge_summary(reason: str) -> dict[str, Any]:
    """Build a zero-score result for a deliberate incomplete-output Judge skip."""
    return {
        'score': 0.0,
        'sample_accuracy': 0.0,
        'raw_reward': None,
        'rubric_accuracy': None,
        'judge_unavailable': False,
        'rationale': reason,
        'requirement_status': [],
        'criteria_passed': None,
        'criteria_total': None,
        'judge_rounds': None,
        'criteria': [],
    }


def _criterion_passed(criterion: dict[str, Any]) -> bool:
    value = criterion.get('value')
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) >= 1.0
    return str(criterion.get('raw') or '').strip().lower() in {'yes', 'pass', 'passed', 'true'}


def _summarize_judge(
    judge_details: dict[str, Any],
    judge_trajectories: list[dict[str, Any]],
    *,
    verifier_error: str,
    fallback_reward: float | None,
) -> dict[str, Any]:
    """Derive stable task metrics from RewardKit's per-criterion output."""
    reward_details = judge_details.get('reward')
    if not isinstance(reward_details, dict):
        reward_details = {}
    criteria = []
    for criterion in reward_details.get('criteria') or []:
        if not isinstance(criterion, dict):
            continue
        normalized = dict(criterion)
        passed = _criterion_passed(normalized)
        normalized.setdefault('id', normalized.get('name'))
        normalized['verdict'] = 'pass' if passed else 'fail'
        normalized.setdefault('evidence', normalized.get('reasoning'))
        criteria.append(normalized)
    requirement_status = ['yes' if _criterion_passed(criterion) else 'no' for criterion in criteria]
    criteria_total = len(criteria)
    criteria_passed = sum(status == 'yes' for status in requirement_status)
    rubric_accuracy = criteria_passed / criteria_total if criteria_total else None
    judge_rounds = sum(
        event.get('type') == 'step_finish'
        for trajectory in judge_trajectories
        for event in trajectory.get('events') or []
        if isinstance(event, dict)
    )
    raw_score = reward_details.get('score', fallback_reward)
    try:
        raw_reward = float(raw_score) if raw_score is not None else None
    except (TypeError, ValueError):
        raw_reward = fallback_reward
    judge_unavailable = bool(verifier_error) or not criteria_total
    sample_accuracy = 1.0 if criteria_total and criteria_passed == criteria_total and not verifier_error else 0.0
    failed = [str(criterion.get('name') or '?') for criterion in criteria if not _criterion_passed(criterion)]
    rationale = f'{criteria_passed}/{criteria_total} criteria passed; one-or-nothing score={sample_accuracy:.1f}.' + (
        f' Failed criteria: {", ".join(failed)}.' if failed else ''
    )
    if verifier_error:
        rationale = f'Judge unavailable: {verifier_error}'
    return {
        'score': sample_accuracy,
        'sample_accuracy': sample_accuracy,
        'raw_reward': raw_reward,
        'rubric_accuracy': rubric_accuracy,
        'judge_unavailable': judge_unavailable,
        'rationale': rationale,
        'requirement_status': requirement_status,
        'criteria_passed': criteria_passed if criteria_total else None,
        'criteria_total': criteria_total if criteria_total else None,
        'judge_rounds': judge_rounds or None,
        'criteria': criteria,
    }


def _reward_for_mode(summary: dict[str, Any], reward_mode: Literal['sample_acc', 'rubric_acc']) -> float:
    """Select the training reward while preserving both accuracy metrics."""
    if reward_mode == 'rubric_acc':
        rubric_accuracy = summary.get('rubric_accuracy')
        return float(rubric_accuracy) if rubric_accuracy is not None else 0.0
    return float(summary.get('sample_accuracy', summary.get('score', 0.0)))


def _split_reward_payload(payload: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """Split a reward.json payload into (numeric_rewards, extra_metadata).

    The historical Harbor result typed ``rewards`` as ``dict[str, float | int]``, so a rich
    reward.json like ``{"reward": 0.5, "metadata": {...}}`` (the shape Canyon/Chakra
    reward functions emit) fails validation on the ``metadata`` dict. We keep only the
    numeric top-level keys are preserved as rewards and everything else as metadata.
    """
    numeric: dict[str, float] = {}
    extra: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        if isinstance(value, bool):  # bools are ints in Python; treat as metadata, not score
            extra[key] = value
        elif isinstance(value, (int, float)):
            numeric[key] = value
        else:
            extra[key] = value
    return numeric, extra


def _raise_for_runtime_upload_failures(response: Any, operation: str) -> None:
    """Raise when Nexus reports any failed file upload."""
    results = getattr(response, 'results', None)
    if results is None:
        raise RuntimeError(f'{operation} upload returned no per-file results')
    failed = [result for result in results if not result.success]
    if failed:
        details = '; '.join(f'{result.path}: {result.message or "upload failed"}' for result in failed)
        raise RuntimeError(f'Failed to upload {operation}: {details}')


async def _upload_path_to_runtime(runtime: Any, source_path: Path, target_path: str) -> None:
    """Upload a local file or directory to a container path."""
    from nexus.extensions.tasks.terminal.utils import read_local_directory_to_file_items, read_local_file_to_file_item

    if source_path.is_dir():
        file_items = read_local_directory_to_file_items(
            local_dir=source_path,
            container_base_path=target_path,
        )
    elif source_path.is_file():
        file_items = [read_local_file_to_file_item(source_path, target_path)]
    else:
        raise FileNotFoundError(f'Upload source not found: {source_path}')

    if file_items:
        response = await runtime.upload_file(files=file_items, overwrite=True)
        _raise_for_runtime_upload_failures(response, str(source_path))


def _find_task_root(staged: Path) -> Path:
    """Locate a task payload without consulting Harbor's ``task.toml``."""
    if (staged / 'tests').is_dir():
        return staged
    candidates = sorted(child.parent for child in staged.glob('*/tests') if child.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        listing = ', '.join(sorted(path.name for path in staged.iterdir())[:10]) or '<empty>'
        raise FileNotFoundError(f'Staged task_files contain no tests/ directory at depth 0 or 1 (top level: {listing})')
    raise ValueError(f'Staged task_files contain {len(candidates)} task roots; expected exactly one')


async def _resolve_task_dir(task_input: EnterpriseBenchNativeTaskInput, ctx: Any) -> tuple[Path, Path | None]:
    """Return the local EnterpriseBench task directory and optional staging directory."""
    if task_input.task_dir:
        candidate = Path(task_input.task_dir).expanduser()
        if candidate.exists():
            return candidate, None

    task_files = getattr(ctx, 'task_files', None)
    if task_files is None:
        raise FileNotFoundError(
            f'Task directory not found: {task_input.task_dir!r}, and this run carries no dataset '
            'task_files to materialize it from.'
        )

    staged = Path(tempfile.mkdtemp(prefix='enterprise_task_files_'))
    try:
        # Backends match this pattern with fnmatch, where '*' spans '/'. Using
        # '**/*' would require a slash and silently omit depth-zero task.toml.
        await task_files.download('**', staged)
        task_dir_path = _find_task_root(staged)
    except (Exception, asyncio.CancelledError):
        shutil.rmtree(staged, ignore_errors=True)
        raise
    logger.info(f'Materialized EnterpriseBench task directory from dataset task_files: {task_dir_path}')
    return task_dir_path, staged


def _ensure_relative_payload_path(value: str, *, field_name: str) -> str:
    normalized = value.strip().replace('\\', '/')
    if not normalized or normalized.startswith('/') or '..' in normalized.split('/'):
        raise ValueError(f'{field_name} must be a relative path inside the task payload: {value!r}')
    return normalized


def _build_enterprise_task_spec(
    task_dir: Path,
    task_input: EnterpriseBenchNativeTaskInput,
) -> EnterpriseTaskSpec:
    """Build the Runtime task spec exclusively from Nexus item configuration."""
    tests_dir = task_dir / 'tests'
    if not tests_dir.is_dir():
        raise FileNotFoundError(f'EnterpriseBench task payload has no tests directory: {tests_dir}')
    if not task_input.instruction.strip():
        raise ValueError('EnterpriseBench Native requires task_config.instruction in the Dataset item')
    normalized_judge = normalize_judge_config(task_input.judge_config)
    prompt_name = _ensure_relative_payload_path(
        str(normalized_judge['judge']['prompt_template']),
        field_name='judge_config.judge.prompt_template',
    )
    if not (tests_dir / prompt_name).is_file():
        raise FileNotFoundError(f'EnterpriseBench prompt template not found: {tests_dir / prompt_name}')
    return EnterpriseTaskSpec(
        name=task_input.benchmark_task_name or task_input.instance_id or task_dir.name,
        instruction=task_input.instruction,
        tests_dir=tests_dir,
        solution_dir=task_dir / 'solution',
    )


def _default_upload_target(workdir: str, source_path: Path) -> str:
    """Default upload target is workdir / basename(source)."""
    return posixpath.join(workdir.rstrip('/') or '/', source_path.name)


async def _apply_environment_uploads(
    runtime: Any,
    *,
    task_dir_path: Path,
    workdir: str,
    task_input: EnterpriseBenchNativeTaskInput,
) -> None:
    """Apply explicit ``[environment.upload]`` items and optional workspace overlay."""
    workspace_dir = task_dir_path / 'environment' / 'workspace'
    if not task_input.skip_environment_upload and workspace_dir.exists() and any(workspace_dir.iterdir()):
        logger.info(f'Overlaying workspace files from {workspace_dir} to {workdir}')
        await _upload_path_to_runtime(runtime, workspace_dir, workdir)

    for item in task_input.environment_uploads or []:
        source = str(item.get('source', '')).strip()
        if not source:
            continue
        source_path = (task_dir_path / source).resolve()
        try:
            source_path.relative_to(task_dir_path.resolve())
        except ValueError as exc:
            raise ValueError(f'environment.upload source must stay inside task_dir: {source}') from exc

        target = str(item.get('target', '')).strip() or _default_upload_target(workdir, source_path)
        ensure_directory = bool(item.get('ensure_directory', False))
        if ensure_directory:
            result = await runtime.run_command(
                command=f'mkdir -p {shlex.quote(target)}',
                timeout=60,
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f'Failed to create environment upload target {target}: '
                    f'{(result.stderr or result.stdout or "unknown error").strip()[:1000]}'
                )
            if not source_path.exists():
                logger.info(
                    'Environment upload source %s is absent; keeping empty target directory %s',
                    source_path,
                    target,
                )
                continue
        logger.info(f'Uploading explicit environment item {source_path} to {target}')
        await _upload_path_to_runtime(runtime, source_path, target)


async def _list_workdir_context_files(
    runtime: Any,
    workdir: str,
    *,
    max_files: int,
) -> list[dict[str, str]]:
    """List files already present in the agent workdir as Context Agent references.

    These paths are inside the configured Runtime, not on the Runner host.
    Create Runtime-local links under the standard Context Agent ``workspace``
    directory, then declare their model-visible ``./workspace/...`` URI and
    separate absolute Runtime path. This matches ordinary Context Agent tasks
    without uploading the same files through the Runner.
    """
    root = workdir.rstrip('/') or '/'
    workspace = posixpath.join(root, 'workspace')
    command = (
        f"find -L {shlex.quote(root)} -path {shlex.quote(workspace)} -prune -o -type f -printf '%P\\n' | LC_ALL=C sort"
    )
    response = await runtime.run_command(command=command, timeout=60)
    if response.return_code != 0:
        raise RuntimeError(
            f'Failed to list workdir files under {root}: '
            f'{(response.stderr or response.stdout or "unknown error").strip()[:1000]}'
        )

    relative_paths: list[str] = []
    for line in (response.stdout or '').splitlines():
        relative_path = posixpath.normpath(line.strip())
        if (
            not relative_path
            or relative_path == '.'
            or relative_path.startswith('../')
            or relative_path.startswith('/')
        ):
            continue
        # The find command prunes Context Agent's reserved scratch directory;
        # keep this guard for runtimes whose find implementation differs.
        if relative_path == 'workspace' or relative_path.startswith('workspace/'):
            continue
        relative_paths.append(relative_path)
    if max_files > 0 and len(relative_paths) > max_files:
        logger.warning(
            'EnterpriseBench workdir contains %d files; exposing only the first %d to the agent manifest',
            len(relative_paths),
            max_files,
        )
        relative_paths = relative_paths[:max_files]

    setup_lines = [
        'set -eu',
        f'rm -rf {shlex.quote(workspace)}',
        f'mkdir -p {shlex.quote(workspace)}',
    ]
    context_files: list[dict[str, str]] = []
    created_parents: set[str] = {workspace}
    for relative_path in relative_paths:
        source = posixpath.join(root, relative_path)
        runtime_path = posixpath.join(workspace, relative_path)
        parent = posixpath.dirname(runtime_path)
        if parent not in created_parents:
            setup_lines.append(f'mkdir -p {shlex.quote(parent)}')
            created_parents.add(parent)
        setup_lines.append(f'ln -sfn {shlex.quote(source)} {shlex.quote(runtime_path)}')
        context_files.append({
            'name': relative_path,
            'uri': f'./workspace/{relative_path}',
            'runtime_path': runtime_path,
        })
    setup = await runtime.run_command(command='\n'.join(setup_lines), timeout=60)
    if setup.return_code != 0:
        raise RuntimeError(
            f'Failed to prepare Context Agent workspace links under {workspace}: '
            f'{(setup.stderr or setup.stdout or "unknown error").strip()[:1000]}'
        )
    logger.info('Discovered %d pre-agent workdir file(s) under %s', len(context_files), root)
    return context_files


async def _boot_image_services(runtime: Any, task_input: EnterpriseBenchNativeTaskInput) -> None:
    """Run the image's in-container init (bypassed ENTRYPOINT) and wait for readiness.

    gongfeng runs nexus-runtime as PID 1, so scripts like Canyon's
    ``/omnibus/start_all.sh`` (which start the MCP HTTP servers) never execute. We
    launch boot_command detached (it may ``sleep infinity``), passing boot_env
    explicitly since the image's ENV lines are not applied to run_command, then poll
    readiness derived from task.toml [environment.healthcheck] until the stack is up.
    """
    if not task_input.boot_command:
        return

    # Export boot_env inline and launch detached with `setsid --fork`. Using a
    # trailing shell `&` here is not sufficient: with a preceding `mkdir &&
    # command &`, bash backgrounds the whole AND-list, and that wrapper shell
    # keeps run_command's stdout/stderr pipes open indefinitely.
    env_prefix = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in (task_input.boot_env or {}).items())
    boot_script = f'echo $$ > /logs/boot.pid; {task_input.boot_command}'
    launch = (
        f'mkdir -p /logs; {env_prefix} setsid --fork bash -lc {shlex.quote(boot_script)} '
        f'> /logs/boot.log 2>&1 < /dev/null; '
        f'echo "boot pid=$(cat /logs/boot.pid 2>/dev/null || echo pending)"'
    )
    logger.info(
        f'Booting image services: {task_input.boot_command} (env keys: {list((task_input.boot_env or {}).keys())})'
    )
    boot_timeout = float(getattr(task_input, 'boot_timeout', 7200.0) or 7200.0)
    started = time.monotonic()
    try:
        res = await runtime.run_command(command=launch, timeout=boot_timeout)
    except (TimeoutError, asyncio.TimeoutError) as e:
        message = f'Image services boot launch timed out after {boot_timeout}s: {task_input.boot_command}'
        logger.error(message)
        raise EnterpriseRuntimeRPCError(
            'world_startup.launch', message, elapsed=time.monotonic() - started, limit=boot_timeout
        ) from e
    if getattr(res.status, 'value', res.status) == 'TimeLimitExceeded':
        raise EnterpriseRuntimeStartupTimeoutError(
            'world_startup.launch',
            'Runtime reported command timeout',
            elapsed=time.monotonic() - started,
            limit=boot_timeout,
        )
    if res.return_code != 0:
        raise EnvironmentInitError(f'Image services boot failed: rc={res.return_code} {res.stderr or res.stdout}')
    logger.info(f'Boot launch: rc={res.return_code} out={(res.stdout or "").strip()[:200]}')

    if not task_input.healthcheck_command and not task_input.healthcheck_url:
        return

    start_period = max(float(task_input.healthcheck_start_period_sec or 0.0), 0.0)
    start_period_cmd = f'sleep {shlex.quote(str(start_period))}; ' if start_period > 0 else ''
    probe_timeout = max(float(task_input.healthcheck_timeout_sec or 3.0), 0.1)

    if task_input.healthcheck_command:
        # Poll readiness from inside the container by executing the task-provided
        # command. This supports compound checks copied from
        # [environment.healthcheck].command; rc 0 means ready.
        ready_check = (
            f'{env_prefix} timeout {shlex.quote(str(probe_timeout))}s '
            f'bash -lc {shlex.quote(task_input.healthcheck_command)}'
        )
        deadline_probe = (
            f'{start_period_cmd}'
            f'for i in $(seq 1 {task_input.healthcheck_retries}); do '
            f'if {ready_check}; then echo "READY after ${{i}} tries"; exit 0; fi; '
            f'sleep {task_input.healthcheck_interval_sec}; done; '
            f'echo "NOT READY after {task_input.healthcheck_retries} tries (healthcheck_command failed)"; '
            f'echo "--- boot.log tail ---"; tail -30 /logs/boot.log 2>/dev/null; exit 1'
        )
    else:
        # Poll readiness from inside the container (curl against the healthcheck_url).
        deadline_probe = (
            f'{start_period_cmd}'
            f'for i in $(seq 1 {task_input.healthcheck_retries}); do '
            f'code=$({env_prefix} curl -s -o /dev/null -w "%{{http_code}}" --max-time {shlex.quote(str(probe_timeout))} {shlex.quote(task_input.healthcheck_url)} 2>/dev/null); '
            f'if [ "$code" = "200" ]; then echo "READY after ${{i}} tries"; exit 0; fi; '
            f'sleep {task_input.healthcheck_interval_sec}; done; '
            f'echo "NOT READY after {task_input.healthcheck_retries} tries (last=$code)"; '
            f'echo "--- boot.log tail ---"; tail -30 /logs/boot.log 2>/dev/null; exit 1'
        )
    total_timeout = int(
        start_period + task_input.healthcheck_retries * (probe_timeout + task_input.healthcheck_interval_sec) + 60
    )
    started = time.monotonic()
    async with enterprise_stage('world_startup.healthcheck'):
        res = await runtime.run_command(command=deadline_probe, timeout=total_timeout)
    if getattr(res.status, 'value', res.status) == 'TimeLimitExceeded':
        raise EnterpriseRuntimeStartupTimeoutError(
            'world_startup.healthcheck',
            'Runtime reported command timeout',
            elapsed=time.monotonic() - started,
            limit=total_timeout,
        )
    out = (res.stdout or '').strip()
    if res.return_code == 0:
        logger.info(f'Image services ready: {out[:200]}')
    else:
        message = f'Image services failed healthcheck: {out[:1000]}'
        logger.error(message)
        raise EnvironmentInitError(message)


def _log_verifier_stdout(text: str, *, chars: int = 4000) -> None:
    """Surface Runtime-returned verifier output without downloading its files."""
    if text:
        logger.error('--- last %d chars of verifier test stdout ---\n%s', chars, text[-chars:])
    else:
        logger.error('Verifier produced no test stdout')


def _opencode_events_to_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenCode JSONL events into a compact chat-like trajectory."""
    messages: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        if current.get('content') or current.get('reasoning_content') or current.get('tool_calls'):
            messages.append(current)
        current = None

    for event in events:
        event_type = event.get('type')
        if event_type == 'user':
            flush()
            parts = event.get('parts') or []
            content = '\n'.join(
                str(part.get('text') or '')
                for part in parts
                if isinstance(part, dict) and part.get('type') == 'text' and part.get('text')
            )
            if content:
                messages.append({'role': 'user', 'content': content})
            continue
        if event_type == 'step_start':
            flush()
            current = {'role': 'assistant', 'content': ''}
            continue
        if event_type == 'step_finish':
            flush()
            continue
        if event_type not in {'text', 'reasoning', 'tool_use'}:
            continue
        if current is None:
            current = {'role': 'assistant', 'content': ''}
        part = event.get('part') or {}
        if event_type == 'text':
            current['content'] = f'{current.get("content", "")}{part.get("text") or ""}'
        elif event_type == 'reasoning':
            current['reasoning_content'] = f'{current.get("reasoning_content", "")}{part.get("text") or ""}'
        else:
            state = part.get('state') or {}
            call_id = str(part.get('callID') or part.get('id') or f'judge_call_{len(messages)}')
            tool_name = str(part.get('tool') or '')
            tool_input = state.get('input') or {}
            current.setdefault('tool_calls', []).append({
                'id': call_id,
                'type': 'function',
                'function': {
                    'name': tool_name,
                    'arguments': json.dumps(tool_input, ensure_ascii=False),
                },
            })
            if state.get('output') is not None:
                flush()
                messages.append({
                    'role': 'tool',
                    'tool_call_id': call_id,
                    'name': tool_name,
                    'content': str(state['output']),
                })
    flush()
    return messages


def _load_judge_artifacts(verifier_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse RewardKit details and OpenCode event streams in their current filesystem."""
    verifier_dir = Path(verifier_dir)
    details_path = verifier_dir / 'reward-details.json'
    details: dict[str, Any] = {}
    if details_path.is_file():
        try:
            payload = json.loads(details_path.read_text(encoding='utf-8'))
            if isinstance(payload, dict):
                details = payload
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning('Could not parse RewardKit details from %s: %s', details_path, exc)

    trace_dir = verifier_dir / 'opencode_judge'
    trajectories: list[dict[str, Any]] = []
    if trace_dir.is_dir():
        trace_paths = [
            *sorted(trace_dir.glob('*.attempt-*.jsonl')),
            *sorted(trace_dir.glob('*.latest.jsonl')),
        ]
        seen_payloads: set[str] = set()
        for trace_path in trace_paths:
            raw_trace = trace_path.read_text(encoding='utf-8', errors='replace')
            if raw_trace in seen_payloads:
                continue
            seen_payloads.add(raw_trace)
            events: list[dict[str, Any]] = []
            for line in raw_trace.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
            stem = trace_path.name.split('.attempt-', 1)[0].removesuffix('.latest.jsonl').removesuffix('.jsonl')
            request_path = trace_dir / f'{stem}.request.json'
            stderr_path = trace_dir / f'{stem}.latest.stderr.log'
            request: dict[str, Any] = {}
            if request_path.is_file():
                try:
                    parsed_request = json.loads(request_path.read_text(encoding='utf-8'))
                    if isinstance(parsed_request, dict):
                        request = parsed_request
                except (OSError, json.JSONDecodeError):
                    pass
            messages = _opencode_events_to_messages(events)
            # OpenCode 1.18.x may omit the initial user turn from its
            # ``--format=json`` event stream. The RewardKit adapter persists
            # the prompt and output schema before launching OpenCode, so
            # reconstruct the exact CLI prompt here rather than exposing a
            # trajectory that incorrectly starts with an assistant message.
            if request.get('prompt') and not any(message.get('role') == 'user' for message in messages):
                prompt = str(request['prompt'])
                schema = request.get('schema')
                if isinstance(schema, dict):
                    prompt += (
                        '\n\nYour final message must be exactly one JSON object matching this '
                        'JSON Schema, with no surrounding prose or code fences:\n' + json.dumps(schema)
                    )
                messages.insert(0, {'role': 'user', 'content': prompt})
            trajectories.append({
                'name': trace_path.name,
                'messages': messages,
                'events': events,
                'request': request,
                'stderr': stderr_path.read_text(encoding='utf-8', errors='replace') if stderr_path.is_file() else '',
            })
    from nexus.extensions.tasks.context_agent.enterprisebench_native.judge_runtime import read_progress

    details['runtime_progress'] = read_progress(str(verifier_dir))
    if (verifier_dir / 'direct_responses').is_dir():
        from nexus.extensions.tasks.context_agent.enterprisebench_native.direct_judge import read_capture

        details['direct_responses'] = read_capture(str(verifier_dir))
    return details, trajectories


def _read_runtime_text_file(path: str, *, limit: int | None = None) -> str | None:
    """Read a Runtime-local UTF-8 file for a structured ``runtime.invoke`` result."""
    source = Path(path)
    if not source.is_file():
        return None
    text = source.read_text(encoding='utf-8', errors='replace')
    return text[:limit] if limit is not None else text


def _read_runtime_verifier_result(
    verifier_dir: str = RUNTIME_VERIFIER_DIR,
    *,
    stdout_limit: int = 10_000,
    external_services_diff_path: str = '/judge_workspace/external_services_diff.md',
) -> dict[str, Any]:
    """Parse verifier files inside Runtime and return only structured result data."""
    root = Path(verifier_dir)
    rewards: dict[str, float] = {}
    reward_metadata: dict[str, Any] = {}
    reward_error = ''
    reward_json = root / 'reward.json'
    reward_text = root / 'reward.txt'
    try:
        if reward_json.is_file():
            payload = json.loads(reward_json.read_text(encoding='utf-8') or '{}')
            if not isinstance(payload, dict):
                raise ValueError('EnterpriseBench reward.json must contain a JSON object')
            rewards, reward_metadata = _split_reward_payload(payload)
        elif reward_text.is_file():
            rewards = {'reward': float(reward_text.read_text(encoding='utf-8'))}
        else:
            reward_error = 'EnterpriseBench verifier produced neither reward.json nor reward.txt'
    except Exception as exc:  # noqa: BLE001 - return diagnostics with partial Judge output
        reward_error = f'{type(exc).__name__}: {exc}'

    judge_details, judge_trajectories = _load_judge_artifacts(root)
    return {
        'rewards': rewards,
        'reward_metadata': reward_metadata,
        'reward_error': reward_error,
        'test_stdout': _read_runtime_text_file(str(root / 'test-stdout.txt'), limit=stdout_limit) or '',
        'judge_details': judge_details,
        'judge_trajectories': judge_trajectories,
        'external_services_diff': _read_runtime_text_file(external_services_diff_path),
    }


async def _run_monitored_judge_command(runtime: Any, monitor: JudgeMonitor, **kwargs: Any) -> Any:
    """Watch only compact stage/tool progress; kill the CLI group on any cancellation."""
    command_task = asyncio.create_task(runtime.run_command(**kwargs))
    stop_task = asyncio.create_task(monitor.stopped.wait())
    progress_signature = None
    try:
        while True:
            done, _ = await asyncio.wait({command_task, stop_task}, timeout=20, return_when=asyncio.FIRST_COMPLETED)
            if monitor.failure is not None:
                raise monitor.failure
            if command_task in done:
                return command_task.result()
            try:
                async with async_timeout(3):
                    progress = await runtime.invoke(
                        'nexus.extensions.tasks.context_agent.enterprisebench_native.judge_runtime::read_progress',
                        compact=True,
                    )
                signature = (
                    progress.get('completed_stages'),
                    progress.get('completed_tools'),
                    progress.get('direct_successes', 0),
                )
                if progress.get('direct_abort') == 'judge.length':
                    monitor.length_failure('Runtime-direct Judge length exceeded')
                elif progress.get('direct_abort'):
                    monitor.trip(progress['direct_abort'], 'Runtime-direct model request failure limit reached')
                if progress.get('direct_requests') and monitor.first_request_seconds is None:
                    monitor.first_request_seconds = progress.get('direct_first_request_seconds')
                if signature != progress_signature and any(signature):
                    monitor.progress('runtime_stage_or_tool_completed')
                monitor.runtime_stage = progress.get('last_stage')
                progress_signature = signature
            except Exception:
                logger.debug('Judge progress probe failed', exc_info=True)
            monitor.check_idle()
    finally:
        if not command_task.done():
            try:
                async with async_timeout(10):
                    await runtime.invoke(
                        'nexus.extensions.tasks.context_agent.enterprisebench_native.judge_runtime::stop_command',
                        f'{RUNTIME_VERIFIER_DIR}/command.pid',
                    )
            except Exception:
                logger.warning('Could not stop Judge command process group', exc_info=True)
            command_task.cancel()
        stop_task.cancel()
        await asyncio.gather(command_task, stop_task, return_exceptions=True)


async def _run_enterprise_verifier(
    *,
    task_spec: EnterpriseTaskSpec,
    judge_config: dict[str, Any],
    judge_mcp_url: str,
    runtime: Any,
    skip_tests_upload: bool,
    timeout_sec: float,
    judge_timeout_sec: int,
    rewardkit_max_concurrent_llm: int,
    rewardkit_max_concurrent_agent: int,
    judge_client: Any | None = None,
    monitor: JudgeMonitor | None = None,
    execution_mode: str = 'host',
    api_records: ResponsesCapture | None = None,
    deadline: float | None = None,
) -> tuple[dict[str, float], dict[str, Any], str, dict[str, Any], list[dict[str, Any]], str, str | None]:
    """Run the SyntaraOffice verifier directly through Nexus Runtime APIs."""

    rewards: dict[str, float] = {}
    reward_metadata: dict[str, Any] = {}
    judge_handler = None
    judge_proxy_id = ''
    direct_id = ''
    proxy_env: dict[str, str] = {}
    started = time.monotonic()
    deadline = deadline if deadline is not None else started + judge_timeout_sec
    monitor = monitor or JudgeMonitor()
    wrapped_client = (
        _YamlAuthoritativeJudgeClient(judge_client, deadline, monitor, api_records=api_records)
        if judge_client is not None and execution_mode == 'host'
        else None
    )
    monitor.record('verifier_setup_start')
    try:
        if execution_mode == 'runtime_direct':
            from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient
            from nexus.extensions.tasks.context_agent.enterprisebench_native.direct_judge import seal

            if not isinstance(judge_client, OpenAIResponsesClient):
                raise ValueError('runtime_direct currently requires an openai_responses Judge client')
            args = judge_client.get_effective_request_args()
            forbidden_fields = {'input', 'messages', 'tools', 'stream', 'extra_headers', 'extra_query'}
            forbidden = forbidden_fields & args.keys()
            if forbidden:
                raise ValueError(f'runtime_direct does not allow configured request fields: {sorted(forbidden)}')
            extra = args.pop('extra_body', {})
            if extra:
                if not isinstance(extra, dict) or forbidden_fields.intersection(extra):
                    raise ValueError('Unsupported runtime_direct extra_body')
                args = {**extra, **args}
            prepared = await runtime.invoke(
                'nexus.extensions.tasks.context_agent.enterprisebench_native.direct_judge::prepare',
            )
            direct_id = prepared['id']
            local_key = uuid.uuid4().hex
            configuration = {
                'endpoint': judge_client.endpoint,
                'api_key': judge_client.client.api_key,
                'local_key': local_key,
                'request_args': args,
                'timeout': max(0.001, deadline - time.monotonic()),
                'max_attempts': judge_client.max_retries,
                'max_failures': monitor.max_failures,
                'max_rate_limits': monitor.max_rate_limits,
            }
            encrypted = seal(prepared['public_key'], configuration)
            proxy = await runtime.invoke(
                'nexus.extensions.tasks.context_agent.enterprisebench_native.direct_judge::configure',
                direct_id,
                encrypted,
            )
            configuration.clear()
            proxy_env['NEXUS_JUDGE_PROXY_URL'] = proxy['base_url']
            proxy_env['NEXUS_JUDGE_PROXY_API_KEY'] = local_key
            monitor.progress('direct_proxy_ready')
        elif judge_client is not None:
            from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient

            if not isinstance(judge_client, OpenAIResponsesClient):
                raise ValueError('EnterpriseBench host Judge requires openai_responses for original output replay')
            from nexus.extensions.tasks.context_agent.enterprisebench_native.judge_llm_handler import (
                JudgeLLMRequestHandler,
            )

            await runtime.ws_connect()
            judge_handler = JudgeLLMRequestHandler(
                judge_deadline=deadline,
                runtime=runtime,
                openai_client=cast(OpenAICompatibleClient, wrapped_client),
                preserve_thinking=False,  # Runtime replays full original output; no second history cache.
            )
            judge_handler.register()
            proxy = await runtime.invoke(
                'nexus.extensions.tasks.context_agent.enterprisebench_native.judge_proxy::start_responses_proxy',
                request_timeout=max(0.001, min(timeout_sec, deadline - time.monotonic())),
            )
            judge_proxy_id = str(proxy['proxy_id'])
            proxy_env['NEXUS_JUDGE_PROXY_URL'] = str(proxy['base_url'])
            monitor.progress('proxy_ready')
            logger.info('Started Runtime-local Responses proxy for EnterpriseBench judge at %s', proxy['base_url'])

        if not skip_tests_upload:
            await _upload_path_to_runtime(runtime, task_spec.tests_dir, RUNTIME_TESTS_DIR)

        # The Dataset item is the single source of truth for Judge behavior.
        # Always replace any authored/legacy judge.toml and OpenCode provider
        # config after uploading the payload, then use the Nexus-owned wrapper
        # instead of the task's test.sh.
        normalized_judge = normalize_judge_config(judge_config)
        judge_output_limit, judge_effort = _judge_launch_settings(judge_client)
        rewardkit_max_concurrent_llm = _positive_int(
            rewardkit_max_concurrent_llm,
            field_name='rewardkit_max_concurrent_llm',
        )
        rewardkit_max_concurrent_agent = _positive_int(
            rewardkit_max_concurrent_agent,
            field_name='rewardkit_max_concurrent_agent',
        )
        prompt_name = _ensure_relative_payload_path(
            str(normalized_judge['judge']['prompt_template']),
            field_name='judge_config.judge.prompt_template',
        )
        prompt_path = task_spec.tests_dir / prompt_name
        if not prompt_path.is_file():
            raise FileNotFoundError(f'EnterpriseBench prompt template not found: {prompt_path}')
        generated_files = [
            FileItem(
                path=f'{RUNTIME_TESTS_DIR}/judge.toml',
                content=render_judge_toml(normalized_judge, timeout=judge_timeout_sec),
                encoding='utf8',
            ),
            FileItem(
                path=f'{RUNTIME_TESTS_DIR}/{prompt_name}',
                content=render_prompt_template(prompt_path.read_text(encoding='utf-8')),
                encoding='utf8',
            ),
            FileItem(
                path=f'{RUNTIME_TESTS_DIR}/opencode.json',
                content=render_opencode_config(mcp_url=judge_mcp_url, output_limit=judge_output_limit),
                encoding='utf8',
            ),
            FileItem(
                path=f'{RUNTIME_TESTS_DIR}/rewardkit_opencode_agent.py',
                content=(Path(__file__).resolve().parent / 'rewardkit_opencode_agent.py').read_text(encoding='utf-8'),
                encoding='utf8',
            ),
            FileItem(
                path=f'{RUNTIME_TESTS_DIR}/judge_runtime.py',
                content=(Path(__file__).resolve().parent / 'judge_runtime.py').read_text(encoding='utf-8'),
                encoding='utf8',
            ),
            FileItem(
                path=f'{RUNTIME_TESTS_DIR}/test.sh',
                content=(Path(__file__).resolve().parent / 'run_rewardkit.sh').read_text(encoding='utf-8'),
                encoding='utf8',
            ),
        ]
        response = await runtime.upload_file(files=generated_files, overwrite=True)
        _raise_for_runtime_upload_failures(response, 'Nexus-owned verifier configuration')
        monitor.progress('verifier_config_uploaded')

        # Nexus-owned routing values win over any legacy verifier env copied
        # from the source task, so the generated Judge configuration cannot be
        # redirected back to an authored model or endpoint.
        nexus_verifier_env = {
            **VERIFIER_ENV,
            'REWARDKIT_MAX_CONCURRENT_LLM': str(rewardkit_max_concurrent_llm),
            'REWARDKIT_MAX_CONCURRENT_AGENT': str(rewardkit_max_concurrent_agent),
        }
        if judge_effort:
            nexus_verifier_env['REWARDKIT_OPENCODE_VARIANT'] = judge_effort
        merged_env = {**nexus_verifier_env, **proxy_env}
        command = (
            f'mkdir -p {shlex.quote(RUNTIME_VERIFIER_DIR)}; '
            f'chmod +x {shlex.quote(RUNTIME_TESTS_DIR + "/test.sh")}; '
            f'python3 {RUNTIME_TESTS_DIR}/judge_runtime.py run {RUNTIME_VERIFIER_DIR}/command.pid '
            f'timeout --foreground --signal=TERM --kill-after=5s {max(0.001, min(timeout_sec, deadline - time.monotonic())):.3f}s '
            f'bash {shlex.quote(RUNTIME_TESTS_DIR + "/test.sh")} '
            f'> {shlex.quote(RUNTIME_TEST_STDOUT)} 2>&1'
        )
        monitor.progress('verifier_command_start')
        command_error = ''
        result = None
        try:
            result = await _run_monitored_judge_command(
                runtime,
                monitor,
                command=command,
                timeout=max(0.001, min(timeout_sec, deadline - time.monotonic())),
                extra_env=merged_env,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise EnterpriseRuntimeRPCError(
                'judge.command',
                str(exc),
                elapsed=time.monotonic() - started,
                limit=timeout_sec,
            ) from exc
        except EnterpriseJudgeError:
            raise
        except Exception as exc:  # noqa: BLE001 - still retrieve partial Runtime-side Judge output
            command_error = f'{type(exc).__name__}: {exc}'

        monitor.record('verifier_command_end')
        parsed = await runtime.invoke(_read_runtime_verifier_result, RUNTIME_VERIFIER_DIR, stdout_limit=10_000)
        if not isinstance(parsed, dict):
            raise RuntimeError('Runtime verifier parser returned invalid data')
        rewards = dict(parsed.get('rewards') or {})
        reward_metadata = dict(parsed.get('reward_metadata') or {})
        test_stdout = str(parsed.get('test_stdout') or '')
        judge_details = dict(parsed.get('judge_details') or {})
        judge_trajectories = list(parsed.get('judge_trajectories') or [])
        external_services_diff = parsed.get('external_services_diff')
        if external_services_diff is not None:
            external_services_diff = str(external_services_diff)

        direct_status = (judge_details.get('direct_responses') or {}).get('status') or {}
        if direct_status.get('abort_reason'):
            error_cls = (
                EnterpriseJudgeLengthError if direct_status['abort_reason'] == 'judge.length' else EnterpriseJudgeError
            )
            raise error_cls(
                direct_status['abort_reason'],
                'Runtime-direct failure limit reached',
                elapsed=time.monotonic() - started,
            )
        if command_error:
            verifier_error = command_error
        elif result is not None and (
            getattr(result.status, 'value', result.status) == 'TimeLimitExceeded' or result.return_code in {124, 137}
        ):
            raise EnterpriseJudgeTimeoutError(
                'judge.command',
                'verifier command deadline exceeded',
                elapsed=time.monotonic() - started,
                limit=min(timeout_sec, judge_timeout_sec),
            )
        elif result is not None and result.return_code != 0:
            detail = (result.stderr or result.stdout or 'no output').strip()[:2000]
            verifier_error = f'EnterpriseBench verifier exited with code {result.return_code}: {detail}'
        else:
            verifier_error = str(parsed.get('reward_error') or '')
        if verifier_error:
            _log_verifier_stdout(test_stdout)
    finally:
        monitor.record('verifier_cleanup')
        if direct_id:
            try:
                async with async_timeout(10):
                    await runtime.invoke(
                        'nexus.extensions.tasks.context_agent.enterprisebench_native.direct_judge::stop',
                        direct_id,
                    )
            except Exception:
                logger.warning('Could not stop runtime-direct Judge relay', exc_info=True)
        if judge_handler is not None:
            judge_handler.unregister()
        if wrapped_client is not None:
            wrapped_client.cancel_pending()
        if judge_proxy_id and runtime is not None:
            try:
                async with async_timeout(10):
                    await runtime.invoke(
                        'nexus.extensions.tasks.context_agent.enterprisebench_native.judge_proxy::stop_responses_proxy',
                        judge_proxy_id,
                    )
            except Exception:
                logger.warning('Failed to stop Runtime-local EnterpriseBench judge proxy', exc_info=True)

    return (
        rewards,
        reward_metadata,
        test_stdout,
        judge_details,
        judge_trajectories,
        verifier_error,
        external_services_diff,
    )


AGENT_DELIVERABLE_DIR = '/tmp/agent_deliverable'


def _msg_text(content: Any) -> str:
    """Flatten a chat-message ``content`` (str or multimodal part list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get('text') or p.get('content') or '')
            else:
                parts.append(str(p))
        return '\n'.join(x for x in parts if x)
    return '' if content is None else str(content)


def _extract_answer_and_transcript(agent_output: Any) -> tuple[str, str]:
    """Pull (final answer, full transcript) off the agent's ``main_trajectory``.

    ``main_trajectory`` (WithMainTrajectory mixin, nexus.core.agent) is the
    standard list[dict] conversation the nexus agents expose for evaluation.
    Mirrors code_mt scoring.extract_agent_trajectory: answer = last non-empty
    assistant message; transcript = whole conversation (incl. tool calls).
    Degrades to ('', '') when no trajectory is available.
    """
    traj = getattr(agent_output, 'main_trajectory', None)
    if traj is None and isinstance(agent_output, dict):
        traj = agent_output.get('main_trajectory')
    messages = traj or []

    declared = (
        agent_output.get('final_answer')
        if isinstance(agent_output, dict)
        else getattr(agent_output, 'final_answer', None)
    )
    answer = _msg_text(declared) if declared is not None else ''
    if declared is None:
        for m in reversed(messages):
            if isinstance(m, dict) and m.get('role') == 'assistant' and not m.get('tool_calls'):
                text = _msg_text(m.get('content'))
                if text.strip():
                    answer = text
                    break

    lines: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role', '?')
        text = _msg_text(m.get('content'))
        if text.strip():
            lines.append(f'[{role}]\n{text}')
        for tc in m.get('tool_calls') or []:
            fn = tc.get('function', {}) if isinstance(tc, dict) else {}
            name = fn.get('name', '') if isinstance(fn, dict) else ''
            args = fn.get('arguments', '') if isinstance(fn, dict) else ''
            lines.append(f'[{role}:tool_call {name}]\n{args}')
    transcript = '\n\n'.join(lines)
    return answer, transcript


async def _persist_agent_deliverable(runtime: Any, agent_output: Any, *, deadline: float) -> None:
    """Write the agent's final answer + transcript into the main sandbox.

    The Runtime-side verifier only sees the sandbox filesystem, but code_mt-style
    rubric/QA judges grade the agent's runtime *deliverable* (final answer or
    whole trajectory), which otherwise lives only in host memory. Persisting it
    to {AGENT_DELIVERABLE_DIR}/answer.txt + trajectory.txt lets an in-sandbox
    judge.sh grade it (and enables joint deliverable+trajectory judging). Written
    to /tmp so the shared-verifier reset (which only empties /logs/verifier +
    /tests) preserves it. Failures abort verification so an absent answer is not graded as a model error.
    """
    started = time.monotonic()
    remaining = max(0.0, deadline - started)
    try:
        async with enterprise_stage('judge.deliverable', remaining, EnterpriseJudgeTimeoutError):
            answer, transcript = _extract_answer_and_transcript(agent_output)
            logger.info(
                'Persisting agent deliverable: answer=%d chars, transcript=%d chars, remaining=%.3fs',
                len(answer),
                len(transcript),
                remaining,
            )
            await runtime.invoke(
                'nexus.extensions.tasks.context_agent.enterprisebench_native.deliverables::write_deliverables',
                answer,
                transcript,
            )
        logger.info('Persisted agent deliverable: answer=%d chars, transcript=%d chars', len(answer), len(transcript))
    except EnterpriseStageError:
        # Preserve typed stage/limit details rather than nesting the same failure.
        raise
    except Exception as exc:  # noqa: BLE001 - retain the stage for non-timeout write failures
        raise EnterpriseJudgeError(
            'judge.deliverable',
            str(exc),
            elapsed=time.monotonic() - started,
            limit=remaining,
        ) from exc


@task_registry.register('context_agent_enterprisebench_native_task')
@report_task_termination
async def enterprisebench_native_task(task_input: EnterpriseBenchNativeTaskInput) -> EnterpriseBenchNativeTaskOutput:
    """Run one SyntaraOffice task with Nexus-native environment and verifier orchestration."""

    logger.info('Starting EnterpriseBench task: %s', task_input.instance_id)

    ctx = require_context()
    if not ctx.runtime_provider:
        raise RuntimeError('EnterpriseBench task requires a runtime provider (gongfeng)')
    ctx.task_config.pop('WEDATA_KEY', None)
    ctx.task_files = bind_task_wedata_key(ctx.task_files, task_input.WEDATA_KEY)

    async with enterprise_stage('task_files'):
        task_dir_path, task_files_staging = await _resolve_task_dir(task_input, ctx)
    stage = 'runtime_startup'
    stage_started = time.monotonic()
    try:
        task_spec = _build_enterprise_task_spec(task_dir_path, task_input)
        if task_input.verifier_mode != 'shared':
            raise ValueError('Nexus-native EnterpriseBench supports verifier_mode="shared" only')
        if task_input.verifier_tests_prebaked:
            raise ValueError('Nexus-native EnterpriseBench does not support verifier_tests_prebaked=true')
        logger.info('Loaded EnterpriseBench task: %s', task_spec.name)
        instruction = task_spec.instruction

        run_id = str(ctx.run_id or uuid.uuid4().hex)
        agent_output: Any = None
        workspace_artifact: dict[str, Any] | None = None
        evidence_artifact: dict[str, Any] | None = None
        external_services_diff: str | None = None
        rewards: dict[str, float] = {}
        reward_metadata: dict[str, Any] = {}
        test_stdout = ''
        verifier_error = ''
        judge_details: dict[str, Any] = {}
        judge_trajectories: list[dict[str, Any]] = []
        summary: dict[str, Any] = _summarize_judge(
            {},
            [],
            verifier_error='Verifier did not run',
            fallback_reward=None,
        )
        reward = 0.0
        judge_failure: Exception | None = None
        evaluation_skipped = False

        async with enterprise_stage('runtime_startup', task_input.boot_timeout, EnterpriseRuntimeStartupTimeoutError):
            runtime = await ctx.get_runtime('main')
        logger.info('Got main Nexus runtime for EnterpriseBench task')
        workdir = checked_workspace_root(task_input.workdir or '/app')
        reset_dirs = [RUNTIME_VERIFIER_DIR, RUNTIME_ARTIFACTS_DIR]
        if not task_input.skip_tests_upload:
            reset_dirs.append(RUNTIME_TESTS_DIR)
        quoted_reset_dirs = ' '.join(shlex.quote(path) for path in reset_dirs)
        await runtime.run_command(
            command=f'rm -rf {quoted_reset_dirs} && mkdir -p {shlex.quote(workdir)} {quoted_reset_dirs}',
            timeout=60,
        )
        await _apply_environment_uploads(
            runtime,
            task_dir_path=task_dir_path,
            workdir=workdir,
            task_input=task_input,
        )
        async with enterprise_stage('world_startup', task_input.boot_timeout, EnterpriseRuntimeStartupTimeoutError):
            await _boot_image_services(runtime, task_input)

        # Prepare the manifest before fixing ownership: manifest construction
        # recreates workspace/ and would otherwise reset it to root:root 0755.
        context_files = None
        if task_input.expose_workdir_files and not task_input.run_oracle and not task_input.run_nop:
            context_files = await _list_workdir_context_files(
                runtime,
                workdir,
                max_files=max(0, int(task_input.workdir_context_max_files)),
            )
        permissions = await runtime.invoke(
            'nexus.extensions.tasks.context_agent.enterprisebench_native.workspace_permissions::prepare_workspace_permissions',
            workdir,
        )
        logger.info('Prepared writable EnterpriseBench directories: %s', permissions)
        ctx.tool_state['execution_identity'] = {'uid': permissions['uid'], 'gid': permissions['gid']}
        isolation = await runtime.invoke(
            'nexus.extensions.tasks.context_agent.enterprisebench_native.workspace_permissions::protect_service_data',
            workdir,
            permissions['uid'],
            permissions['gid'],
        )
        logger.info('Protected EnterpriseBench service data: %s', isolation)
        await runtime.invoke(
            'nexus.extensions.tasks.context_agent.enterprisebench_native.deliverables::prepare_deliverable_directory',
        )

        stage = 'agent'
        stage_started = time.monotonic()
        ctx.task_config['working_dir'] = workdir
        if task_input.run_oracle:
            async with enterprise_stage('agent.oracle', task_input.agent_timeout, EnterpriseAgentTimeoutError):
                await _run_oracle(runtime, task_spec, timeout_sec=task_input.agent_timeout)
        elif task_input.run_nop:
            logger.info('RUN_NOP enabled: skipping agent and oracle before verification')
        else:
            agent_kwargs: dict[str, Any] = {
                'user_prompt': instruction,
                'working_dir': workdir,
            }
            if context_files is not None:
                agent_kwargs['context_files'] = context_files
            async with enterprise_stage('agent', task_input.agent_timeout, EnterpriseAgentTimeoutError):
                agent_output = await task_input.run_agent(**agent_kwargs)

        if task_input.artifact_root:
            workspace_artifact = await persist_workspace_evidence(
                runtime,
                artifact_root=task_input.artifact_root,
                run_id=run_id,
                task_id=task_input.instance_id or task_spec.name,
                final_root=workdir,
                max_files=task_input.artifact_max_files,
                max_bytes=task_input.artifact_max_bytes,
            )

        stage = 'judge'
        stage_started = time.monotonic()
        final_answer, _transcript = _extract_answer_and_transcript(agent_output)
        skip_reason = judge_skip_reason(agent_output, final_answer)
        if skip_reason is not None:
            evaluation_skipped = True
            summary = _skipped_judge_summary(skip_reason)
            reward = 0.0
            rewards = {'reward': reward}
            logger.info('%s score=0.0', skip_reason)
        else:
            monitor = JudgeMonitor(
                max_failures=task_input.judge_max_consecutive_failures,
                max_rate_limits=task_input.judge_max_consecutive_rate_limits,
                idle_timeout=task_input.judge_no_progress_timeout,
            )
            api_records = ResponsesCapture()
            judge_client = ctx.get_openai_client(task_input.judge_client_name) if task_input.judge_client_name else None
            judge_deadline = stage_started + task_input.judge_timeout
            try:
                # Sequential scopes share one deadline. Nesting equal-deadline
                # timers can hide judge.deliverable behind an outer judge timeout.
                monitor.record('deliverable_start')
                if not task_input.run_nop and not task_input.run_oracle:
                    await _persist_agent_deliverable(runtime, agent_output, deadline=judge_deadline)
                monitor.progress('deliverable_ready')
                async with enterprise_stage(
                    'judge',
                    max(0.0, judge_deadline - time.monotonic()),
                    EnterpriseJudgeTimeoutError,
                ):
                    (
                        rewards,
                        reward_metadata,
                        test_stdout,
                        judge_details,
                        judge_trajectories,
                        verifier_error,
                        external_services_diff,
                    ) = await _run_enterprise_verifier(
                        task_spec=task_spec,
                        judge_config=task_input.judge_config,
                        judge_mcp_url=task_input.judge_mcp_url,
                        runtime=runtime,
                        skip_tests_upload=task_input.skip_tests_upload,
                        timeout_sec=task_input.test_timeout,
                        judge_timeout_sec=task_input.judge_timeout,
                        rewardkit_max_concurrent_llm=task_input.rewardkit_max_concurrent_llm,
                        rewardkit_max_concurrent_agent=task_input.rewardkit_max_concurrent_agent,
                        judge_client=judge_client,
                        monitor=monitor,
                        execution_mode=task_input.judge_execution_mode,
                        api_records=api_records,
                        deadline=judge_deadline,
                    )
            except Exception as exc:  # noqa: BLE001 - preserve partial judge evidence
                judge_failure = exc
                verifier_error = f'{type(exc).__name__}: {exc}'
                rewards = {}
                reward_metadata = {}
                try:
                    async with async_timeout(10):
                        partial = await runtime.invoke(_read_runtime_verifier_result, RUNTIME_VERIFIER_DIR)
                except Exception:
                    partial = {}
                if isinstance(partial, dict):
                    rewards = dict(partial.get('rewards') or {})
                    reward_metadata = dict(partial.get('reward_metadata') or {})
                    test_stdout = str(partial.get('test_stdout') or '')
                    judge_details = dict(partial.get('judge_details') or {})
                    judge_trajectories = list(partial.get('judge_trajectories') or [])
                    external_services_diff = partial.get('external_services_diff')
                    if external_services_diff is not None:
                        external_services_diff = str(external_services_diff)
                logger.exception('EnterpriseBench verifier failed after agent completion: %s', exc)

            if task_input.judge_execution_mode == 'host':
                judge_details['host_responses'] = api_records.data
                judge_details['tools'] = api_records.tools
            else:
                judge_details['host_progress'] = monitor.snapshot()
            span_id = get_current_span_id()
            if span_id:
                update_span(span_id, attributes={'judge_progress': monitor.snapshot()})
            if not verifier_error:
                expected = {
                    criterion['name'] for criterion in normalize_judge_config(task_input.judge_config)['criteria']
                }
                actual = (judge_details.get('reward') or {}).get('criteria') or []
                names = [criterion.get('name') for criterion in actual if isinstance(criterion, dict)]
                if set(names) != expected or len(names) != len(expected) or len(actual) != len(expected):
                    verifier_error = 'Verifier returned missing, duplicate, or unexpected criteria'
                elif any(
                    not isinstance(criterion.get('value'), (int, float)) or not math.isfinite(float(criterion['value']))
                    for criterion in actual
                ):
                    verifier_error = 'Verifier returned invalid criterion values'

            raw_reward_value = rewards.get('reward')
            summary = _summarize_judge(
                judge_details,
                judge_trajectories,
                verifier_error=verifier_error,
                fallback_reward=float(raw_reward_value) if raw_reward_value is not None else None,
            )
            if task_input.judge_execution_mode == 'host':
                # Count steps before discarding the redundant chat/event projection.
                # Keep CLI stderr for failures that happen before any API request.
                stderr = [str(tr.get('stderr')) for tr in judge_trajectories if tr.get('stderr')]
                if stderr:
                    judge_details['judge_stderr'] = stderr
                judge_trajectories = []
            reward = _reward_for_mode(summary, task_input.reward_mode)
            rewards = dict(rewards)
            rewards['reward'] = reward

        reward, extra_fields = apply_tool_name_resolution_score(
            reward,
            agent_output,
            enabled=task_input.zero_score_on_tool_name_resolution_error,
            eligible=not (summary['judge_unavailable'] or evaluation_skipped),
        )
        rewards = {**rewards, 'reward': reward}
        score, extra_fields = apply_bad_pattern_score(
            reward,
            agent_output,
            task_input.tool_call_bad_pattern_policy,
            extra_fields=extra_fields,
            termination_reason='REWARD_ERROR'
            if summary['judge_unavailable']
            else _agent_termination_reason(agent_output),
        )

        if task_input.artifact_root:
            final_answer, _transcript = _extract_answer_and_transcript(agent_output)
            trajectory = getattr(agent_output, 'main_trajectory', None) or []
            if isinstance(agent_output, dict):
                trajectory = agent_output.get('main_trajectory') or trajectory
            records = persist_run_records(
                artifact_root=task_input.artifact_root,
                run_id=run_id,
                final_answer=final_answer,
                agent_trajectory=list(trajectory),
                judge_details=judge_details,
                judge_trajectories=judge_trajectories,
                external_services_diff=external_services_diff,
                result_summary={
                    'task_id': task_input.instance_id or task_spec.name,
                    'score': score,
                    'reward_mode': task_input.reward_mode,
                    'extra_fields': extra_fields,
                    'raw_reward': summary['raw_reward'],
                    'sample_accuracy': summary['sample_accuracy'],
                    'rubric_accuracy': summary['rubric_accuracy'],
                    'criteria_passed': summary['criteria_passed'],
                    'criteria_total': summary['criteria_total'],
                    'judge_rounds': summary['judge_rounds'],
                    'judge_unavailable': summary['judge_unavailable'],
                    'rationale': summary['rationale'],
                },
            )
            evidence_artifact = {
                'workspace': workspace_artifact,
                'records': records,
            }
        logger.info(
            'EnterpriseBench result: score=%s raw_reward=%s criteria=%s/%s',
            score,
            summary['raw_reward'],
            summary['criteria_passed'],
            summary['criteria_total'],
        )
        agent_termination_reason = _agent_termination_reason(agent_output)
        termination_reason = 'REWARD_ERROR' if summary['judge_unavailable'] else agent_termination_reason
        output = EnterpriseBenchNativeTaskOutput(
            extra_fields=extra_fields,
            final_answer=final_answer or None,
            score=score,
            reward=reward,
            rewards=rewards,
            raw_reward=summary['raw_reward'],
            sample_accuracy=summary['sample_accuracy'],
            rubric_accuracy=summary['rubric_accuracy'],
            reward_metadata=reward_metadata,
            test_stdout=test_stdout,
            verifier_error=verifier_error,
            judge_unavailable=summary['judge_unavailable'],
            evaluation_skipped=evaluation_skipped,
            rationale=summary['rationale'],
            requirement_status=summary['requirement_status'],
            criteria_passed=summary['criteria_passed'],
            criteria_total=summary['criteria_total'],
            judge_rounds=summary['judge_rounds'],
            criteria=summary['criteria'],
            workspace_artifact=workspace_artifact,
            evidence_artifact=evidence_artifact,
            external_services_diff=external_services_diff,
            judge_details=judge_details,
            judge_trajectories=judge_trajectories,
            termination_reason=termination_reason,
        )
        if summary['judge_unavailable']:
            # Keep partial evidence on the Task span; runner intentionally sees
            # an exception (task_output=null) while retaining agent_output.
            span_id = get_current_span_id()
            if span_id:
                update_span(span_id, attributes={'partial_task_output': output.model_dump()})
            if isinstance(judge_failure, (EnterpriseJudgeError, EnterpriseRuntimeRPCError)):
                raise judge_failure
            raise EnterpriseJudgeError(
                'judge',
                verifier_error or 'Verifier returned no valid criteria',
                elapsed=time.monotonic() - stage_started,
                limit=task_input.judge_timeout,
            ) from judge_failure
        return output
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise EnterpriseRuntimeRPCError(
            stage,
            str(exc),
            elapsed=time.monotonic() - stage_started,
        ) from exc

    finally:
        if task_files_staging is not None:
            shutil.rmtree(task_files_staging, ignore_errors=True)


async def _run_oracle(runtime: Any, task_spec: EnterpriseTaskSpec, timeout_sec: int) -> None:
    """Run an optional SyntaraOffice oracle solution directly in the Runtime."""

    solution_dir = task_spec.solution_dir
    if not solution_dir.exists():
        raise FileNotFoundError(f'No solution directory found at {solution_dir}')
    await runtime.run_command(command='mkdir -p /solution', timeout=30)
    await _upload_path_to_runtime(runtime, solution_dir, '/solution')
    result = await runtime.run_command(
        command='chmod +x /solution/solve.sh && bash /solution/solve.sh',
        timeout=timeout_sec,
    )
    if result.return_code != 0:
        logger.warning('Oracle solution failed with code %s: %s', result.return_code, result.stderr)
