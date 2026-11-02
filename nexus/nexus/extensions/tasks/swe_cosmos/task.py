"""Host-side orchestration for the SWE-Cosmos bugmaker pipeline.

The task acquires a single root sandbox built from the SWE-Cosmos image, uploads the per-repo "repo dict", registers a host-side LLM request handler (so the pipeline's LLM calls are forwarded out of the sandbox), then runs the whole pipeline inside one long ``runtime.invoke`` job. When the pipeline finishes it downloads the artifacts and returns the funnel counts plus a ``done`` flag.

Per-repo output lands under ``output_dir/<repo_name>/``: ``run_bugmaker.log`` (the tee'd pipeline log), ``<repo_name>.<bugmaker_strategy>_instances.jsonl`` (the final issuegen jsonl, absent when the run exits before producing it), and ``result.json`` (the stats summary). ``result.json`` is always written — the output dir is created up front and a ``"Error"`` summary is emitted even when the run aborts before or during the in-sandbox pipeline.

LLM forwarding: the sandbox-side target starts an OpenAI-compatible proxy; the host :class:`LLMRequestHandler` answers ``llm_request`` events using the configured ``main`` OpenAI client. The pipeline therefore needs no direct network path to the LLM gateway.

Sandbox max-runtime: there is currently NO nexus/Gongfeng provider parameter to set a hard wall-clock TTL on a sandbox (the Gongfeng ``/runtime/start`` payload carries no lifetime field; ``purpose`` is the only lifetime-adjacent label and its recycle policy is backend-defined). The host-side ceiling you CAN control is ``pipeline_timeout`` below: it bounds both the in-sandbox ``run_command`` and the per-invoke ``request_timeout`` the client waits on. Raise it for long runs.
"""

import asyncio
import base64
import json
import logging
import random

from pathlib import Path
from typing import Any, Literal

from nexus.core import TaskRunInput, TaskRunOutput, WithScore, require_context, task_registry
from nexus.extensions.tasks.swe_cosmos.utils import (
    BUGMAKER_CLIENT,
    ISSUEGEN_CLIENT,
    RoutingLLMRequestHandler,
    model_routing_name,
    resolve_question_mark_path,
)
from nexus.runtime.executors.models.file import FileItem
from nexus.runtime.models import InvokeRequest


logger = logging.getLogger(__name__)

_TARGET = 'nexus.extensions.tasks.swe_cosmos.runtime_setup::run_pipeline'


class BugmakerPipelineInput(TaskRunInput):
    """Business inputs for one repo's bugmaker pipeline run."""

    # Identity + IO
    repo_name: str
    repo_dict_path: str  # host path to the repo dict JSON, uploaded into the sandbox
    output_dir: str  # host dir that receives a per-repo subdir of artifacts

    # Sandbox-side layout (paths inside the SWE-Cosmos image)
    swe_cosmos_dir: str = '/root/workspace/SWE-Cosmos'
    sandbox_output_dir: str = '/root/workspace/SWE-Cosmos/outputs'
    runtime_name: str = 'main'

    # Pipeline knobs (forwarded to scripts/test_bugmaker_pipeline.sh as env vars)
    # Chosen from: 'function_rewrite' (default) | 'function_compose'
    bugmaker_strategy: str = 'function_rewrite'
    max_bugs: int = 100
    max_workers: int = 4
    seed: int = 3407
    # 1 = 沙盒侧不重试。litellm 对任何异常都重试（含 host 为「别重试」而特意返回的 400），且会与 host profile 的 max_retries 相乘。
    num_retries: int = 1

    # function_rewrite: optional entity file fed to bugmaker as --input-path.
    # Host path; uploaded into the sandbox when provided.
    entity_path: str = ''

    # function_compose: prior bug-data file fed to bugmaker as --input-path
    # (REQUIRED for this strategy). Host path; uploaded into the sandbox.
    bug_data_path: str = ''
    num_patches: int = 2
    compose_criteria: str = 'basic'
    conflict_strategy: str = 'reject'
    compose_max_reuse: float = 1.33

    # Child-sandbox (Gongfeng) execution backend for SWE-Cosmos' own harness. These are exported into the pipeline's environment and consumed by scripts/lib_env_args.sh + swecosmos' GongfengEnvironment.
    environment_type: str = 'gongfeng'
    gongfeng_api_base_url: str = 'http://hyrl-sandbox.prod.woa.com:8052'
    gongfeng_api_key: str = ''
    gongfeng_runtime_image: str = 'current'
    gongfeng_cpu: int = 2
    gongfeng_memory: int = 8
    gongfeng_debug: bool = False

    # Optionally refresh the baked SWE-Cosmos checkout before running, by overwrite-extracting a pre-built source tarball (host path to a ``.tar.gz`` with files at the archive root, e.g. CI's ``swe-cosmos.latest.tar.gz``). Empty = keep the checkout baked into the image.
    git_archive_path: str | None = None

    # Arbitrary extra env passed through to the pipeline (highest precedence).
    extra_env: dict[str, str] = {}

    # Host-side ceiling on the whole run (seconds). Bounds run_command in the sandbox and the per-invoke request_timeout the client waits on.
    pipeline_timeout: float = 86400.0


class BugmakerPipelineOutput(TaskRunOutput, WithScore):
    """Result of one repo's pipeline run."""

    done: bool
    repo_name: str
    status: Literal['Finished', 'Error', 'TimeLimitExceeded']
    timed_out: bool = False
    return_code: int | None = None
    exhausted: int = 0
    unvalidated: int = 0
    validated: int = 0
    resolved: int = 0
    issued: int = 0
    instances_path: str | None = None  # host path of the downloaded final jsonl (if any)
    log_path: str | None = None  # host path of the downloaded run log (if any)
    git_log: str | None = None
    execution_time: float = 0.0


def _build_pipeline_env(task_input: BugmakerPipelineInput) -> dict[str, str]:
    """Assemble the env vars the pipeline script reads (excluding LLM proxy vars,
    which the sandbox target injects once it knows the proxy port)."""
    env: dict[str, str] = {
        'ENVIRONMENT_TYPE': task_input.environment_type,
        'MAX_BUGS': str(task_input.max_bugs),
        'MAX_WORKERS': str(task_input.max_workers),
        'SEED': str(task_input.seed),
        'NUM_RETRIES': str(task_input.num_retries),
        'GONGFENG_DEBUG': 'true' if task_input.gongfeng_debug else 'false',
        # function_compose knobs (ignored by the script for function_rewrite).
        'NUM_PATCHES': str(task_input.num_patches),
        'COMPOSE_CRITERIA': task_input.compose_criteria,
        'CONFLICT_STRATEGY': task_input.conflict_strategy,
        'COMPOSE_MAX_REUSE': str(task_input.compose_max_reuse),
    }

    if task_input.environment_type == 'gongfeng':
        env['GONGFENG_API_BASE_URL'] = task_input.gongfeng_api_base_url
        if task_input.gongfeng_api_key:
            env['GONGFENG_API_KEY'] = task_input.gongfeng_api_key
        if task_input.gongfeng_runtime_image:
            env['GONGFENG_RUNTIME_IMAGE'] = task_input.gongfeng_runtime_image
        if task_input.gongfeng_cpu > 0:
            env['GONGFENG_CPU'] = str(task_input.gongfeng_cpu)
        if task_input.gongfeng_memory > 0:
            env['GONGFENG_MEMORY'] = str(task_input.gongfeng_memory)

    # Caller overrides win.
    env.update(task_input.extra_env)
    return env


async def _upload_optional_input(runtime: Any, host_path: str, sandbox_path: str) -> str:
    """Upload an optional host input file into the sandbox, returning its sandbox path.

    Used for function_rewrite's entity file and function_compose's bug-data file, which are passed to ``swecosmos bugmaker`` as ``--input-path``.
    """
    src = Path(host_path)
    if not src.is_file():
        raise FileNotFoundError(f'input file not found on host: {host_path}')
    await runtime.upload_file([
        FileItem(path=sandbox_path, content=src.read_text(encoding='utf-8'), encoding='utf8'),
    ])
    return sandbox_path


async def _upload_binary_input(runtime: Any, host_path: str, sandbox_path: str) -> str:
    """Upload a binary host file (e.g. a ``.tar.gz``) into the sandbox as base64, returning its sandbox path."""
    src = Path(host_path)
    if not src.is_file():
        raise FileNotFoundError(f'input file not found on host: {host_path}')
    content = base64.b64encode(src.read_bytes()).decode('ascii')
    await runtime.upload_file([
        FileItem(path=sandbox_path, content=content, encoding='base64'),
    ])
    return sandbox_path


async def _download_text_artifact(runtime: Any, sandbox_path: str, dest: Path) -> str | None:
    """Download a single utf8 text file from the sandbox, tolerating absence."""
    try:
        resp = await runtime.download_file([FileItem(path=sandbox_path, encoding='utf8')])
    except Exception as exc:  # noqa: BLE001 - missing artifacts are expected on early-exit
        logger.warning('Failed to download %s: %s', sandbox_path, exc)
        return None
    if not resp.files:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.files[0].content or '', encoding='utf-8')
    return str(dest)


def _write_result_json(
    host_out_dir: Path, status: str, counts: dict[str, Any], return_code: int | None, score: float
) -> str:
    """Write the per-repo ``result.json`` summary with a fixed 12-key schema.

    ``status`` is chosen from ``Finished``, ``Error``, ``TimeLimitExceeded``. ``return_code`` is the pipeline script's exit code (``None`` when the run never produced one). ``score`` mirrors the task output's score (1.0 only when the run issued at least one instance and produced the final jsonl). All counts (``max_bugs``/ ``exhausted``/ ``unvalidated``/``validated``/``resolved``/``issued``) are read from ``counts``, defaulting to 0 when absent (e.g. an early-exit run).

    The three rates are ratios of successive funnel stages, each 0.0 when its denominator is not positive: ``validated_rate = validated / unvalidated``, ``resolved_rate = resolved / validated``, ``issued_rate = issued / unvalidated``.
    """
    max_bugs = int(counts.get('max_bugs', 0))
    exhausted = int(counts.get('exhausted', 0))
    unvalidated = int(counts.get('unvalidated', 0))
    validated = int(counts.get('validated', 0))
    resolved = int(counts.get('resolved', 0))
    issued = int(counts.get('issued', 0))
    result_data = {
        'status': status,
        'return_code': return_code,
        'max_bugs': max_bugs,
        'exhausted': exhausted,
        'unvalidated': unvalidated,
        'validated': validated,
        'resolved': resolved,
        'issued': issued,
        'validated_rate': (validated / unvalidated) if unvalidated > 0 else 0.0,
        'resolved_rate': (resolved / validated) if validated > 0 else 0.0,
        'issued_rate': (issued / unvalidated) if unvalidated > 0 else 0.0,
        'score': score,
    }
    dest = host_out_dir / 'result.json'
    dest.write_text(json.dumps(result_data, ensure_ascii=False, indent=4), encoding='utf-8')
    return str(dest)


@task_registry.register('swecosmos_bugmaker_pipeline')
async def run_bugmaker_pipeline(task_input: BugmakerPipelineInput) -> BugmakerPipelineOutput:
    # Create the per-repo output dir up front so a result.json is always written, even when the run aborts before or during the in-sandbox pipeline.
    host_out_dir = Path(task_input.output_dir) / task_input.repo_name
    host_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ctx = require_context()
        if ctx.runtime_provider is None:
            raise RuntimeError('swecosmos_bugmaker_pipeline requires a runtime provider.')
        if BUGMAKER_CLIENT not in ctx.openai_clients or ISSUEGEN_CLIENT not in ctx.openai_clients:
            raise RuntimeError(
                f'swecosmos_bugmaker_pipeline requires both {BUGMAKER_CLIENT} and {ISSUEGEN_CLIENT} OpenAI clients for LLM forwarding.'
            )

        # Routing name = `<bare model id>.<stage>`, derived from each stage client's configured
        # model. The sandbox tags its calls with openai/<routing-name>, which both selects the host
        # client and keeps the real backend model visible in the pipeline's logs and metadata. Stage
        # clients are still looked up by their stage key; only the tag on the request changes.
        bugmaker_model_name = model_routing_name(
            ctx.openai_clients[BUGMAKER_CLIENT].config.request_args.get('model'), stage=BUGMAKER_CLIENT
        )
        issuegen_model_name = model_routing_name(
            ctx.openai_clients[ISSUEGEN_CLIENT].config.request_args.get('model'), stage=ISSUEGEN_CLIENT
        )
        if not bugmaker_model_name or not issuegen_model_name:
            raise RuntimeError(
                'swecosmos_bugmaker_pipeline could not derive a model routing name from the OpenAI client config '
                f'(bugmaker={bugmaker_model_name!r}, issuegen={issuegen_model_name!r}).'
            )
        # Map each routing name back to its stage client. When both stages share a model they collapse
        # to one entry (identical config), which is fine.
        routing_clients = {
            bugmaker_model_name: ctx.openai_clients[BUGMAKER_CLIENT],
            issuegen_model_name: ctx.openai_clients[ISSUEGEN_CLIENT],
        }

        if task_input.bugmaker_strategy == 'function_compose' and not task_input.bug_data_path:
            raise ValueError('bug_data_path is required when bugmaker_strategy="function_compose".')

        repo_dict_file = Path(task_input.repo_dict_path)
        if not repo_dict_file.is_file():
            raise FileNotFoundError(f'repo dict not found on host: {task_input.repo_dict_path}')
        repo_dict_text = repo_dict_file.read_text(encoding='utf-8')

        sandbox_inputs_dir = f'{task_input.sandbox_output_dir.rstrip("/")}/inputs'
        sandbox_repo_dict_path = f'{sandbox_inputs_dir}/{task_input.repo_name}.json'

        # Stagger initial image pull to avoid registry QPS spikes
        startup_jitter = random.uniform(0, 120)
        await asyncio.sleep(startup_jitter)

        # The 'docker' child backend runs each repo test env as an inner container inside this root sandbox, which requires Docker-in-Docker; the 'gongfeng' backend spawns separate sandboxes and needs none. Derive enable_dind purely from the environment type so the single-run and batch entrypoints stay in sync without a separate flag. (enable_dind=False is a no-op for every provider; enable_dind=True is honored only by the Gongfeng/gVisor runtime path.)
        runtime = await ctx.get_runtime(
            task_input.runtime_name,
            enable_dind=(task_input.environment_type == 'docker'),
        )
        await runtime.ws_connect()

        await runtime.upload_file([
            FileItem(path=sandbox_repo_dict_path, content=repo_dict_text, encoding='utf8'),
        ])

        # Upload optional bugmaker input files and inject their sandbox paths into the pipeline env so both strategies are supported.
        pipeline_env = _build_pipeline_env(task_input)
        if task_input.entity_path:
            entity_path = resolve_question_mark_path(task_input.entity_path)
            if entity_path is None:
                raise FileNotFoundError(f'entity file not found for path template: {task_input.entity_path}')
            pipeline_env['ENTITY_PATH'] = await _upload_optional_input(
                runtime, entity_path, f'{sandbox_inputs_dir}/{Path(entity_path).name}'
            )
        if task_input.bug_data_path:
            bug_data_path = resolve_question_mark_path(task_input.bug_data_path)
            if bug_data_path is None:
                raise FileNotFoundError(f'bug data file not found for path template: {task_input.bug_data_path}')
            pipeline_env['BUG_DATA_PATH'] = await _upload_optional_input(
                runtime, bug_data_path, f'{sandbox_inputs_dir}/{Path(bug_data_path).name}'
            )

        # Upload the prebuilt tarball under swe_cosmos_dir; the sandbox target extracts it over the baked checkout.
        sandbox_archive_path: str | None = None
        if task_input.git_archive_path:
            if not Path(task_input.git_archive_path).is_file():
                raise FileNotFoundError(f'git archive file not found on host: {task_input.git_archive_path}')
            sandbox_archive_path = await _upload_binary_input(
                runtime,
                task_input.git_archive_path,
                f'{task_input.swe_cosmos_dir.rstrip("/")}/swe-cosmos.latest.tar.gz',
            )

        # Route each stage's LLM calls (tagged openai/<model-id>) to its host client.
        handler = RoutingLLMRequestHandler(runtime=runtime, openai_clients=routing_clients)
        handler.register()

        try:
            request = InvokeRequest.build(
                _TARGET,
                request_timeout=task_input.pipeline_timeout,
                venv_id=runtime.default_venv_id,
                repo_name=task_input.repo_name,
                repo_dict_path=sandbox_repo_dict_path,
                pipeline_env=pipeline_env,
                swe_cosmos_dir=task_input.swe_cosmos_dir,
                output_dir=task_input.sandbox_output_dir,
                bugmaker_strategy=task_input.bugmaker_strategy,
                timeout=task_input.pipeline_timeout,
                git_archive_path=sandbox_archive_path,
                bugmaker_model_name=bugmaker_model_name,
                issuegen_model_name=issuegen_model_name,
            )
            result: dict[str, Any] = await runtime.invoke(request)
        finally:
            handler.unregister()

        counts = result.get('counts') or {}
        # score is 1.0 if the pipeline issued at least one instance and produced a final jsonl, 0.0 otherwise
        score = float(counts.get('issued', 0) > 0 and result.get('sandbox_instances_path') is not None)
        result_status = result.get('status', 'Error')

        # Download the final instances jsonl (<repo>.<strategy>_instances.jsonl). Absent when the pipeline exited before producing it.
        instances_host_path: str | None = None
        if result.get('sandbox_instances_path'):
            instances_host_path = await _download_text_artifact(
                runtime,
                result['sandbox_instances_path'],
                host_out_dir / f'{task_input.repo_name}.{task_input.bugmaker_strategy}_instances.jsonl',
            )

        # Download the tee'd pipeline log as run_bugmaker.log.
        log_host_path: str | None = None
        if result.get('sandbox_log_path'):
            log_host_path = await _download_text_artifact(
                runtime,
                result['sandbox_log_path'],
                host_out_dir / 'run_bugmaker.log',
            )

        _write_result_json(host_out_dir, result_status, counts, return_code=result.get('return_code'), score=score)

        return BugmakerPipelineOutput(
            score=score,
            done=bool(result.get('done')),
            repo_name=task_input.repo_name,
            status=result.get('status', 'Error'),
            timed_out=bool(result.get('timed_out')),
            return_code=result.get('return_code'),
            exhausted=int(counts.get('exhausted', 0)),
            unvalidated=int(counts.get('unvalidated', 0)),
            validated=int(counts.get('validated', 0)),
            resolved=int(counts.get('resolved', 0)),
            issued=int(counts.get('issued', 0)),
            instances_path=instances_host_path,
            log_path=log_host_path,
            git_log=result.get('git_log'),
            execution_time=float(result.get('execution_time', 0.0)),
        )
    except Exception:
        # The repo errored out or never ran: still emit a failed result.json (the instances jsonl / log may legitimately be absent), then re-raise so the batch layer can retry.
        _write_result_json(host_out_dir, 'Error', {}, return_code=None, score=0.0)
        logger.exception('swecosmos_bugmaker_pipeline failed for %s', task_input.repo_name)
        raise
