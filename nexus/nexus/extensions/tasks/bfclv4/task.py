"""BFCL v4 task implementation for Nexus framework.

Registers a 'bfcl' task that runs BFCL inference and evaluation on a single
test entry using the Nexus OpenAI client. No agent is needed — this task
directly calls the LLM via BFCL's handler pipeline.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import time

from copy import deepcopy
from pathlib import Path
from types import MethodType
from typing import Any

from openai import AsyncOpenAI, OpenAI


# Dedicated thread pool for BFCL inference calls.
# Must NOT use asyncio's default executor (asyncio.to_thread does exactly that)
# to avoid deadlock when concurrent tasks exhaust the default pool, starving
# httpx/anyio internal operations that also need the default pool.
_bfcl_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=256,
    thread_name_prefix='bfcl',
)

from nexus.core import require_context, task_registry  # noqa: E402
from nexus.extensions.tasks.bfclv4.evaluator import (  # noqa: E402
    evaluate_agentic,
    evaluate_multi_turn,
    evaluate_relevance,
    evaluate_single_turn,
)
from nexus.extensions.tasks.bfclv4.handler import NexusOpenAIHandler  # noqa: E402
from nexus.extensions.tasks.bfclv4.types import BFCLTaskInput, BFCLTaskOutput, get_category_type  # noqa: E402
from nexus.trace import observe as _observe  # noqa: E402


logger = logging.getLogger(__name__)

# Module-level cached OpenAI sync client to avoid creating a new client per task.
# SyncHttpxClientWrapper.__del__ triggers RecursionError in asyncio + threading
# environments, causing massive C-stack memory leaks. Reusing one client avoids
# repeated __del__ invocations during GC.
_shared_sync_client: Any | None = None
_shared_sync_client_lock = threading.Lock()
# Extra kwargs (e.g. extra_body for Taiji) injected into every chat.completions.create() call.
_shared_extra_create_kwargs: dict[str, Any] = {}

# Per-scenario locks for memory categories. Memory entries within the same
# (category, scenario) share state via snapshot files and must execute in strict
# order. Different scenarios are independent and run in parallel.
# Key: "{test_category}:{scenario}" → asyncio.Lock
_memory_scenario_locks: dict[str, asyncio.Lock] = {}
_memory_scenario_locks_mutex = threading.Lock()


def _get_memory_lock(test_category: str, scenario: str) -> asyncio.Lock:
    """Get or create a per-scenario lock for memory categories."""
    key = f'{test_category}:{scenario}'
    if key not in _memory_scenario_locks:
        with _memory_scenario_locks_mutex:
            if key not in _memory_scenario_locks:
                _memory_scenario_locks[key] = asyncio.Lock()
    return _memory_scenario_locks[key]


def _get_rss_mb() -> float:
    """Get current process RSS in MB from /proc/self/status."""
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def _obj_size_summary(obj: Any, label: str = '') -> str:
    """Return a short string describing the size of an object (JSON-serializable)."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
        return f'{label}={len(s) / 1024:.1f}KB'
    except Exception:
        return f'{label}=?'


def _sanitize_output(obj: Any) -> Any:
    """Force-convert an object to pure JSON-safe Python primitives.

    The Nexus observe decorator calls serialize_object(result) on task outputs,
    which recursively traverses __dict__ of any non-primitive objects. BFCL's
    multi_turn inference leaves references to internal objects (GorillaFileSystem,
    Directory, etc.) inside model_responses and metadata. These objects have deeply
    nested __dict__ trees that cause 16M+ recursive serialize_object calls and
    multi-GB anonymous mmap memory growth.

    This function round-trips through JSON to strip all non-primitive references,
    ensuring serialize_object only sees plain dicts/lists/strings/numbers.
    """
    if obj is None:
        return None
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except (TypeError, ValueError, OverflowError):
        # Fallback: convert to string representation
        return str(obj)


def _restore_path_in_initial_config(test_entry: dict[str, Any]) -> None:
    """Convert model_result_dir strings back to Path objects in initial_config.

    The generator stores Path as str for JSON serialization. BFCL's
    _prepare_snapshot expects pathlib.Path for path operations.
    """
    ic = test_entry.get('initial_config', {})
    for cls_config in ic.values():
        if isinstance(cls_config, dict) and 'model_result_dir' in cls_config:
            cls_config['model_result_dir'] = Path(cls_config['model_result_dir'])


def _patch_taiji_openapi_port(nexus_client: Any) -> int | None:
    """Patch a task-local TaijiClient instance to use :1081/openapi."""
    try:
        from nexus.extensions.openai_clients.taiji import TaijiClient
    except ImportError:
        return None

    if not isinstance(nexus_client, TaijiClient) or nexus_client.url_pattern != 'openapi':
        return None
    if getattr(nexus_client, '_bfcl_openapi_port_patched', False):
        return int(getattr(nexus_client, '_bfcl_openapi_port', 1081))

    openapi_port = int(os.getenv('BFCL_TAIJI_OPENAPI_PORT', '1081'))

    def _create_openai_client_with_port(self: Any, host: str, port: int) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=f'http://{host}:{openapi_port}/openapi',
            api_key='dummy',
            timeout=self.timeout,
            default_headers={
                'Authorization': self.authorization,
                'wsid': self.wsid,
            },
        )

    nexus_client._create_openai_client = MethodType(_create_openai_client_with_port, nexus_client)
    nexus_client._bfcl_openapi_port_patched = True
    nexus_client._bfcl_openapi_port = openapi_port
    return openapi_port


class _SyncChatCompletionsAdapter:
    """Sync adapter for BFCL handler over a Nexus async client."""

    def __init__(self, nexus_client: Any, loop: asyncio.AbstractEventLoop):
        self._nexus_client = nexus_client
        self._loop = loop

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.pop('messages')
        model = kwargs.pop('model', None)
        future = asyncio.run_coroutine_threadsafe(
            self._nexus_client._create_completion(messages=messages, model=model, **kwargs),
            self._loop,
        )
        return future.result()


class _SyncChatAdapter:
    def __init__(self, nexus_client: Any, loop: asyncio.AbstractEventLoop):
        self.completions = _SyncChatCompletionsAdapter(nexus_client, loop)


class _NexusSyncClientAdapter:
    """Expose `.chat.completions.create()` for BFCL's sync handler."""

    def __init__(self, nexus_client: Any, base_url: str):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name='bfcl-taiji-loop', daemon=True)
        self._thread.start()
        self.chat = _SyncChatAdapter(nexus_client, self._loop)
        self.base_url = base_url

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()


def _build_sync_client(ctx) -> Any:
    """Get or create a shared sync OpenAI client.

    Reuses a single global OpenAI client instance across all concurrent tasks
    to avoid per-task httpx connection pool / SSL context memory accumulation.
    The OpenAI sync client is thread-safe (httpx.Client uses connection pooling).

    For TaijiClient: uses Polaris service discovery to resolve the endpoint,
    then creates a standard OpenAI sync client with the discovered address and
    appropriate headers. Also populates _shared_extra_create_kwargs with
    taiji-specific extra_body params.
    """
    global _shared_sync_client, _shared_extra_create_kwargs

    if _shared_sync_client is not None:
        return _shared_sync_client

    with _shared_sync_client_lock:
        # Double-check after acquiring lock
        if _shared_sync_client is not None:
            return _shared_sync_client

        # Build from context or task_config
        if ctx.openai_clients:
            nexus_client = ctx.get_openai_client('main')
            _shared_sync_client, _shared_extra_create_kwargs = _create_sync_client_from_nexus(nexus_client)
        else:
            client_config = ctx.task_config.get('_openai_client_config')
            if client_config:
                _shared_sync_client = OpenAI(**client_config)
            else:
                raise RuntimeError(
                    'No OpenAI client available. Ensure BFCL_OPENAI_BASE_URL and BFCL_OPENAI_API_KEY are set.'
                )

        logger.debug(f'Created shared sync OpenAI client: base_url={_shared_sync_client.base_url}')
        return _shared_sync_client


def _create_sync_client_from_nexus(nexus_client) -> tuple[Any, dict[str, Any]]:
    """Create a sync OpenAI client from a nexus OpenAICompatibleClient.

    Returns (sync_client, extra_create_kwargs). extra_create_kwargs is non-empty
    only for TaijiClient (contains extra_body for Polaris-served models).
    """
    try:
        from nexus.extensions.openai_clients.taiji import TaijiClient, _discover_service

        if isinstance(nexus_client, TaijiClient):
            host, port = _discover_service(nexus_client.namespace, nexus_client.service)
            openapi_port = _patch_taiji_openapi_port(nexus_client)
            base_url = (
                f'http://{host}:{openapi_port}/openapi' if openapi_port is not None else f'http://{host}:{port}/v1'
            )
            client = _NexusSyncClientAdapter(nexus_client=nexus_client, base_url=base_url)
            logger.debug(
                f'Created BFCL sync adapter from TaijiClient: base_url={base_url}, '
                f'enable_thinking={nexus_client.enable_thinking}, protocol=nexus-client'
            )
            return client, {}
    except ImportError:
        pass

    config = nexus_client.config
    client_args = {k: v for k, v in config.client_args.items() if k != 'max_retries'}
    return OpenAI(**client_args), {}


# ---------------------------------------------------------------------------
# Task registration with observe(ignore_inputs=True)
#
# The default validate_task processor wraps every task with
#   observe(name=..., ignore_inputs=False)
# which causes serialize_object to recursively traverse the entire
# BFCLTaskInput.test_entry (including BFCL internal objects like
# GorillaFileSystem/Directory that have deeply nested __dict__).
# For multi_turn entries this produces 16 M+ recursive serialize calls
# and multi-GB anonymous mmap memory growth.
#
# Fix: register normally (so the validator is happy), then replace the
# stored function object with one that uses ignore_inputs=True.
# This is BFCL-local — no framework code is changed.
# ---------------------------------------------------------------------------


@task_registry.register('bfcl')
async def bfcl_task(task_input: BFCLTaskInput) -> BFCLTaskOutput:
    """Run a single BFCL test case: inference + evaluation.

    Flow:
    1. Get OpenAI client from ExecutionContext
    2. Create NexusOpenAIHandler with sync client
    3. Run handler.inference() in a thread (sync → async bridge)
    4. Route to appropriate evaluator based on category type
    5. Return BFCLTaskOutput with score
    """
    # Memory categories must run serially within each scenario (snapshot file
    # dependency chain), but different scenarios can run in parallel.
    is_memory = 'memory' in task_input.test_category
    if is_memory:
        scenario = task_input.test_entry.get('scenario', 'unknown')
        lock = _get_memory_lock(task_input.test_category, scenario)
        async with lock:
            return await _bfcl_task_inner(task_input)
    return await _bfcl_task_inner(task_input)


async def _bfcl_task_inner(task_input: BFCLTaskInput) -> BFCLTaskOutput:
    """Core BFCL task logic."""
    ctx = require_context()

    # Deep copy test entry because BFCL mutates it during inference
    test_entry = deepcopy(task_input.test_entry)
    test_id = test_entry['id']
    test_category = task_input.test_category
    category_type = get_category_type(test_category)

    # Restore Path objects for memory categories. The generator converts Path to str
    # for JSON serialization, but BFCL's _prepare_snapshot expects pathlib.Path.
    _restore_path_in_initial_config(test_entry)

    rss_start = _get_rss_mb()
    logger.debug(
        f'BFCL task: {test_id} (category={test_category}, type={category_type}), '
        f'RSS={rss_start:.0f}MB, {_obj_size_summary(test_entry, "entry")}'
    )

    # Build handler
    try:
        sync_client = _build_sync_client(ctx)
        # Use a simple registry name that won't conflict with MODEL_CONFIG_MAPPING lookups.
        registry_name = task_input.model_name.replace('/', '_').replace('-', '_').replace('.', '_')
        handler = NexusOpenAIHandler(
            client=sync_client,
            model_name=task_input.model_name,
            temperature=task_input.temperature,
            registry_name=registry_name,
            is_fc_model=task_input.is_fc_model,
            extra_create_kwargs=_shared_extra_create_kwargs,
        )
    except Exception as e:
        logger.error(f'Failed to create handler for {test_id}: {e}')
        return BFCLTaskOutput(
            score=0.0,
            test_id=test_id,
            test_category=test_category,
            error=f'Handler creation failed: {e}',
        )

    # Run inference (sync) in dedicated thread pool to avoid blocking event loop.
    # Uses _bfcl_executor instead of asyncio.to_thread (which uses the default
    # executor and can deadlock at high concurrency).
    try:
        start_time = time.time()
        loop = asyncio.get_event_loop()
        model_responses, metadata = await loop.run_in_executor(
            _bfcl_executor,
            lambda: handler.inference(
                test_entry,
                include_input_log=False,
                exclude_state_log=True,
            ),
        )
        latency = time.time() - start_time
    except Exception as e:
        rss_err = _get_rss_mb()
        logger.error(f'Inference failed for {test_id}: {e}, RSS={rss_err:.0f}MB (delta={rss_err - rss_start:+.0f}MB)')
        return BFCLTaskOutput(
            score=0.0,
            test_id=test_id,
            test_category=test_category,
            error=f'Inference failed: {e}',
        )

    # Extract token usage
    token_usage = _extract_token_usage(metadata)

    # Measure response size
    resp_size = _obj_size_summary(model_responses, 'resp')
    meta_size = _obj_size_summary(metadata, 'meta')
    rss_after_inference = _get_rss_mb()

    logger.info(
        f'Inference complete for {test_id}: latency={latency:.2f}s, '
        f'tokens={token_usage.get("total_input", "N/A")}/{token_usage.get("total_output", "N/A")}, '
        f'{resp_size}, {meta_size}, RSS={rss_after_inference:.0f}MB (delta={rss_after_inference - rss_start:+.0f}MB)'
    )

    # Warn if response or metadata is unusually large
    try:
        resp_bytes = len(json.dumps(model_responses, ensure_ascii=False, default=str))
        meta_bytes = len(json.dumps(metadata, ensure_ascii=False, default=str))
        if resp_bytes > 1024 * 1024:  # > 1MB
            logger.warning(
                f'LARGE RESPONSE for {test_id}: resp={resp_bytes / 1024 / 1024:.1f}MB. '
                f'First 500 chars: {json.dumps(model_responses, ensure_ascii=False, default=str)[:500]}'
            )
        if meta_bytes > 1024 * 1024:  # > 1MB
            logger.warning(f'LARGE METADATA for {test_id}: meta={meta_bytes / 1024 / 1024:.1f}MB')
    except Exception:
        pass

    # Run evaluation
    try:
        valid, eval_detail = _evaluate(
            handler=handler,
            model_responses=model_responses,
            ground_truth=task_input.ground_truth,
            test_entry=test_entry,
            test_category=test_category,
            category_type=category_type,
            model_name=registry_name,
            is_fc_model=task_input.is_fc_model,
        )
    except Exception as e:
        rss_eval_err = _get_rss_mb()
        logger.error(
            f'Evaluation failed for {test_id}: {e}, RSS={rss_eval_err:.0f}MB (delta={rss_eval_err - rss_start:+.0f}MB)'
        )
        return BFCLTaskOutput(
            score=0.0,
            test_id=test_id,
            test_category=test_category,
            model_responses=_sanitize_output(model_responses),
            token_usage=token_usage,
            latency=latency,
            error=f'Evaluation failed: {e}',
        )

    score = 1.0 if valid else 0.0
    rss_final = _get_rss_mb()
    logger.info(f'BFCL result: {test_id} → score={score}, RSS={rss_final:.0f}MB (delta={rss_final - rss_start:+.0f}MB)')

    # Warn on large RSS growth
    rss_delta = rss_final - rss_start
    if rss_delta > 500:
        logger.warning(
            f'HIGH MEMORY GROWTH for {test_id}: delta={rss_delta:.0f}MB. This entry may be causing memory issues.'
        )

    # Sanitize all output fields to pure JSON-safe primitives.
    # This prevents serialize_object() in the observe decorator from
    # recursively traversing BFCL internal objects (GorillaFileSystem,
    # Directory, etc.) which causes multi-GB memory explosions.
    return BFCLTaskOutput(
        score=score,
        test_id=test_id,
        test_category=test_category,
        model_responses=_sanitize_output(model_responses),
        eval_result=_sanitize_output(eval_detail),
        token_usage=token_usage,
        latency=latency,
    )


# Post-registration: replace the observe-wrapped function with one that uses
# ignore_inputs=True. This prevents serialize_object from recursively traversing
# BFCLTaskInput.test_entry (which contains BFCL objects like GorillaFileSystem).
_bfcl_task_unwrapped = bfcl_task.__wrapped__ if hasattr(bfcl_task, '__wrapped__') else bfcl_task
_bfcl_task_patched = _observe(
    name='task.bfcl',
    attributes={'type': 'task', 'name': 'bfcl'},
    ignore_inputs=True,
)(_bfcl_task_unwrapped)
task_registry._registry['bfcl'].obj = _bfcl_task_patched


def _evaluate(
    handler: NexusOpenAIHandler,
    model_responses: Any,
    ground_truth: dict[str, Any] | None,
    test_entry: dict[str, Any],
    test_category: str,
    category_type: str,
    model_name: str,
    is_fc_model: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Route evaluation to the appropriate checker based on category type."""
    if category_type == 'relevance':
        return evaluate_relevance(handler, model_responses, test_entry, test_category)

    if ground_truth is None:
        return False, {'error': 'Ground truth is required for this category type', 'error_type': 'missing_ground_truth'}

    if category_type == 'single_turn':
        return evaluate_single_turn(
            handler, model_responses, ground_truth, test_entry, test_category, model_name, is_fc_model
        )

    if category_type == 'multi_turn':
        return evaluate_multi_turn(
            handler, model_responses, ground_truth, test_entry, test_category, model_name, is_fc_model
        )

    if category_type == 'agentic':
        return evaluate_agentic(handler, model_responses, ground_truth, test_entry, test_category)

    if category_type == 'format_sensitivity':
        # Format sensitivity uses the same AST checker as single-turn
        return evaluate_single_turn(
            handler, model_responses, ground_truth, test_entry, test_category, model_name, is_fc_model
        )

    return False, {'error': f'Unknown category type: {category_type}', 'error_type': 'unknown_category'}


def _extract_token_usage(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract token usage from BFCL metadata into a flat dict."""
    usage: dict[str, Any] = {}

    input_tokens = metadata.get('input_token_count')
    output_tokens = metadata.get('output_token_count')
    latency = metadata.get('latency')

    if input_tokens is not None:
        if isinstance(input_tokens, list):
            # Multi-turn: list of lists
            flat_input = _flatten_nested(input_tokens)
            flat_output = _flatten_nested(output_tokens) if output_tokens else []
            usage['total_input'] = sum(x for x in flat_input if isinstance(x, (int, float)))
            usage['total_output'] = sum(x for x in flat_output if isinstance(x, (int, float)))
            usage['per_turn_input'] = input_tokens
            usage['per_turn_output'] = output_tokens
        else:
            usage['total_input'] = input_tokens
            usage['total_output'] = output_tokens

    if latency is not None:
        if isinstance(latency, list):
            flat_latency = _flatten_nested(latency)
            usage['total_latency'] = sum(x for x in flat_latency if isinstance(x, (int, float)))
        else:
            usage['total_latency'] = latency

    return usage


def _flatten_nested(lst: Any) -> list:
    """Flatten a nested list structure into a flat list."""
    if not isinstance(lst, list):
        return [lst]
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(_flatten_nested(item))
        else:
            result.append(item)
    return result
