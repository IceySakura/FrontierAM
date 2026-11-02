"""WAA evaluation orchestrator.

Combines getters and metrics to evaluate task completion.
Mirrors the logic of DesktopEnv.evaluate() from the original WAA codebase.
"""

import asyncio
import logging
import ntpath

from typing import Any

from nexus.extensions.tasks.waa.getters import GETTER_REGISTRY
from nexus.extensions.tasks.waa.metrics import METRIC_REGISTRY
from nexus.extensions.tasks.waa.registry import setup_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger(__name__)


async def evaluate_task(
    runtime: AGSRawWindowsAgentArenaRuntime,
    evaluator_config: dict,
    agent_conclusion: str,
) -> float:
    """Evaluate whether a WAA task was successfully completed.

    This mirrors DesktopEnv.evaluate() logic:
    1. Execute postconfig (if any)
    2. Handle infeasible tasks
    3. Get result state via getter
    4. Get expected state via getter
    5. Compute metric score

    Args:
        runtime: AGSRawWindowsAgentArenaRuntime instance.
        evaluator_config: The "evaluator" dict from task config JSON.
        agent_conclusion: Agent's final conclusion — 'DONE' or 'FAIL'.

    Returns:
        float: Evaluation score (typically 0.0 or 1.0).
    """
    # 1. Execute postconfig (evaluation-time setup actions)
    postconfig = evaluator_config.get('postconfig', [])
    if postconfig:
        logger.debug(f'Executing {len(postconfig)} postconfig actions')
        if not await _execute_postconfig(runtime, evaluator_config, postconfig):
            return 0.0

    # 2. Handle infeasible tasks
    func_name = evaluator_config.get('func', '')
    if func_name == 'infeasible':
        if agent_conclusion == 'FAIL':
            logger.info('Infeasible task: agent correctly returned FAIL')
            return 1.0
        logger.info('Infeasible task: agent did not return FAIL')
        return 0.0

    # For feasible tasks, if agent returned FAIL → score 0
    if agent_conclusion == 'FAIL':
        logger.info('Feasible task but agent returned FAIL')
        return 0.0

    # 3. Evaluate — single metric or multiple metrics
    if isinstance(func_name, list):
        return await _evaluate_multi_metric(runtime, evaluator_config, func_name)
    return await _evaluate_single_metric(runtime, evaluator_config, func_name)


async def _execute_postconfig(
    runtime: AGSRawWindowsAgentArenaRuntime,
    evaluator_config: dict,
    postconfig: list[dict],
) -> bool:
    """Run evaluator postconfig.

    Missing result files should score 0.0 instead of marking the whole runner
    failed. Other postconfig failures are still infrastructure/setup errors.
    """
    # 有些任务要求 agent 生成文件后 postconfig 再打开/转换。文件不存在说明 agent 没完成任务，
    # 应该给 0 分；如果让 FileNotFoundError 冒泡，会把 agent 质量问题误报成 runner failed。
    result_paths = {_normalize_vm_path(path) for path in _iter_vm_result_paths(evaluator_config.get('result'))}

    for cfg in postconfig:
        config_type = cfg.get('type', '')
        parameters = cfg.get('parameters', {})

        handler = setup_registry.get(config_type)
        if handler is None:
            logger.warning(f'Unknown setup type: {config_type}, skipping')
            continue

        logger.debug(f'Setup: {config_type}({parameters})')
        try:
            await handler(runtime, **parameters)
        except Exception as exc:
            if _is_missing_result_open(config_type, parameters, exc, result_paths):
                logger.info(
                    'Evaluator result file is missing during postconfig open; returning score=0.0: %s',
                    parameters.get('path', ''),
                )
                return False
            raise

    return True


def _iter_vm_result_paths(result_config: Any):
    if isinstance(result_config, list):
        for item in result_config:
            yield from _iter_vm_result_paths(item)
        return

    if not isinstance(result_config, dict) or result_config.get('type') != 'vm_file':
        return

    path = result_config.get('path')
    if isinstance(path, str):
        yield path
    elif isinstance(path, list):
        for item in path:
            if isinstance(item, str):
                yield item


def _normalize_vm_path(path: str) -> str:
    return ntpath.normcase(ntpath.normpath(path.replace('/', '\\')))


def _is_missing_result_open(
    config_type: str,
    parameters: dict,
    exc: Exception,
    result_paths: set[str],
) -> bool:
    if config_type != 'open':
        return False

    path = parameters.get('path')
    if not isinstance(path, str) or _normalize_vm_path(path) not in result_paths:
        return False

    if isinstance(exc, FileNotFoundError):
        return True

    text = str(exc)
    return 'HTTP 404' in text and 'File not found' in text


async def _evaluate_single_metric(
    runtime: AGSRawWindowsAgentArenaRuntime,
    evaluator_config: dict,
    func_name: str,
) -> float:
    """Evaluate with a single metric function."""
    # Get metric function
    metric_fn = METRIC_REGISTRY.get(func_name)
    if metric_fn is None:
        raise ValueError(f'Unknown metric function: {func_name}')

    # Get result state
    try:
        result_state = await _get_result_state(runtime, evaluator_config.get('result'))
    except FileNotFoundError:
        # result getter 找不到 agent 产物时按任务失败计分，而不是让整条 batch 标记为基础设施失败。
        logger.error('File not found in result getter')
        return 0.0
    except Exception:
        logger.exception('Unexpected error in result getter')
        return 0.0

    # Get expected state
    expected_state = await _get_expected_state(runtime, evaluator_config.get('expected'))

    # Get metric options
    options = evaluator_config.get('options', {}) or {}

    # Compute score — metrics are sync, run in thread to avoid blocking
    # Note: when expected_state is None, only pass result_state (some metrics are single-arg)
    try:
        if expected_state is not None:
            score = await asyncio.to_thread(metric_fn, result_state, expected_state, **options)
        else:
            score = await asyncio.to_thread(metric_fn, result_state, **options)
    except Exception:
        logger.exception(f'Metric {func_name} raised an exception')
        return 0.0

    return float(score)


async def _evaluate_multi_metric(
    runtime: AGSRawWindowsAgentArenaRuntime,
    evaluator_config: dict,
    func_names: list[str],
) -> float:
    """Evaluate with multiple metric functions combined by conjunction.

    When func is a list, result/expected/options must also be lists of the same length.
    """
    conj = evaluator_config.get('conj', 'and')
    result_configs = evaluator_config.get('result', [])
    expected_configs = evaluator_config.get('expected', [])
    options_list = evaluator_config.get('options', [])

    # Ensure all lists are the same length
    n = len(func_names)
    if not isinstance(result_configs, list):
        result_configs = [result_configs] * n
    if not isinstance(expected_configs, list):
        expected_configs = [expected_configs] * n
    if not isinstance(options_list, list):
        options_list = [options_list] * n

    # Pad lists if shorter
    result_configs = result_configs + [None] * (n - len(result_configs))
    expected_configs = expected_configs + [None] * (n - len(expected_configs))
    options_list = options_list + [{}] * (n - len(options_list))

    results = []
    for idx, func_name in enumerate(func_names):
        metric_fn = METRIC_REGISTRY.get(func_name)
        if metric_fn is None:
            raise ValueError(f'Unknown metric function: {func_name}')

        # Get result state
        try:
            result_state = await _get_result_state(runtime, result_configs[idx])
        except Exception:
            logger.exception(f'Error in result getter for metric {func_name}[{idx}]')
            if conj == 'and':
                return 0.0
            results.append(0.0)
            continue

        # Get expected state
        try:
            expected_state = await _get_expected_state(runtime, expected_configs[idx])
        except Exception:
            logger.exception(f'Error in expected getter for metric {func_name}[{idx}]')
            if conj == 'and':
                return 0.0
            results.append(0.0)
            continue

        # Get options
        options = options_list[idx] or {}

        # Compute metric — metrics are sync, run in thread to avoid blocking
        # Note: when expected_state is None, only pass result_state (some metrics are single-arg)
        try:
            if expected_state is not None:
                score = float(await asyncio.to_thread(metric_fn, result_state, expected_state, **options))
            else:
                score = float(await asyncio.to_thread(metric_fn, result_state, **options))
        except Exception:
            logger.exception(f'Metric {func_name}[{idx}] raised an exception')
            score = 0.0

        # Short-circuit evaluation
        if conj == 'and' and score == 0.0:
            return 0.0
        if conj == 'or' and score == 1.0:
            return 1.0

        results.append(score)

    if not results:
        return 0.0

    if conj == 'and':
        return sum(results) / len(results)
    # 'or'
    return max(results)


async def _get_result_state(runtime: AGSRawWindowsAgentArenaRuntime, result_config: dict | None) -> Any:
    """Call the appropriate result getter.

    result_config format: {"type": "storage_sense_run_frequency", ...other params...}
    The getter is looked up by "get_{type}" in the GETTER_REGISTRY.
    """
    if not result_config:
        return None

    getter_type = result_config.get('type', '')
    getter_fn = GETTER_REGISTRY.get(getter_type)

    if getter_fn is None:
        raise ValueError(f'Unknown result getter type: {getter_type}')

    return await getter_fn(runtime, result_config)


async def _get_expected_state(runtime: AGSRawWindowsAgentArenaRuntime, expected_config: dict | None) -> Any:
    """Call the appropriate expected getter.

    expected_config format: {"type": "rule", "rules": {...}}
    The getter is looked up by "get_{type}" in the GETTER_REGISTRY.
    """
    if not expected_config:
        return None

    getter_type = expected_config.get('type', '')
    getter_fn = GETTER_REGISTRY.get(getter_type)

    if getter_fn is None:
        raise ValueError(f'Unknown expected getter type: {getter_type}')

    return await getter_fn(runtime, expected_config)
