"""OSWorld evaluator pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.os_world.evaluation.errors import EvaluatorInfrastructureError
from nexus.extensions.tasks.os_world.execution.cdp import connect_chromium
from nexus.extensions.tasks.os_world.execution.commands import command_to_shell, osworld_user_command
from nexus.extensions.tasks.os_world.execution.setup import (
    add_chrome_proxy_arg,
    osworld_runtime_use_proxy,
    run_setup_action,
)
from nexus.extensions.tasks.os_world.getters import GETTER_REGISTRY
from nexus.extensions.tasks.os_world.getters.runtime import (
    CACHE_DIR_CONFIG_KEY,
    response_stderr,
    response_stdout,
    run_bash,
    write_text,
)
from nexus.extensions.tasks.os_world.metrics import METRIC_REGISTRY, OfficialMetricUnavailableError


logger = logging.getLogger(__name__)

_CHROME_INJECT_JS_RETRY_ATTEMPTS = 15
_CHROME_INJECT_JS_RETRY_DELAY_SECONDS = 5.0


@dataclass(frozen=True)
class _EvaluationOutcome:
    score: float
    actual: Any = None
    has_actual: bool = False


def _json_safe_actual(value: Any) -> Any:
    if isinstance(value, bytes):
        return {'type': 'bytes', 'length': len(value)}
    if isinstance(value, dict):
        return {key: _json_safe_actual(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_actual(item) for item in value]
    return value


async def evaluate_task(
    runtime: Any,
    evaluator_config: dict[str, Any],
    agent_conclusion: str | None,
    *,
    cache_dir: Path | None = None,
    client_password: str | None = None,
) -> float:
    """Evaluate an OSWorld task using WAA-style getter/metric orchestration."""

    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix='nexus-osworld-evaluator-'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    outcome = await _evaluate_task_outcome(
        runtime, evaluator_config, agent_conclusion, cache_dir, client_password=client_password
    )
    return outcome.score


async def _evaluate_task_outcome(
    runtime: Any,
    evaluator_config: dict[str, Any],
    agent_conclusion: str | None,
    cache_dir: Path,
    *,
    client_password: str | None = None,
) -> _EvaluationOutcome:
    """Evaluate a task and retain its observed state in the result payload."""

    postconfig = evaluator_config.get('postconfig') or []
    if postconfig:
        await _run_postconfig(runtime, postconfig, cache_dir, client_password=client_password)

    func_name = evaluator_config.get('func', '')
    if func_name == 'infeasible':
        return _EvaluationOutcome(score=1.0 if agent_conclusion == 'FAIL' else 0.0)
    if agent_conclusion == 'FAIL':
        return _EvaluationOutcome(score=0.0)

    if isinstance(func_name, list):
        return await _evaluate_multi_metric(runtime, evaluator_config, func_name, cache_dir)
    return await _evaluate_single_metric(runtime, evaluator_config, str(func_name), cache_dir)


async def _evaluate_single_metric(
    runtime: Any,
    evaluator_config: dict[str, Any],
    func_name: str,
    cache_dir: Path,
) -> _EvaluationOutcome:
    metric_fn = METRIC_REGISTRY.get(func_name)
    if metric_fn is None:
        raise ValueError(f'Unknown OSWorld metric function: {func_name}')

    try:
        result_state = await _get_state(runtime, evaluator_config.get('result'), cache_dir)
    except FileNotFoundError:
        logger.exception('OSWorld result getter file not found for metric %s', func_name)
        return _EvaluationOutcome(score=0.0)
    except Exception as exc:
        raise EvaluatorInfrastructureError(
            f'OSWorld result getter failed for {func_name} ({type(exc).__name__})'
        ) from exc

    try:
        expected_state = await _get_state(runtime, evaluator_config.get('expected'), cache_dir)
    except Exception as exc:
        raise EvaluatorInfrastructureError(
            f'OSWorld expected getter failed for {func_name} ({type(exc).__name__})'
        ) from exc

    options = evaluator_config.get('options') or {}
    score = await _call_metric(func_name, metric_fn, result_state, expected_state, options)
    return _EvaluationOutcome(score=score, actual=result_state, has_actual=True)


async def _evaluate_multi_metric(
    runtime: Any,
    evaluator_config: dict[str, Any],
    func_names: list[str],
    cache_dir: Path,
) -> _EvaluationOutcome:
    conj = evaluator_config.get('conj', 'and')
    n = len(func_names)
    result_configs = _coerce_metric_list(evaluator_config.get('result'), n)
    expected_configs = _coerce_metric_list(evaluator_config.get('expected'), n)
    options_list = _coerce_metric_list(evaluator_config.get('options'), n, default={})

    scores: list[float] = []
    for idx, func_name in enumerate(func_names):
        metric_fn = METRIC_REGISTRY.get(func_name)
        if metric_fn is None:
            raise ValueError(f'Unknown OSWorld metric function: {func_name}')

        try:
            result_state = await _get_state(runtime, result_configs[idx], cache_dir)
        except FileNotFoundError:
            logger.exception('OSWorld getter failed for multi metric %s[%d]', func_name, idx)
            if conj == 'and':
                return _EvaluationOutcome(score=0.0)
            scores.append(0.0)
            continue
        except Exception as exc:
            raise EvaluatorInfrastructureError(
                f'OSWorld result getter failed for {func_name}[{idx}] ({type(exc).__name__})'
            ) from exc
        try:
            expected_state = await _get_state(runtime, expected_configs[idx], cache_dir)
        except Exception as exc:
            raise EvaluatorInfrastructureError(
                f'OSWorld expected getter failed for {func_name}[{idx}] ({type(exc).__name__})'
            ) from exc

        score = await _call_metric(func_name, metric_fn, result_state, expected_state, options_list[idx] or {})
        if conj == 'and' and score == 0.0:
            return _EvaluationOutcome(score=0.0)
        if conj == 'or' and score == 1.0:
            return _EvaluationOutcome(score=1.0)
        scores.append(score)

    if not scores:
        return _EvaluationOutcome(score=0.0)
    return _EvaluationOutcome(score=sum(scores) / len(scores) if conj == 'and' else max(scores))


def _coerce_metric_list(value: Any, length: int, *, default: Any = None) -> list[Any]:
    if isinstance(value, list):
        items = list(value)
    elif value is None:
        items = [default] * length
    else:
        items = [value] * length
    if len(items) < length:
        items.extend([default] * (length - len(items)))
    return items


async def _get_state(runtime: Any, config: dict[str, Any] | None, cache_dir: Path) -> Any:
    if not config:
        return None
    getter_type = config.get('type', '')
    getter_fn = GETTER_REGISTRY.get(str(getter_type))
    if getter_fn is None:
        raise ValueError(f'Unknown OSWorld getter type: {getter_type}')
    return await getter_fn(runtime, {**config, CACHE_DIR_CONFIG_KEY: str(cache_dir)})


async def _call_metric(
    func_name: str,
    metric_fn: Any,
    result_state: Any,
    expected_state: Any,
    options: dict[str, Any],
) -> float:
    try:
        if expected_state is None:
            score = await asyncio.to_thread(metric_fn, result_state, **options)
        else:
            score = await asyncio.to_thread(metric_fn, result_state, expected_state, **options)
    except OfficialMetricUnavailableError:
        raise
    except EvaluatorInfrastructureError:
        raise
    except ImportError as exc:
        raise EvaluatorInfrastructureError(
            f'OSWorld metric {func_name!r} could not import an evaluator dependency ({type(exc).__name__})'
        ) from exc
    except NotImplementedError:
        raise
    except Exception:
        logger.exception('OSWorld metric %s failed', func_name)
        return 0.0
    return float(score)


async def evaluate_os_world(
    runtime: Any,
    evaluator: dict[str, Any] | None,
    *,
    unsupported_policy: str = 'skip',
    last_action: str | dict[str, Any] | None = None,
    client_password: str | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one OSWorld evaluator block and return its score, status and observed state."""

    if not evaluator:
        return {'score': 0.0, 'supported': False, 'error': 'No evaluator configured'}

    func = evaluator.get('func')
    last_action_type = _last_action_type(last_action)
    # Use an explicit per-task cache or allocate a fresh isolated directory.
    requested = str(cache_dir).strip() if cache_dir is not None else ''
    if requested:
        resolved_cache_dir = Path(requested)
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        resolved_cache_dir = Path(tempfile.mkdtemp(prefix='nexus-osworld-evaluator-'))
    try:
        outcome = await _evaluate_task_outcome(
            runtime,
            evaluator,
            last_action_type,
            resolved_cache_dir,
            client_password=client_password,
        )
        result: dict[str, Any] = {
            'score': outcome.score,
            'supported': True,
            'func': func,
            'cache_dir': str(resolved_cache_dir),
        }
        if outcome.has_actual:
            result['actual'] = _json_safe_actual(outcome.actual)
        if last_action_type is not None:
            result['last_action'] = last_action_type
        return result
    except (NotImplementedError, ValueError) as exc:
        if unsupported_policy == 'fail':
            raise
        return {
            'score': 0.0,
            'supported': False,
            'error': str(exc),
            'func': func,
            'cache_dir': str(resolved_cache_dir),
        }


def evaluator_result_json(result: dict[str, Any] | None) -> str:
    return json.dumps(result or {}, ensure_ascii=False, sort_keys=True)


async def _safe_actual_state(runtime: Any, evaluator: dict[str, Any], cache_dir: Path) -> Any:
    if evaluator.get('func') == 'infeasible' or 'result' not in evaluator:
        return None
    try:
        result_config = evaluator.get('result')
        if isinstance(result_config, list):
            return [await _get_state(runtime, item, cache_dir) for item in result_config]
        return await _get_state(runtime, result_config, cache_dir)
    except Exception:
        logger.debug('Failed to collect OSWorld actual state for result payload', exc_info=True)
        return None


def _last_action_type(action: str | dict[str, Any] | None) -> str | None:
    if isinstance(action, dict):
        value = action.get('action_type')
        return str(value) if value is not None else None
    return None if action is None else str(action)


async def _run_postconfig(
    runtime: Any,
    actions: list[dict[str, Any]],
    cache_dir: Path,
    *,
    client_password: str | None = None,
) -> list[dict[str, Any]]:
    # Placeholders are resolved once per postconfig block, not per action: the screen probe is
    # a round trip to the VM. The caller configured the VM, so it knows the real password; the
    # default is the public benchmark image's login password.
    client_password = client_password or 'password'
    screen_size = (1280, 800)
    results = []
    for action in actions:
        result = await _run_postconfig_action(
            runtime, action, cache_dir, client_password=client_password, screen_size=screen_size
        )
        results.append(result)
        parameters = action.get('parameters') or {}
        stdout_name = parameters.get('stdout')
        if stdout_name:
            await write_text(cache_dir / str(stdout_name), str(result.get('stdout', '')))
        stderr_name = parameters.get('stderr')
        if stderr_name:
            await write_text(cache_dir / str(stderr_name), str(result.get('stderr', '')))
    return results


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {'0', 'false', 'no', 'off'}
    return bool(value)


def _chrome_inject_js_source(parameters: dict[str, Any]) -> str:
    for key in ('js', 'script', 'code'):
        value = parameters.get(key)
        if value is not None:
            source = str(value)
            if source.strip():
                return source
    raise ValueError(
        'Missing JavaScript for OSWorld postconfig chrome_inject_js action; expected one of: js, script, code'
    )


def _chrome_inject_js_target_index(parameters: dict[str, Any]) -> int:
    value = parameters.get('target_index', parameters.get('tab_index', parameters.get('tab', 0)))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'OSWorld postconfig chrome_inject_js target_index must be an integer, got {value!r}') from exc


async def _run_chrome_inject_js(
    runtime: Any,
    *,
    script: str,
    wait_for_load: bool,
    target_index: int,
    all_pages: bool,
    timeout_ms: int,
) -> dict[str, Any]:
    async with connect_chromium(runtime) as browser:
        pages = [
            page for context in getattr(browser, 'contexts', []) or [] for page in getattr(context, 'pages', []) or []
        ]
        if not pages:
            return {
                'success': True,
                'stdout': '',
                'stderr': 'chrome_inject_js: no open tabs to inject into',
            }

        selected_pages = list(enumerate(pages))
        if not all_pages:
            index = target_index if 0 <= target_index < len(pages) else 0
            selected_pages = [(index, pages[index])]

        warnings: list[str] = []
        failures: list[str] = []
        evaluated = 0
        for index, page in selected_pages:
            if wait_for_load:
                try:
                    await page.wait_for_load_state('load', timeout=timeout_ms)
                except Exception as exc:
                    warnings.append(f'chrome_inject_js: wait_for_load_state failed for tab {index}: {exc}')
            try:
                await page.evaluate(script)
                evaluated += 1
            except Exception as exc:
                failures.append(f'chrome_inject_js: failed to evaluate JS in tab {index}: {exc}')

        return {
            'success': not failures,
            'stdout': f'evaluated JS in {evaluated} tab(s)',
            'stderr': '\n'.join([*warnings, *failures]),
        }


async def _run_chrome_inject_js_postconfig(runtime: Any, parameters: dict[str, Any]) -> dict[str, Any]:
    script = _chrome_inject_js_source(parameters)
    wait_for_load = _coerce_bool(parameters.get('wait_for_load'), default=True)
    all_pages = _coerce_bool(parameters.get('all_pages'), default=False)
    target_index = _chrome_inject_js_target_index(parameters)
    timeout_seconds = float(parameters.get('timeout', 60))
    if timeout_seconds < 0:
        raise ValueError('OSWorld postconfig chrome_inject_js timeout must be non-negative')
    timeout_ms = int(timeout_seconds * 1000)

    last_error: Exception | None = None
    for attempt in range(_CHROME_INJECT_JS_RETRY_ATTEMPTS):
        try:
            result = await _run_chrome_inject_js(
                runtime,
                script=script,
                wait_for_load=wait_for_load,
                target_index=target_index,
                all_pages=all_pages,
                timeout_ms=timeout_ms,
            )
            return {'type': 'chrome_inject_js', **result}
        except Exception as exc:
            last_error = exc
            if attempt + 1 < _CHROME_INJECT_JS_RETRY_ATTEMPTS:
                await asyncio.sleep(_CHROME_INJECT_JS_RETRY_DELAY_SECONDS)
    raise RuntimeError(
        f'OSWorld postconfig chrome_inject_js failed after {_CHROME_INJECT_JS_RETRY_ATTEMPTS} attempts: {last_error}'
    )


async def _run_postconfig_action(
    runtime: Any,
    action: dict[str, Any],
    cache_dir: Path,
    *,
    client_password: str = 'password',
    screen_size: tuple[int, int] = (1280, 800),
) -> dict[str, Any]:
    action_type = str(action.get('type', ''))
    parameters = action.get('parameters') or {}
    if not isinstance(parameters, dict):
        raise ValueError(f'OSWorld postconfig parameters must be an object for {action_type!r}')

    def _expand(command: Any) -> Any:
        """Postconfig commands are executed literally."""
        return command

    if action_type == 'sleep':
        seconds = float(parameters.get('seconds', 0))
        await asyncio.sleep(seconds)
        return {'type': action_type, 'success': True, 'stdout': '', 'stderr': ''}

    if action_type == 'download':
        user_command_fn = (
            (lambda command: command)
            if getattr(runtime, 'osworld_command_as_user', None) is False
            else osworld_user_command
        )
        result = await run_setup_action(runtime, action, user_command_fn=user_command_fn, cache_dir=cache_dir)
        return {**result, 'stdout': '', 'stderr': ''}

    if action_type in {'execute', 'command'}:
        command = command_to_shell(_expand(parameters.get('command')))
        result = await run_bash(runtime, command, timeout=int(parameters.get('timeout', 120)))
        return {
            'type': action_type,
            'success': _result_success(result),
            'stdout': response_stdout(result),
            'stderr': response_stderr(result),
        }

    if action_type == 'launch':
        raw_command = parameters.get('command')
        if raw_command is None:
            raise ValueError('Missing command for OSWorld postconfig launch action')
        command = command_to_shell(
            _expand(add_chrome_proxy_arg(raw_command, use_proxy=osworld_runtime_use_proxy(runtime)))
        )
        result = await run_bash(runtime, f'nohup {command} >/tmp/osworld-postconfig-launch.log 2>&1 &', timeout=10)
        return {
            'type': action_type,
            'success': _result_success(result),
            'stdout': response_stdout(result),
            'stderr': response_stderr(result),
        }

    if action_type == 'open':
        path = str(parameters['path'])
        result = await run_bash(
            runtime, f'xdg-open {shlex.quote(path)} >/tmp/osworld-postconfig-open.log 2>&1 &', timeout=10
        )
        return {
            'type': action_type,
            'success': _result_success(result),
            'stdout': response_stdout(result),
            'stderr': response_stderr(result),
        }

    if action_type == 'activate_window':
        window_name = str(parameters['window_name'])
        flags = (
            '-'
            + ('x' if parameters.get('by_class', False) else '')
            + ('F' if parameters.get('strict', False) else '')
            + 'a'
        )
        result = await run_bash(runtime, f'wmctrl {flags} {shlex.quote(window_name)}', timeout=30)
        return {
            'type': action_type,
            'success': _result_success(result),
            'stdout': response_stdout(result),
            'stderr': response_stderr(result),
        }

    if action_type == 'close_window':
        window_name = str(parameters['window_name'])
        flags = (
            '-'
            + ('x' if parameters.get('by_class', False) else '')
            + ('F' if parameters.get('strict', False) else '')
            + 'c'
        )
        result = await run_bash(runtime, f'wmctrl {flags} {shlex.quote(window_name)}', timeout=30)
        return {
            'type': action_type,
            'success': _result_success(result),
            'stdout': response_stdout(result),
            'stderr': response_stderr(result),
        }

    if action_type == 'chrome_open_tabs':
        urls = parameters.get('urls_to_open') or parameters.get('urls') or []
        if not isinstance(urls, list):
            raise ValueError('chrome_open_tabs expects urls_to_open to be a list')
        command = ' '.join(['google-chrome', '--new-window', *(shlex.quote(str(url)) for url in urls)])
        result = await run_bash(runtime, f'nohup {command} >/tmp/osworld-postconfig-chrome.log 2>&1 &', timeout=10)
        return {
            'type': action_type,
            'success': _result_success(result),
            'stdout': response_stdout(result),
            'stderr': response_stderr(result),
        }

    if action_type == 'chrome_close_tabs':
        return {'type': action_type, 'success': True, 'stdout': '', 'stderr': ''}

    if action_type == 'chrome_inject_js':
        return await _run_chrome_inject_js_postconfig(runtime, parameters)

    raise NotImplementedError(f'Unsupported OSWorld postconfig type: {action_type}')


def _result_success(result: Any) -> bool:
    if isinstance(result, dict):
        code = result.get('returncode', result.get('return_code'))
        return code in (0, None)
    code = getattr(result, 'return_code', None)
    return code in (0, None)
