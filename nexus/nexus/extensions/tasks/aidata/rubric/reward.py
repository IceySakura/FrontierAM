"""Rubric task 级 reward 解析。

默认路径保持 ``compute_scores`` + :class:`ScorePolicy` 的既有行为。 当
``rubric.config.reward.script`` 存在时，在所有 checks 完成后执行 case 随包
携带的脚本，并把脚本 stdout 解析为最终 reward。
"""

from __future__ import annotations

import asyncio
import json
import math
import shlex

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .policy import ScorePolicy, finalize_score
from ..runtime.sandbox import is_case_asset_shippable


_REWARD_INPUT_VERSION = '1.0'
_DEFAULT_REWARD_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class RewardOutcome:
    """最终 reward 及其可复现配置。"""

    score: float
    continuous_score: float
    policy: dict[str, Any]


class RewardScriptError(RuntimeError):
    """Reward 脚本配置、执行或输出不满足契约。"""

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class _RewardScriptSpec:
    script: str
    timeout_sec: float

    def policy_dict(self) -> dict[str, Any]:
        return {
            'mode': 'script',
            'script': self.script,
            'timeout_sec': self.timeout_sec,
        }


def _parse_reward_script_spec(rubric: dict[str, Any]) -> _RewardScriptSpec | None:
    config = rubric.get('config') or {}
    if not isinstance(config, dict):
        raise RewardScriptError('rubric.config must be a mapping', kind='invalid_config')

    raw = config.get('reward')
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RewardScriptError('rubric.config.reward must be a mapping', kind='invalid_config')
    if config.get('score') is not None:
        raise RewardScriptError(
            'rubric.config.reward and rubric.config.score are mutually exclusive',
            kind='invalid_config',
        )

    script_raw = raw.get('script')
    if not isinstance(script_raw, str) or not script_raw.strip():
        raise RewardScriptError('rubric.config.reward.script must be a non-empty string', kind='invalid_config')
    script = script_raw.strip()
    path = PurePosixPath(script)
    if path.is_absolute() or '..' in path.parts or path.as_posix() in {'', '.'}:
        raise RewardScriptError(
            'rubric.config.reward.script must be a relative path inside the case directory',
            kind='invalid_config',
        )
    if path.suffix not in {'.py', '.sh'}:
        raise RewardScriptError(
            'rubric.config.reward.script must point to a .py or .sh file',
            kind='invalid_config',
        )

    timeout_raw = raw.get('timeout_sec', _DEFAULT_REWARD_TIMEOUT_S)
    if isinstance(timeout_raw, bool):
        raise RewardScriptError('rubric.config.reward.timeout_sec must be a positive number', kind='invalid_config')
    try:
        timeout_sec = float(timeout_raw)
    except (TypeError, ValueError) as exc:
        raise RewardScriptError(
            'rubric.config.reward.timeout_sec must be a positive number',
            kind='invalid_config',
        ) from exc
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise RewardScriptError('rubric.config.reward.timeout_sec must be a positive number', kind='invalid_config')

    return _RewardScriptSpec(script=path.as_posix(), timeout_sec=timeout_sec)


def _script_command(script: str) -> str:
    quoted = shlex.quote(script)
    if script.endswith('.py'):
        return f'python3 {quoted}'
    return f'bash {quoted}'


def _command_succeeded(result: Any) -> bool:
    status = getattr(getattr(result, 'status', None), 'value', getattr(result, 'status', None))
    return_code = getattr(result, 'return_code', None)
    if return_code is None:
        return_code = getattr(result, 'exit_code', None)
    return status in (None, 'Finished') and return_code in (0, None)


def _parse_script_score(stdout: Any) -> float:
    text = str(stdout or '').strip()
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RewardScriptError(
            'reward script stdout must be a JSON number or an object containing numeric `score`',
            kind='invalid_output',
        ) from exc
    if isinstance(value, dict):
        value = value.get('score')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RewardScriptError(
            'reward script stdout must contain a numeric score',
            kind='invalid_output',
        )
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RewardScriptError('reward script score must be finite and within [0, 1]', kind='invalid_output')
    return score


def _apply_hard_gate(score: float, rubric: dict[str, Any], score_summary: dict[str, Any]) -> float:
    if not score_summary.get('hard_gate_triggered'):
        return score
    config = rubric.get('config') or {}
    try:
        cap = float(config.get('hard_gate_cap') or 0.0)
    except (TypeError, ValueError):
        cap = 0.0
    if not math.isfinite(cap):
        cap = 0.0
    return min(score, max(0.0, min(1.0, cap)))


async def resolve_reward(
    rubric: dict[str, Any],
    rubric_results: list[dict[str, Any]],
    score_summary: dict[str, Any],
    *,
    runtime: Any,
    cwd: str,
    state_summary: dict[str, Any] | None = None,
) -> RewardOutcome:
    """把 check 结果解析成 task 最终 reward。

    未配置 ``config.reward`` 时完全沿用 ScorePolicy。 脚本模式会通过 stdin
    接收版本化 JSON，并且必须在 stdout 单独输出 JSON number 或
    ``{"score": number}``。 执行或输出错误会抛出 :class:`RewardScriptError`，
    调用方应将样本标记为 reward 无效，不能回退到静态加权分。
    """
    spec = _parse_reward_script_spec(rubric)
    clamped_score = float(score_summary['score'])
    if spec is None:
        policy = ScorePolicy.resolve(rubric)
        return RewardOutcome(
            score=finalize_score(clamped_score, policy),
            continuous_score=clamped_score,
            policy=policy.as_dict(),
        )

    if runtime is None or not callable(getattr(runtime, 'run_command', None)):
        raise RewardScriptError('reward script requires a runtime with run_command()', kind='no_runtime')

    payload = {
        'version': _REWARD_INPUT_VERSION,
        'rubric_results': rubric_results,
        'score_summary': score_summary,
        'state_summary': state_summary or {},
    }
    try:
        result = await runtime.run_command(
            _script_command(spec.script),
            cwd=cwd,
            timeout=spec.timeout_sec,
            stdin=json.dumps(payload, ensure_ascii=False, default=str),
        )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise RewardScriptError(
            f'reward script timed out after {spec.timeout_sec:g}s',
            kind='timeout',
        ) from exc
    except Exception as exc:
        raise RewardScriptError(f'reward script execution failed: {exc}', kind='execution_failed') from exc

    if not _command_succeeded(result):
        status = getattr(getattr(result, 'status', None), 'value', getattr(result, 'status', None))
        return_code = getattr(result, 'return_code', None)
        if return_code is None:
            return_code = getattr(result, 'exit_code', None)
        stderr = str(getattr(result, 'stderr', None) or '').strip()
        raise RewardScriptError(
            f'reward script failed status={status!r} return_code={return_code!r}; stderr head: {stderr[:400]!r}',
            kind='nonzero_exit',
        )

    score = _apply_hard_gate(_parse_script_score(getattr(result, 'stdout', None)), rubric, score_summary)
    return RewardOutcome(
        score=score,
        continuous_score=score,
        policy=spec.policy_dict(),
    )


def validate_reward_config(
    rubric: dict[str, Any],
    *,
    case_dir: Path | None = None,
    rubric_rel: str = 'rubric.yaml',
) -> list[str]:
    """返回 reward 脚本配置的静态校验错误。"""
    try:
        spec = _parse_reward_script_spec(rubric)
    except RewardScriptError as exc:
        return [f'{rubric_rel}: {exc}']
    if spec is None or case_dir is None:
        return []
    if not is_case_asset_shippable(spec.script):
        return [f'{rubric_rel}: config.reward.script is not a shippable case asset: {spec.script}']
    script_path = case_dir / spec.script
    if not script_path.is_file():
        return [f'{rubric_rel}: config.reward.script file not found: {spec.script}']
    return []


__all__ = ['RewardOutcome', 'RewardScriptError', 'resolve_reward', 'validate_reward_config']
