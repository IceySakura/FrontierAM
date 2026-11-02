"""最终得分坍缩策略（binary / continuous / threshold）。

独立模块 —— 不依赖任何其它新文件。
"""

import logging
import os

from dataclasses import dataclass
from typing import Any


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 最终得分坍缩策略（binary / continuous / threshold）
# ---------------------------------------------------------------------------

_DEFAULT_SCORE_THRESHOLD = 1.0


def _env_bool(name: str) -> bool | None:
    """解析 ``AIDATA_*`` 布尔类型环境变量。

    未设置时返回 ``None``，方便调用方回退到 rubric 配置或默认值。 ``'0'`` /
    ``'false'`` / ``'no'`` / ``'off'`` / 空 → ``False``；其它任何值 → ``True``
    （与 ``_state_dump_enabled`` 风格保持一致）。
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() not in {'', '0', 'false', 'no', 'off'}


def _env_float(name: str) -> float | None:
    """解析 ``AIDATA_*`` 浮点类型环境变量；未设置 / 空白 / 无效时返回 ``None``。"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        _logger.warning('[rubric.score] %s=%r not a float; ignoring', name, raw)
        return None


@dataclass(frozen=True)
class ScorePolicy:
    """连续的 hard-gate 后得分如何坍缩为最终 task 得分。

    三个正交开关（优先级：env > ``rubric['config']['score']`` > 默认值）：

    - ``threshold`` （float，默认 1.0）：应用阈值时的截断点。
    - ``continuous`` （bool，默认 True）：输出连续分数而非二值化到 0/1。
    - ``use_threshold`` （bool，默认 False）：是否应用截断。

    组合语义（详见 :func:`finalize_score`）：

    ===========  =============  =========================================
    continuous   use_threshold  输出（``s`` = hard-gate 后的连续得分）
    ===========  =============  =========================================
    False        True           二值：``1.0 if s>=thr else 0.0``
    True         False          纯连续：``s`` （默认）
    True         True           带门控的连续：``s if s>=thr else 0.0``
    False        False          退化：透传 ``s`` （+ 警告）
    ===========  =============  =========================================

    环境变量：
      - ``AIDATA_SCORE_THRESHOLD``     （float）
      - ``AIDATA_SCORE_CONTINUOUS``    （bool）
      - ``AIDATA_SCORE_USE_THRESHOLD`` （bool）

    默认（任何位置都未设置）会输出 hard-gate 后的连续得分。
    配置 ``continuous: false`` 与 ``use_threshold: true`` 可复现二值评分；
    如果需要旧的 0.8 截断，也需一并设置 ``threshold: 0.8`` 。
    """

    threshold: float = _DEFAULT_SCORE_THRESHOLD
    continuous: bool = True
    use_threshold: bool = False

    @classmethod
    def resolve(cls, rubric: dict | None = None) -> 'ScorePolicy':
        """按 默认值 → rubric 配置 → env 的顺序构建生效的策略。"""
        threshold = cls.threshold
        continuous = cls.continuous
        use_threshold = cls.use_threshold

        # 层 1：rubric['config']['score'] （每个 case 单独设置）。
        cfg = (rubric or {}).get('config') or {}
        score_cfg = cfg.get('score') if isinstance(cfg, dict) else None
        if isinstance(score_cfg, dict):
            if 'threshold' in score_cfg:
                try:
                    threshold = float(score_cfg['threshold'])
                except (TypeError, ValueError):
                    _logger.warning(
                        '[rubric.score] config.score.threshold=%r not a float; using %s',
                        score_cfg['threshold'],
                        threshold,
                    )
            if 'continuous' in score_cfg:
                continuous = bool(score_cfg['continuous'])
            if 'use_threshold' in score_cfg:
                use_threshold = bool(score_cfg['use_threshold'])

        # 层 2：环境变量覆盖（最高优先级；便于 ``export`` ）。
        env_threshold = _env_float('AIDATA_SCORE_THRESHOLD')
        if env_threshold is not None:
            threshold = env_threshold
        env_continuous = _env_bool('AIDATA_SCORE_CONTINUOUS')
        if env_continuous is not None:
            continuous = env_continuous
        env_use_threshold = _env_bool('AIDATA_SCORE_USE_THRESHOLD')
        if env_use_threshold is not None:
            use_threshold = env_use_threshold

        return cls(threshold=threshold, continuous=continuous, use_threshold=use_threshold)

    def as_dict(self) -> dict[str, Any]:
        return {
            'threshold': self.threshold,
            'continuous': self.continuous,
            'use_threshold': self.use_threshold,
        }


def finalize_score(raw_score: float, policy: ScorePolicy | None = None) -> float:
    """将 hard-gate 后的连续得分坍缩为最终 task 得分。

    ``raw_score`` 为 ``compute_scores(...)['score']`` （hard-gate 截断后的加权通过率）。
    按 :class:`ScorePolicy` 返回相应值。
    """
    policy = policy or ScorePolicy()
    s = float(raw_score)
    if policy.continuous:
        if policy.use_threshold:
            return s if s >= policy.threshold else 0.0
        return s
    if policy.use_threshold:
        return 1.0 if s >= policy.threshold else 0.0
    _logger.warning(
        '[rubric.score] degenerate policy (continuous=False, use_threshold=False); '
        'passing through continuous score %.3f',
        s,
    )
    return s
