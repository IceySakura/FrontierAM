"""rubric 包的基础模块。

承载运行时共享配置：``RubricLimits`` dataclass、``JunitLookupContext`` dataclass、
``VerifierResult`` typedef、并发解析器，以及横切常量（``_DEFAULT_LIMITS``、
``_INTERNAL_STATE_KEYS``、``_RUBRIC_LEGACY_MODE``、``_DEFAULT_RUBRIC_CONCURRENCY``）。

包内其他所有模块都从这里导入；本模块不依赖它们中的任何一个。
"""

import logging
import os

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


# 回滚逃生阀。 设为 '1' 可将 primitive 路由至它们的 Commit-3 之前的
# 直接状态访问实现。 在 Commit 3 中随 Fetcher 门面迁移一同引入；
# 若某个基于 fetcher 的 primitive 在任何生产用例上出现回归，
# 无需 git revert，直接切换该环境变量即可。
# 移除目标：Commit 3 平稳落地后再过一个发布周期。
_RUBRIC_LEGACY_MODE = os.environ.get('NEXUS_AIDATA_RUBRIC_LEGACY') == '1'


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 运行级 limits（可通过 ``rubric['config']`` 或传参配置）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricLimits:
    """每次运行的字符上限，用于轨迹扁平化 + judge 源裁剪。

    这四个值都作用于 “将 state 扁平化为供 judge / ``text_contains`` 使用的单一字符串”
    这条路径。 ``tool_called`` 和 ``state_equals`` 不读取这些，因为它们直接遍历原始
    ``state['trajectory']`` 结构。

    默认值已从原始硬编码值重新平衡（user_head 32000 / arg 400 / result 500 /
    source 48000），以便让 judge 有更多空间关注 agent 实际做了什么——更小的
    task-header 上限，更大的每结果与总预算。 rubric 作者可通过如下方式
    在每个 case 中覆盖::

        version: "4.2"
        config:
          trajectory:
            user_head_chars:     16000
            tool_call_arg_chars: 800
            tool_result_chars:   2000
          judge:
            source_max_chars: 96000
        checks: [...]

    优先级：外部传入的 ``RubricLimits`` （:func:`run_rubric_async` 的 kwarg）>
    rubric 中的 ``config`` 块 > 默认值。 未知配置键被忽略（前向兼容）；非正数/
    非数值将回退到默认值并附带警告，避免因笔误静默禁用裁剪。
    """

    user_head_chars: int = 16_000
    tool_call_arg_chars: int = 800
    tool_result_chars: int = 2_000
    judge_source_max_chars: int = 96_000

    @classmethod
    def from_rubric(cls, rubric: dict | None) -> 'RubricLimits':
        """将 ``rubric['config']`` 解析为 ``RubricLimits``。

        缺失/格式错误的 config 块会静默回退到默认值。
        个别错误值会按字段发出警告并同样回退。
        """
        defaults = cls()
        cfg = (rubric or {}).get('config') or {}
        if not isinstance(cfg, dict):
            return defaults
        traj_val = cfg.get('trajectory')
        traj: dict[str, Any] = traj_val if isinstance(traj_val, dict) else {}
        judge_val = cfg.get('judge')
        judge: dict[str, Any] = judge_val if isinstance(judge_val, dict) else {}

        # (section_dict, yaml_key, dataclass_field, section_label)
        spec = [
            (traj, 'user_head_chars', 'user_head_chars', 'trajectory'),
            (traj, 'tool_call_arg_chars', 'tool_call_arg_chars', 'trajectory'),
            (traj, 'tool_result_chars', 'tool_result_chars', 'trajectory'),
            (judge, 'source_max_chars', 'judge_source_max_chars', 'judge'),
        ]
        kwargs: dict[str, int] = {}
        for section, yaml_key, field_name, label in spec:
            if yaml_key not in section:
                continue
            raw = section[yaml_key]
            try:
                val = int(raw)
            except (TypeError, ValueError):
                _logger.warning(
                    '[rubric.config] %s.%s=%r not an int; using default %d',
                    label,
                    yaml_key,
                    raw,
                    getattr(defaults, field_name),
                )
                continue
            if val <= 0:
                _logger.warning(
                    '[rubric.config] %s.%s=%d must be > 0; using default %d',
                    label,
                    yaml_key,
                    val,
                    getattr(defaults, field_name),
                )
                continue
            kwargs[field_name] = val
        return cls(**kwargs) if kwargs else defaults


# 模块级的 sentinel，让不传入 limits 的调用方沿用历史行为，
# 且不必为每次调用重新分配 dataclass。
_DEFAULT_LIMITS = RubricLimits()
_INTERNAL_STATE_KEYS = frozenset({
    'trajectory',
    'agent_answer',
    '_sandbox_files',
    '_rubric_sqlite_path',
    '_rubric_sqlite_service',
    '_initial_trajectory_len',
    '_state_sqlite_cache',
})


# ---------------------------------------------------------------------------
# Verifier 注册表 typedefs
# ---------------------------------------------------------------------------


# VerifierResult 打包了 (passed, verdict, detail, raw_head)。 保留 `raw_head`
# 纯粹是为了让 LLM judge 能够继续显露模型字面回复的前 N 个字符以便调试；
# 确定性 verifier 会将其留空。 verifier 可以是同步或异步——分发器会按需 await。
VerifierResult = tuple[bool, str, str, str]
VerifierFn = Callable[..., VerifierResult | Awaitable[VerifierResult]]


@dataclass
class JunitLookupContext:
    """传递给 ``test_result`` 求值器的打包上下文，用于将
    ``check.id`` → junit ``testcase`` 查找无需重解析 XML。
    """

    report: Any  # engine.junit.JunitReport——类型放宽以规避 import 循环
    explicit_map: dict[str, str]


# ---------------------------------------------------------------------------
# 并发解析器
# ---------------------------------------------------------------------------


_DEFAULT_RUBRIC_CONCURRENCY = 8


def _resolve_concurrency(explicit: int | None) -> int:
    """为 ``run_rubric_async`` 选择并发上限。

    优先级：显式 kwarg > ``NEXUS_RUBRIC_CONCURRENCY`` 环境变量 >
    ``_DEFAULT_RUBRIC_CONCURRENCY`` （8）。 负数/非数值会回退到默认值
    并附带警告，避免因笔误静默把整个 rubric 串行化。
    """
    if explicit is not None:
        try:
            n = int(explicit)
        except (TypeError, ValueError):
            _logger.warning(
                '[rubric] concurrency=%r not an int; using default %d',
                explicit,
                _DEFAULT_RUBRIC_CONCURRENCY,
            )
            return _DEFAULT_RUBRIC_CONCURRENCY
        if n < 1:
            _logger.warning(
                '[rubric] concurrency=%d must be >=1; using default %d',
                n,
                _DEFAULT_RUBRIC_CONCURRENCY,
            )
            return _DEFAULT_RUBRIC_CONCURRENCY
        return n
    raw = os.environ.get('NEXUS_RUBRIC_CONCURRENCY')
    if raw is None or not raw.strip():
        return _DEFAULT_RUBRIC_CONCURRENCY
    try:
        n = int(raw)
    except ValueError:
        _logger.warning(
            '[rubric] NEXUS_RUBRIC_CONCURRENCY=%r not an int; using default %d',
            raw,
            _DEFAULT_RUBRIC_CONCURRENCY,
        )
        return _DEFAULT_RUBRIC_CONCURRENCY
    if n < 1:
        _logger.warning(
            '[rubric] NEXUS_RUBRIC_CONCURRENCY=%d must be >=1; using default %d',
            n,
            _DEFAULT_RUBRIC_CONCURRENCY,
        )
        return _DEFAULT_RUBRIC_CONCURRENCY
    return n
