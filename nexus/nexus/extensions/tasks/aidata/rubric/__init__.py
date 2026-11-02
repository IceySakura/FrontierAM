"""aidata rubric 子系统 —— 从单体 engine/rubric.py 拆分而来。

公共入口：run_rubric、run_rubric_async、compute_scores、ScorePolicy、
finalize_score，以及所有 _eval_* / _v_* 原语求值器（私有但受测试覆盖）。
"""

# 重新导出整个 _impl 表面 —— 来自 engine.rubric 的全部 86 个符号
from ._impl import *  # noqa: F401,F403

# 显式重新导出以下划线开头且被测试导入的符号
# （通配符默认不会拉取以下划线开头的符号；同时也用于记录 API）
from ._impl import (  # noqa: F401
    RubricLimits,
    ScorePolicy,
    VerifierResult,
    _eval_judge_claim,
    _eval_state_equals,
    _eval_state_equals_jmespath,
    _eval_test_result,
    _eval_text_contains,
    _eval_text_not_contains,
    _eval_tool_called,
    _eval_tool_called_sequence,
    _gather_source,
    _gather_trajectory,
    _judge_once,
    _llm_judge,
    _match_tool_call,
    _rubric_is_v42,
    _run_v2_check,
    _run_v42_check,
    _snapshot_sql_scalar,
    _snapshot_state_sql_scalar,
    _text_source_selector,
    _v_eq,
    _v_llm_judge,
    _v_regex,
    _v_shell,
    compute_scores,
    finalize_score,
    register_verifier,
    run_rubric,
    run_rubric_async,
)
from .fetch import Fetcher, FileSource, ProbeSpec  # noqa: F401
