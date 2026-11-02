"""兼容 shim：engine.rubric 已拆分至 aidata.rubric 子包。

新代码请从 `nexus.extensions.tasks.aidata.rubric` 导入。
本 shim 重新导出完整的公共 API 与被测的私有符号表面， 以保持对现有测试
和下游消费者的向后兼容。
"""

# noqa: F401
from ..rubric import *  # noqa: F403

# Explicit re-export of tested private symbols so `from ...engine.rubric import _foo` works.
from ..rubric import (  # noqa: F401
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
