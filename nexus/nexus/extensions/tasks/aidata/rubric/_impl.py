"""兼容 shim —— 原 _impl.py 已按职责拆分为多个子模块。

本文件只做 re-export，让 ``from ...rubric._impl import <symbol>`` 继续可用，
同时保持 ``rubric/__init__.py`` 与 ``engine/rubric.py`` 已经显式 re-export
的公开 API 语义完全不变。

按新拆分：

- ``config``               —— RubricLimits / JunitLookupContext / VerifierResult
                              及运行级常量、并发 resolver。
- ``selectors``            —— trajectory / docs / 各种 state 切片文本装配。
- ``judge``                —— LLM Y/N 判官（同步 urllib + retry）。
- ``sql``                  —— sqlite / state_sql 探针层。
- ``jmespath_predicates``  —— JMESPath 断言分派 + 数值比较协助。
- ``tool_matcher``         —— tool_call step-spec 通用匹配器。
- ``legacy_v2``            —— v2 verifier 注册表 + shell/eq/regex/llm_judge
                              + ``_run_v2_check``。
- ``primitives/*``         —— v4.2 六大 assert 原语。
- ``dispatch``             —— ``run_rubric_async`` / ``run_rubric`` /
                              ``compute_scores`` / v4.2 分派 + 共享 helper。
- ``policy``               —— ScorePolicy / finalize_score。
"""
# ruff: noqa: F401,F403

import logging as _logging

from .config import (
    _DEFAULT_LIMITS,
    _DEFAULT_RUBRIC_CONCURRENCY,
    _INTERNAL_STATE_KEYS,
    _RUBRIC_LEGACY_MODE,
    JunitLookupContext,
    RubricLimits,
    VerifierFn,
    VerifierResult,
    _resolve_concurrency,
)
from .dispatch import (
    _ASSERT_EVALUATORS,
    _accepts_limits,
    _collect_hard_gate_ids,
    _dotted_get,
    _manifest_entry_legacy,
    _manifest_entry_v42,
    _result_record,
    _rubric_is_v42,
    _run_v42_check,
    _v42_result_record,
    compute_scores,
    run_rubric,
    run_rubric_async,
)
from .jmespath_predicates import (
    _COUNT_OPS,
    _JMESPATH_PREDICATE_KEYS,
    _check_jmespath_predicate,
    _coerce_numeric_equal,
    _is_truthy_jmespath_result,
    _jmespath_compile_cached,
)
from .judge import (
    _JUDGE_SYSTEM_PROMPT,
    _RAW_RESPONSE_HEAD_CHARS,
    _judge_once,
    _llm_judge,
)
from .legacy_v2 import (
    _VERIFIERS,
    _run_v2_check,
    _v_eq,
    _v_llm_judge,
    _v_regex,
    _v_shell,
    register_verifier,
)
from .policy import (
    _DEFAULT_SCORE_THRESHOLD,
    ScorePolicy,
    _env_bool,
    _env_float,
    finalize_score,
)
from .primitives.judge_claim import _eval_judge_claim
from .primitives.script_probe import _eval_script_probe
from .primitives.state_equals import (
    _eval_state_equals,
    _eval_state_equals_jmespath,
)
from .primitives.test_result import _eval_test_result
from .primitives.text_contains import (
    _eval_text_contains,
    _eval_text_not_contains,
)
from .primitives.tool_called import (
    _eval_tool_called,
    _eval_tool_called_sequence,
)
from .selectors import (
    _DOCUMENT_SLICE_KEYS,
    _DOCUMENT_TYPE_FIELDS,
    _document_matches_type,
    _document_type_from_source_ref,
    _dump_json_block,
    _gather_docs,
    _gather_source,
    _gather_trajectory,
    _iter_document_items,
    _resolve_state_source,
    _text_source_selector,
    _trajectory_messages_for_scope,
)
from .sql import (
    _SQL_PROBE_SNIPPET,
    _STATE_SQLITE_CACHE,
    _compare_sql_scalar,
    _load_state_sql_tables,
    _parse_state_sql_namespaces,
    _quote_sqlite_identifier,
    _resolve_sql_runtime,
    _rubric_sqlite_path,
    _snapshot_sql_scalar,
    _snapshot_sql_scalar_via_runtime,
    _snapshot_state_sql_scalar,
    _state_sql_cache,
    _state_sql_cell,
    _state_sql_columns_from_schema,
    _state_sql_rows,
    _state_sqlite_connection,
)
from .tool_matcher import (
    MATCHER_REGISTRY,
    _acceptable_tool_names,
    _acceptable_tools_shape_valid,
    _coerce_to_str,
    _eval_arg_empty,
    _eval_arg_not_in,
    _eval_with_text,
    _match_tool_call,
    _normalise_tool_name,
    _tool_call_args,
    _tool_call_field,
    _tool_name_matches,
)


# 若旧测试或外部调用方仍以 module attr 形式访问 ``_llm_judge``（常见于
# ``monkeypatch.setattr(_impl, '_llm_judge', fake)``），上面的
# ``from .judge import _llm_judge`` 已经把它作为本模块 attr 暴露；
# 语义与原来一致。
_logger = _logging.getLogger('nexus.extensions.tasks.aidata.rubric')


# 保持与原 _impl.py 相同的 __all__ —— ``rubric/__init__.py`` 的
# ``from ._impl import *`` 仍会拿到相同的公开符号集合。
__all__ = [
    'compute_scores',
    'finalize_score',
    'JunitLookupContext',
    'register_verifier',
    'RubricLimits',
    'run_rubric',
    'run_rubric_async',
    'ScorePolicy',
]
