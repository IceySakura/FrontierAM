"""aidata rubric+case schema 常量的单一真相源。

这些常量在以下模块之间共享：
- CLI 校验器： aidata.py::_validate_case （在 `aidata validate` 时调用）
- 运行时加载器： case/spec.py::_parse_case_metadata + load_case_rubric

在此次合并之前， aidata.py 与 case/spec.py 各自定义了自己的
副本， 两边发生了漂移（例如新增的 assert 原语只加到了其中一边）。
现在所有的 schema 决策都放在此处； 两个消费者都从本模块导入。
"""

import re


# --- rubric v4.2 assert 原语 ---------------------------------------
V42_ASSERT_TYPES: frozenset[str] = frozenset({
    'state_equals',
    'text_contains',
    'tool_called',
    'tool_called_sequence',
    'judge_claim',
    'test_result',
    'script_probe',
})

# --- rubric v2 遗留 verify kinds ---------------------------------------
# 保留用于向后兼容； 新的 rubric 必须使用 v4.2 （在 validate 阶段会拒绝 v2 ）。
LEGACY_VERIFY_KINDS: frozenset[str] = frozenset({
    'llm_judge',
    'eq',
    'regex',
    'shell',
})

# --- rubric 维度标签 -----------------------------------------------
ALLOWED_DIMENSIONS: frozenset[str] = frozenset({
    'correctness',
    'functional',
    'robustness',
})

# --- 版本字面量 ----------------------------------------------------
# 表示 “这是 v2 rubric ” 的取值 —— 用于在 validate 阶段拒绝新的 v2 ，
# 并在 run_rubric_async 加载时驱动 DeprecationWarning 。
V2_VERSION_LITERALS: frozenset = frozenset({'2', '2.0', 2})

# --- 元数据字段正则 ----------------------------------------------
# case.yaml / task.yaml 顶层元数据： created_at 日期
METADATA_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

# --- 允许的 agent-tool 分类 ---------------------------------------
# 与 ``engine.case_spec._ALLOWED_AGENT_TOOLS_BY_CATEGORY`` 保持一致。 每个 key
# 是 ``nexus/extensions/tools/`` 下的一个目录； value 为该目录内的文件名主干
# （即工具注册表 ID ）。 放在 schema 模块中， 使
# ``aidata validate`` 无需导入 engine 即可运行。
ALLOWED_AGENT_TOOLS_BY_CATEGORY: dict[str, frozenset[str]] = {
    'web_search': frozenset({
        'google_serper_search',
        'google_search',
        'hunyuan_search',
        'visit',
        'visit_image',
    }),
}

# --- 私有名 别名 （用于 shim 兼容） ----------------------
# aidata.py + case/spec.py 过去将这些名字前缀写作下划线。
# 我们保留那些拼写作为别名， 使得任何从 aidata.py 导入这些
# 私有名的外部代码在重构之后仍然可用。
_V42_ASSERT_TYPES = V42_ASSERT_TYPES
_LEGACY_VERIFY_KINDS = LEGACY_VERIFY_KINDS
_ALLOWED_DIMENSIONS = ALLOWED_DIMENSIONS
_METADATA_DATE_RE = METADATA_DATE_RE
_ALLOWED_AGENT_TOOLS_BY_CATEGORY = ALLOWED_AGENT_TOOLS_BY_CATEGORY

__all__ = [
    'V42_ASSERT_TYPES',
    'LEGACY_VERIFY_KINDS',
    'ALLOWED_DIMENSIONS',
    'V2_VERSION_LITERALS',
    'METADATA_DATE_RE',
    'ALLOWED_AGENT_TOOLS_BY_CATEGORY',
    # 下划线 别名 （兼容）
    '_V42_ASSERT_TYPES',
    '_LEGACY_VERIFY_KINDS',
    '_ALLOWED_DIMENSIONS',
    '_METADATA_DATE_RE',
    '_ALLOWED_AGENT_TOOLS_BY_CATEGORY',
]
