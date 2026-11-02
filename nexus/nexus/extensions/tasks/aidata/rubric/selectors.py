"""rubric 层的 State → text 选择器。

由 LLM 判定器、 ``text_contains`` 、 ``judge_claim`` 和 ``tool_called*``
原语使用的纯状态遍历辅助函数。 不依赖分发器（ ``_dotted_get`` 在
``_resolve_state_source`` 内部延迟导入，以打破 dispatch↔selectors 的循环依赖）。
"""

import json
import logging

from pathlib import Path
from typing import Any

from .config import _DEFAULT_LIMITS, _INTERNAL_STATE_KEYS, RubricLimits


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 源装配
# ---------------------------------------------------------------------------


def _dump_json_block(label: str, items: Any) -> str:
    if not items:
        return f'## {label}\n(empty)\n'
    if isinstance(items, list):
        body = '\n'.join(json.dumps(x, ensure_ascii=False) for x in items)
    else:
        body = json.dumps(items, ensure_ascii=False)
    return f'## {label}\n{body}\n'


def _gather_trajectory(state: dict, *, limits: RubricLimits | None = None) -> str:
    """为 rubric 判定器摊平轨迹。

    我们包含首个用户消息（其中包含任务 + 任何内联的
    上下文 / 文件清单）——否则像 " agent 读取了简历 " 这样的论断
    就没有证据可锚定，因为提取的文本是通过该消息传递的，
    而不是通过工具调用。 后续的用户消息（若有）
    会被跳过——它们通常是运行时澄清，而非任务
    素材。

    ``limits`` 控制每个跨度被截断的激进程度：

    - ``user_head_chars``      —— 初始用户消息的头部上限
      （32k 字符 ≈ 8k tokens，对典型的 PDF 简历 / 简短
      纲要 / 长 TXT 文件足够）。
    - ``tool_call_arg_chars``  —— 每个工具调用参数负载的上限。
    - ``tool_result_chars``    —— 每个 tool_result 主体的上限。

    传入 ``None`` 则使用模块级默认值（历史行为）。
    """
    lim = limits or _DEFAULT_LIMITS
    user_head_chars = lim.user_head_chars
    arg_chars = lim.tool_call_arg_chars
    result_chars = lim.tool_result_chars
    parts: list[str] = []
    first_user_shown = False
    for msg in state.get('trajectory', []) or []:
        role = msg.get('role')
        if role == 'user' and not first_user_shown:
            c = msg.get('content')
            if isinstance(c, str) and c.strip():
                head = (
                    c
                    if len(c) <= user_head_chars
                    else (c[:user_head_chars] + f'\n…(truncated, {len(c) - user_head_chars} more chars)')
                )
                parts.append(f'[user.initial_message]\n{head}')
            first_user_shown = True
        elif role == 'assistant':
            c = msg.get('content')
            if isinstance(c, str) and c.strip():
                parts.append(f'[assistant.content]\n{c}')
            for tc in msg.get('tool_calls') or []:
                fn = tc.get('function') or {}
                parts.append(f'[assistant.tool_call] {fn.get("name", "?")}({str(fn.get("arguments", ""))[:arg_chars]})')
        elif role == 'tool':
            c = msg.get('content', '')
            if isinstance(c, str) and c:
                parts.append(f'[tool_result] {c[:result_chars]}')
    return '\n'.join(parts)


_DOCUMENT_SLICE_KEYS = ('documents', 'docs', 'uploaded_docs', 'online_docs')
_DOCUMENT_TYPE_FIELDS = ('doc_type', 'type', 'kind', 'ext', 'extension', 'format')


def _iter_document_items(state: dict) -> list[Any]:
    """收集类似 document 的 dump_state 切片，不绑定单个 MCP。

    R3 状态收集已经在其命名空间下保存每个 MCP 的负载，
    并对旧键进行扁平合并。 该辅助函数将规范的
    ``documents`` 切片视为其中一个可能的生产者，然后也在每个命名空间桶内
    检查常见的 document 切片名称。
    """
    docs: list[Any] = []
    seen: set[int] = set()

    def add(raw: Any) -> None:
        if isinstance(raw, list):
            for item in raw:
                marker = id(item)
                if marker in seen:
                    continue
                seen.add(marker)
                docs.append(item)

    for key in _DOCUMENT_SLICE_KEYS:
        add(state.get(key))
    for bucket in state.values():
        if not isinstance(bucket, dict):
            continue
        for key in _DOCUMENT_SLICE_KEYS:
            add(bucket.get(key))
    return docs


def _document_matches_type(doc: Any, want_type: str) -> bool:
    if not want_type:
        return True
    if not isinstance(doc, dict):
        return False
    want = want_type.lower()
    for field in _DOCUMENT_TYPE_FIELDS:
        raw = doc.get(field)
        if raw is not None and str(raw).lower().lstrip('.') == want:
            return True
    filename = str(doc.get('filename') or doc.get('title') or doc.get('name') or '')
    suffix = Path(filename).suffix.lower().lstrip('.')
    return suffix == want


def _gather_docs(state: dict, only_type: str | None = None) -> str:
    """为判定器渲染上传 / 生成的 document 切片。

    ``docs`` 是稳定的 rubric 选择器，而非具体的 MCP 名称。 生产者
    可以在 ``documents`` （首选）或兼容的 document
    切片键下暴露一个列表，无论是扁平还是位于其 MCP 命名空间下。 ``docs:<type>``
    按常见类型字段过滤，如 ``doc_type`` / ``type`` / ``ext`` 。
    """
    docs = _iter_document_items(state)
    if only_type:
        docs = [d for d in docs if _document_matches_type(d, only_type)]
    if not docs:
        return ''
    return _dump_json_block('documents', docs)


def _document_type_from_source_ref(source_ref: dict | None) -> str | None:
    """从 ``source_ref`` 中提取显式的 document 类型过滤器。

    新的 rubric 首选 ``from: docs`` + ``source_ref: {type: excel}`` 。
    较旧的紧凑选择器 ``from: docs:excel`` 在
    :func:`_gather_source` 中仍受现有案例支持。
    """
    if not isinstance(source_ref, dict):
        return None
    candidates = [source_ref]
    where = source_ref.get('where')
    if isinstance(where, dict):
        candidates.append(where)
    for cfg in candidates:
        for key in _DOCUMENT_TYPE_FIELDS:
            raw = cfg.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def _resolve_state_source(state: dict, source: str) -> tuple[bool, Any]:
    """针对状态解析源选择器。

    出于向后兼容，精确的顶级键优先。 若不存在精确键，则点分
    选择器如 ``taobao.orders`` 或 ``tencent_docs.documents.0`` 会遍历
    从 MCP dump_state 收集的命名空间状态树。
    """
    if source in state:
        return True, state[source]
    if '.' in source:
        from .dispatch import _dotted_get

        return _dotted_get(state, source)
    return False, None


def _gather_source(
    state: dict,
    source: str,
    source_ref: dict | None,
    *,
    limits: RubricLimits | None = None,
) -> str:
    source_ref = source_ref or {}
    # 生成 / 迁移的 rubric 使用的向后兼容别名。
    #
    # 只保留了 trajectory 别名——它们解析到一个合成键
    # （ ``trajectory`` ），本函数通过下方专用分支处理，
    # 而非任何 dump_state 派生的状态键。
    #
    # 一旦 ``state.py`` 开始为每个 mock 同时写入命名空间桶和
    # 扁平切片，Mock-namespace 别名（例如 ``notes -> memos`` ）就已被移除：
    # 编写 ``from: <mock_name>`` 的 rubric 作者
    # （ README 推荐的模式）现在直接命中命名空间桶，
    # 该桶携带完整的 dump_state 负载，且不会与其他碰巧共享通用顶级
    # 键（如 ``seq`` / ``orders`` / ``messages`` ）的 mock 冲突。
    source_aliases = {
        'agent_trajectory': 'trajectory',
        'agent_log': 'trajectory',
        'agent_action': 'trajectory',
    }
    source = source_aliases.get(source, source)

    # --- 特殊数据路径（不来自 dump_state） ---
    if source == 'trajectory':
        text = _gather_trajectory(state, limits=limits)
        return f'## agent trajectory\n{text or "(empty)"}\n'
    if source == 'agent_answer':
        val = state.get('agent_answer')
        return val if isinstance(val, str) else ''
    # 通过绝对路径的任意沙箱文件： ``from: file:/workspace/.../x.md`` 。
    # 引擎在评估前会将 rubric 引用的每个 ``file:`` 路径读入
    # ``state['_sandbox_files']`` 。 使用它来评估
    # agent 持久化的交付物（例如最终 markdown 答案文件），
    # 而非从轨迹中提取的 ``agent_answer`` 。
    if source.startswith('file:'):
        path = source.split(':', 1)[1].strip()
        files = state.get('_sandbox_files')
        if isinstance(files, dict):
            val = files.get(path)
            return val if isinstance(val, str) else ''
        return ''
    if source == 'docs':
        return _gather_docs(state, only_type=_document_type_from_source_ref(source_ref))
    if source.startswith('docs:'):
        return _gather_docs(state, only_type=source.split(':', 1)[1])

    # --- "all"：聚合状态 + 轨迹中的每个键 ---
    if source == 'all':
        parts = []
        for key in sorted(state.keys()):
            if key in _INTERNAL_STATE_KEYS:
                continue
            parts.append(_gather_source(state, key, None, limits=limits))
        parts.append('## trajectory\n' + _gather_trajectory(state, limits=limits))
        return '\n\n'.join(parts)

    # --- 通用：将任意状态键 / 命名空间点分选择器转储为 JSON 块 ---
    found, val = _resolve_state_source(state, source)
    if not found:
        return ''
    doc_type = _document_type_from_source_ref(source_ref)
    if doc_type and isinstance(val, list) and source.rsplit('.', 1)[-1] in _DOCUMENT_SLICE_KEYS:
        val = [d for d in val if _document_matches_type(d, doc_type)]
    if isinstance(val, (list, dict)):
        return _dump_json_block(source, val)
    return str(val)


def _text_source_selector(cfg: dict, *, default: str = 'all') -> str:
    """解析 judge/text 断言的 text-source 选择器。

    v4.2 将 ``from`` 记录为规范字段，而较旧 / 生成的
    rubric 有时使用 ``source`` 。 当二者都存在时优先 ``from`` 。
    """
    for key in ('from', 'source'):
        raw = cfg.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return default


def _trajectory_messages_for_scope(state: dict, scope: Any = None) -> list[dict]:
    """为可选的 ``tool_called`` 作用域返回轨迹消息。

    ``initial_trajectory`` 由 ``main_agent`` 在系统消息之后插入。
    任务运行器在 state 中记录其长度，以便 rubric
    检查在不重放副作用的情况下区分种子历史与实时工具调用。
    """
    trajectory = list(state.get('trajectory', []) or [])
    if scope not in ('seed', 'live'):
        return trajectory
    try:
        seed_len = int(state.get('_initial_trajectory_len') or 0)
    except (TypeError, ValueError):
        seed_len = 0
    if seed_len <= 0:
        return [] if scope == 'seed' else trajectory
    seed_start = 1 if trajectory and trajectory[0].get('role') == 'system' else 0
    seed_end = min(seed_start + seed_len, len(trajectory))
    if scope == 'seed':
        return trajectory[seed_start:seed_end]
    return trajectory[:seed_start] + trajectory[seed_end:]
