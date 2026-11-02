"""``tool_called`` / ``tool_called_sequence`` primitive —— 检查轨迹中
是否包含匹配 step 规格的 tool_call （或 tool_call 的有序子序列）。 二者
共享同一 step 规格接口， 均通过 ``tool_matcher._match_tool_call``。
"""

from typing import Any

from ..config import JunitLookupContext, RubricLimits, VerifierResult
from ..selectors import _trajectory_messages_for_scope
from ..tool_matcher import _acceptable_tools_shape_valid, _match_tool_call


async def _eval_tool_called(
    cfg: dict, state: dict, runtime: Any, junit_ctx: JunitLookupContext | None, *, limits: RubricLimits | None = None
) -> VerifierResult:
    """Agent 在计数区间内调用了匹配 step 规格的工具。

    必填： ``tool`` 或 ``tool_prefix`` 或 ``acceptable_tools`` （三选一）。
    其他所有 step-spec 谓词均为可选， 详见
    :func:`_match_tool_call` 的文档 （``with_text``、 ``arg_empty``、 ``arg_not_in``、
    ``all_of`` / ``any_of`` / ``not``、 ``matcher``）。

    命名： 来自 MCP 的工具在轨迹中呈现为
    ``<server_name>_<tool_name>`` （见 core/tool/mcp/loader.py —— 每个函数
    都会以 MCP server 的注册键作为前缀， 这样两个 server 就可以注册同名的
    业务工具而不冲突）。 对于 ``tool:``， 我们既接受裸业务名 （``email_send``），
    也接受完全限定形式 （``email_email_send``）； 推荐写裸名， 因为用例
    编写者不必记住 server-name 到 server-key 的映射。 ``tool_prefix:`` 仍是
    对轨迹名的原始字符串前缀匹配 （所以如果你确实想匹配整个 MCP namespace
    ， 依然可以用它锁定 ``email_email_``）。
    """
    if not (cfg.get('tool') or cfg.get('tool_prefix') or cfg.get('acceptable_tools')):
        return False, 'INVALID', 'need_tool_or_tool_prefix_or_acceptable_tools', ''
    if not _acceptable_tools_shape_valid(cfg):
        return False, 'INVALID', 'acceptable_tools_must_be_list_or_string', ''
    raw_max_count = cfg.get('max_count')
    try:
        max_count = int(raw_max_count) if raw_max_count is not None else None
        # ``max_count: 0`` 是表达 “必须不调用此工具” 的自然写法； 除非作者
        # 显式设置 min_count， 否则不要让通常的正默认值 min_count=1 使这
        # 变得不可能。
        default_min = 0 if raw_max_count is not None and max_count == 0 and 'min_count' not in cfg else 1
        min_count = int(cfg.get('min_count', default_min))
    except (TypeError, ValueError):
        return (
            False,
            'INVALID',
            (f'invalid_count_bounds min_count={cfg.get("min_count")!r} max_count={raw_max_count!r}'),
            '',
        )
    if min_count < 0 or (max_count is not None and max_count < 0):
        return False, 'INVALID', f'negative_count_bounds min={min_count} max={max_count}', ''
    if max_count is not None and min_count > max_count:
        return False, 'INVALID', f'count_bounds_conflict min={min_count} max={max_count}', ''

    count = 0
    samples: list[str] = []
    for msg in _trajectory_messages_for_scope(state, cfg.get('scope')):
        if msg.get('role') != 'assistant':
            continue
        for tc in msg.get('tool_calls') or []:
            if not _match_tool_call(cfg, tc):
                continue
            count += 1
            if len(samples) < 3:
                samples.append((tc.get('function') or {}).get('name') or '')
    passed = count >= min_count and (max_count is None or count <= max_count)
    max_part = f' max={max_count}' if max_count is not None else ''
    detail = f'matched={count} min={min_count}{max_part} samples={samples!r}'
    return passed, 'Y' if passed else 'N', detail, ''


async def _eval_tool_called_sequence(
    cfg: dict, state: dict, runtime: Any, junit_ctx: JunitLookupContext | None, *, limits: RubricLimits | None = None
) -> VerifierResult:
    """在轨迹的 tool_calls 上进行有序子序列匹配。

    必填： ``steps`` —— step 规格列表 （形状与 ``tool_called`` 顶层相同，
    详见 :func:`_match_tool_call` 的文档）。

    匹配从左到右， 使用游标进行： 每个 step 找到第一条匹配的后续调用后， 游
    标推进到其之后。 空 ``steps`` 列表平凡通过。

    评分为二元 （当且仅当每个 step 按序匹配时通过）。 detail 字符串报告
    ``matched N/M (indices=[...])``， 便于评审者查看部分运行时哪一步落
    空。 indices 列表中的 ``-1`` 表示 “前一个游标之后没有调用能匹配这一
    step”。
    """
    steps = cfg.get('steps') or []
    if not steps:
        return True, 'Y', 'empty_sequence', ''

    calls: list[dict] = []
    for msg in _trajectory_messages_for_scope(state, cfg.get('scope')):
        if msg.get('role') != 'assistant':
            continue
        for tc in msg.get('tool_calls') or []:
            calls.append(tc)

    cursor = 0
    matched = 0
    matched_idx: list[int] = []
    for step in steps:
        found = False
        for i in range(cursor, len(calls)):
            if _match_tool_call(step, calls[i]):
                cursor = i + 1
                matched += 1
                matched_idx.append(i)
                found = True
                break
        if not found:
            matched_idx.append(-1)

    all_pass = matched == len(steps)
    detail = f'matched {matched}/{len(steps)} (indices={matched_idx})'
    return all_pass, 'Y' if all_pass else 'N', detail, ''
