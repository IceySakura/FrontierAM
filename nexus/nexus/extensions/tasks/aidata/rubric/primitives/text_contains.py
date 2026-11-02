"""``text_contains`` / ``text_not_contains`` 原语 —— 在文本切片上进行子串 /
正则匹配。
"""

import re

from typing import Any

from ..config import JunitLookupContext, RubricLimits, VerifierResult
from ..selectors import _gather_source, _text_source_selector


async def _eval_text_contains(
    cfg: dict, state: dict, runtime: Any, junit_ctx: JunitLookupContext | None, *, limits: RubricLimits | None = None
) -> VerifierResult:
    """在文本切片上进行子串 / 正则检查。 字段： ``from``
    （selector ，默认为 ``all`` ），以及至少一个谓词：

    - ``contains``:     单个字符串或列表（列表中所有元素都必须出现）
    - ``contains_all``: 列表，所有元素都必须出现（等同于列表形式的 ``contains`` ）
    - ``contains_any``: 列表，至少一个必须出现
    - ``text``:         ``contains`` 单字符串形式的兼容别名
    - ``keywords``:     ``contains_any`` 的兼容别名
    - ``regex``:        Python 正则；当 ``re.search`` 命中时通过。
      ``pattern`` 作为兼容别名接受，用于已有的生成 rubric ；
      当二者同时存在时以 ``regex`` 为准。
      正则本身的大小写敏感性由 pattern 控制
      （若需 ASCII 大小写折叠，可使用 ``(?i)`` 内联标志）。
      与子串变体不同， regex 始终匹配原始的、保留大小写的文本。

    多个谓词以 AND 组合 —— 每个提供的谓词都必须通过。
    ``case_sensitive: false`` （默认）会将干草堆和针都转为小写用于
    ``contains*`` 系列； regex 不受影响。

    典型的 ``from`` selector: ``all`` （默认）、 ``trajectory`` 、
    ``agent_answer`` （agent 最终的自然语言回复，由引擎在 rubric 运行前
    写入 ``state['agent_answer']`` ），加上 aidata 原生切片
    （ ``emails`` 、 ``wechat_log`` 、 ``docs`` 等） —— 详见
    :func:`_gather_source` 。
    """
    slice_name = _text_source_selector(cfg)
    text = _gather_source(state, slice_name, cfg.get('source_ref'), limits=limits)
    if not text:
        return False, 'EMPTY_SOURCE', f'empty_source ({slice_name})', ''

    case_sensitive = bool(cfg.get('case_sensitive', False))
    hay = text if case_sensitive else text.lower()

    def _needles(raw: Any) -> list[str]:
        if isinstance(raw, str):
            items = [raw]
        elif isinstance(raw, list):
            items = [str(x) for x in raw]
        else:
            return []
        return items if case_sensitive else [x.lower() for x in items]

    notes: list[str] = []
    ok = True
    has_predicate = False

    if 'contains_any' in cfg:
        has_predicate = True
        needles = _needles(cfg['contains_any'])
        if not needles:
            return False, 'INVALID', 'empty_contains_any', ''
        found = [n for n in needles if n in hay]
        ok = ok and bool(found)
        notes.append(f'any_of={needles!r} found={found!r}')
    if 'keywords' in cfg:
        has_predicate = True
        needles = _needles(cfg['keywords'])
        if not needles:
            return False, 'INVALID', 'empty_keywords', ''
        found = [n for n in needles if n in hay]
        ok = ok and bool(found)
        notes.append(f'keywords_any={needles!r} found={found!r}')
    if 'contains_all' in cfg or 'contains' in cfg:
        has_predicate = True
        raw = cfg.get('contains_all', cfg.get('contains'))
        needles = _needles(raw)
        if not needles:
            return False, 'INVALID', 'empty_contains', ''
        missing = [n for n in needles if n not in hay]
        ok = ok and not missing
        notes.append(f'missing={missing!r}' if missing else f'all_found={needles!r}')
    if 'text' in cfg:
        has_predicate = True
        needles = _needles(cfg['text'])
        if not needles:
            return False, 'INVALID', 'empty_text', ''
        missing = [n for n in needles if n not in hay]
        ok = ok and not missing
        notes.append(f'text_missing={missing!r}' if missing else f'text_found={needles!r}')
    regex_key = 'regex' if 'regex' in cfg else 'pattern' if 'pattern' in cfg else None
    if regex_key is not None:
        has_predicate = True
        pattern = str(cfg[regex_key])
        try:
            matched = re.search(pattern, text) is not None
        except re.error as exc:
            return False, 'INVALID', f'bad_regex: {exc}', ''
        ok = ok and matched
        notes.append(f'{regex_key}={"match" if matched else "miss"}')

    if not has_predicate:
        return False, 'INVALID', 'missing_contains', ''
    detail = ', '.join(notes) if notes else 'noop'
    return ok, 'Y' if ok else 'N', detail, ''


async def _eval_text_not_contains(
    cfg: dict, state: dict, runtime: Any, junit_ctx: JunitLookupContext | None, *, limits: RubricLimits | None = None
) -> VerifierResult:
    """``text_contains`` 的否定形式，用于生成的 rubric 。

    使用与 ``text_contains`` 相同的 source 与谓词别名。 只有决定性的正 / 负
    匹配会被反转； INVALID / EMPTY_SOURCE 保持失败，这样缺失的
    answer/state 不会意外通过一个鲁棒性检查。
    """
    passed, verdict, detail, raw_head = await _eval_text_contains(
        cfg,
        state,
        runtime,
        junit_ctx,
        limits=limits,
    )
    if verdict not in ('Y', 'N'):
        return False, verdict, detail, raw_head
    ok = not passed
    return ok, 'Y' if ok else 'N', f'not_contains: {detail}', raw_head
