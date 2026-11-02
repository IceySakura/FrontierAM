"""``judge_claim`` 原语——针对状态切片的 LLM 判定器。"""

import asyncio
import json

from typing import Any

from jmespath.exceptions import JMESPathError

from ..config import _DEFAULT_LIMITS, JunitLookupContext, RubricLimits, VerifierResult
from ..jmespath_predicates import _jmespath_compile_cached
from ..judge import _llm_judge
from ..selectors import _gather_source, _resolve_state_source, _text_source_selector


def _is_empty_source(value: Any) -> bool:
    """Return whether a selected value contains no evidence for the judge."""
    return value is None or (isinstance(value, (str, list, dict, tuple, set)) and not value)


def _render_selected_source(source: str, query: str, value: Any) -> str:
    """Render a JMESPath result without serializing the unfiltered source."""
    if isinstance(value, list):
        body = '\n'.join(json.dumps(item, ensure_ascii=False) for item in value)
    elif isinstance(value, dict):
        body = json.dumps(value, ensure_ascii=False)
    else:
        body = str(value)
    return f'## {source} | jmespath: {query}\n{body}\n'


def _gather_jmespath_source(state: dict, source: str, query: str) -> tuple[str, str]:
    """Select raw state with JMESPath before rendering it for the LLM judge."""
    if source == 'all':
        source_value: Any = state
    else:
        found, source_value = _resolve_state_source(state, source)
        if not found:
            return '', f'empty_source ({source})'
    try:
        selected = _jmespath_compile_cached(query).search(source_value)
    except JMESPathError as exc:
        return '', f'bad_jmespath: {exc}'
    if _is_empty_source(selected):
        return '', f'empty_jmespath_result ({source}: {query})'
    return _render_selected_source(source, query, selected), ''


async def _eval_judge_claim(
    cfg: dict, state: dict, runtime: Any, junit_ctx: JunitLookupContext | None, *, limits: RubricLimits | None = None
) -> VerifierResult:
    """LLM 判定器包装，采用标注者友好的命名：``from`` （切片） +
    ``claim`` （自然语言陈述）。 接受旧的 ``source`` 作为 ``from`` 的
    别名，使 v2 → v4.2 迁移成为纯粹的重命名。
    """
    slice_name = _text_source_selector(cfg)
    claim = cfg.get('claim', '')
    if not claim:
        return False, 'INVALID', 'empty_claim', ''
    lim = limits or _DEFAULT_LIMITS
    has_query = 'jmespath' in cfg
    query = cfg.get('jmespath')
    normalized_query: str | None = None
    if has_query:
        if not isinstance(query, str) or not query.strip():
            return False, 'INVALID', 'empty_jmespath_query', ''
        normalized_query = query.strip()
        source_text, selection_error = _gather_jmespath_source(state, slice_name, normalized_query)
        if selection_error.startswith('bad_jmespath:'):
            return False, 'INVALID', selection_error, ''
        if selection_error:
            return False, 'EMPTY_SOURCE', selection_error, ''
    else:
        source_text = _gather_source(state, slice_name, cfg.get('source_ref'), limits=lim)
    if not source_text:
        return False, 'EMPTY_SOURCE', f'empty_source ({slice_name})', ''
    # ``_llm_judge`` 是同步的（urllib + time.sleep 重试）；卸载到工作线程，
    # 使多个在分发器信号量下运行的 judge_claim 检查能够真正并行执行，
    # 而不是在事件循环上串行化。
    verdict, raw_head = await asyncio.to_thread(
        _llm_judge,
        source_text,
        claim,
        max_source_chars=lim.judge_source_max_chars,
    )
    detail = f'verdict={verdict}'
    if normalized_query is not None:
        detail += f' jmespath={normalized_query!r}'
    return verdict == 'Y', verdict, detail, raw_head
