"""Resolve General rollout language to zh/en using the shared CJK heuristic."""

from __future__ import annotations

from nexus.extensions.agents.context_agent.lang import detect_lang


Lang = str  # "en" | "zh"

_ZH_ALIASES = frozenset({'cn', 'zh', 'zh-cn', 'zh_cn', 'chinese', '中文'})
_EN_ALIASES = frozenset({'en', 'eng', 'english'})


def resolve_lang(explicit: str | None, question: str | None) -> Lang:
    """Resolve the rollout language to ``"zh"`` or ``"en"``.

    Prefer an explicit, pre-tagged value; otherwise infer it from the question
    text. The shared ``detect_lang`` returns ``"cn"`` for Chinese, so
    its verdict is translated here rather than at every call site.
    """
    if isinstance(explicit, str):
        norm = explicit.strip().lower()
        if norm in _ZH_ALIASES:
            return 'zh'
        if norm in _EN_ALIASES:
            return 'en'
    return 'zh' if detect_lang(question) == 'cn' else 'en'
