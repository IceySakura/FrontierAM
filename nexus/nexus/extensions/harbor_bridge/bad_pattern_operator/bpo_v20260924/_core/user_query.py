"""WorkBuddy / CodeBuddy 脚手架：从 user 轮里抽出真正的提问。

``<user_query>…</user_query>`` 是这两家 harness 的标签，不是 Claude Code
等通用约定。它们会把时间、打开的文件、工作区 identity、摘要续写套话
塞进同一条 ``role=user``，人真正说的那句在标签里面。

有 ``<user_query>`` 就只用里面的正文（可多段；丢掉图片占位和摘要续写
套话）。没有该标签时，剥掉 user_info / system-reminder / additional_data
外壳，剩下的当提问。
"""
from __future__ import annotations

import re

_USER_QUERY_RE = re.compile(
    r"<user_query(?:\s[^>]*)?>(?P<query>[\s\S]*?)</user_query\s*>",
    re.IGNORECASE,
)

_SHELL_TAGS = (
    "additional_data",
    "system-reminder",
    "system_reminder",
    "user_info",
    "identity_context",
)
_SHELL_BLOCK_RE = re.compile(
    r"<(?P<tag>" + "|".join(re.escape(tag) for tag in _SHELL_TAGS) + r")"
    r"(?:\s[^>]*)?>.*?</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)

_CONTINUE_AFTER_SUMMARY_RE = re.compile(
    r"Please continue with the conversation based on the summarized context",
    re.IGNORECASE,
)
_IMAGE_ONLY_RE = re.compile(
    r"^(?:"
    r"@image#\S+"
    r"|\[Image from[\s\S]*\]"
    r"|<image_local_path>[\s\S]*</image_local_path>"
    r")\s*$",
    re.IGNORECASE,
)


def _is_scaffold_filler_query(query: str) -> bool:
    text = query.strip()
    if not text:
        return True
    if _CONTINUE_AFTER_SUMMARY_RE.search(text):
        return True
    if _IMAGE_ONLY_RE.match(text):
        return True
    return False


def extract_user_query(content: str) -> str:
    """Return the human ask inside a user-turn ``content`` string.

    ``content`` 保持原样给别的算子；这里只读，不改 Message。
    """
    if not content:
        return ""
    matches = list(_USER_QUERY_RE.finditer(content))
    if matches:
        queries = [
            match.group("query").strip()
            for match in matches
            if not _is_scaffold_filler_query(match.group("query"))
        ]
        return "\n".join(queries)
    return _SHELL_BLOCK_RE.sub("", content).strip()
