"""F5_json_format_error/_shared/_harness_utils — harness 喊话检测的共用工具。

仅 F5-3 (``F5_3_harness_complaint_loop``) 用，但放 _shared 是为了：
  1. F5-3 的 ``JSON_ERROR_PATTERN`` 是一条通用 regex，将来扩展到其它
     "外部喊话" 类 detector 也可能复用；
  2. ``MAX_MSG_BYTES`` / ``SHELL_PROMPT_RE`` 是 detector 无关的"消息层"
     兜底常量，集中放更好维护。

F5-3 的主规则（DSL）：

    HIT_ONE(msg) :=
        role IN {user, tool, system}
        AND len(text) <= MAX_MSG_BYTES                 # = 30000
        AND NOT IS_TERMINAL_DUMP(text)                  # shell prompt / fenced code dump
        AND REGEX(text, JSON_ERROR_PATTERN)             # json + error keyword bidirectional

    TRIGGER F5-3 :=
        EXISTS run = [i_1, ..., i_k] WITH
            k >= F5_3_MIN_CONSECUTIVE                   # = 2
            AND ALL i_j: HIT_ONE(messages[i_j])
            AND CONSECUTIVE(run)
            # 允许中间有 assistant turn（重试，正常）；
            # 不允许中间有 non-error 的 user/tool/system msg。
"""
from __future__ import annotations

import re as _re
from typing import Any, Dict, List, Optional


# ============================================================
# 长度 / role 兜底
# ============================================================

# 长 file dump 多半是 log / terminal capture，不算 harness 喊话。
MAX_MSG_BYTES = 30_000

# 把 user/tool/system 看成"非 assistant 喊话端"；F5-3 / F5-3-loop 都只扫这三类。
NON_ASSISTANT_ROLES = ("user", "tool", "system")


# ============================================================
# Terminal dump 兜底
# ============================================================

# Terminal stdout / shell prompt 开头：``root@host:#`` / ``$ `` / ``> `` /
# ``PS `` 等都视为 terminal dump，**即使内含 JSONDecodeError 也不信**——
# 多半是 agent 自己跑了什么 python 脚本 traceback。
SHELL_PROMPT_RE = _re.compile(
    r"(?m)^(?:root@[\w\-]+|\$|>|#|PS )\s|^[\w\-\.]+@[\w\-\.]+:[^\$#]*[#\$]\s",
)

TRIPLE_BACKTICK = "```"


def _is_terminal_dump(text: str) -> bool:
    """消息看起来是 terminal output / file dump 噪声？"""
    if SHELL_PROMPT_RE.search(text):
        return True
    if text.count(TRIPLE_BACKTICK) >= 4 and len(text) > 5000:
        return True
    return False


# ============================================================
# 通用 JSON-error pattern（F5-3 主 regex）
# ============================================================

# Strong harness receipts that remain authoritative even when a platform wraps
# the feedback in a successful transport envelope.  Keep these phrases narrow:
# arbitrary tool stdout frequently contains JSON parser source code.
HARNESS_COMPLAINT_PATTERN = _re.compile(
    r"(?:\bprevious response had parsing errors?\b"
    r"|\binvalid_tool_arguments\b"
    r"|\btb2[^\n]{0,80}\bparsing\b"
    r"|\bopenclaw\s+mcporter\b)",
    _re.IGNORECASE,
)

# Main complaint regex.  This deliberately requires a parse/decode failure,
# instead of the former broad `json ... any error word` window which matched
# ordinary source listings such as JSON error-handler implementations.
JSON_ERROR_PATTERN = _re.compile(
    r"(?:\bprevious response had parsing errors?\b"
    r"|\binvalid_tool_arguments\b"
    r"|\bjson\b.{0,80}\b(?:parse|parsing|decode|decoding|syntax)\w*\b"
    r".{0,80}\b(?:error|fail|failed|failure|invalid|reject|rejected)\w*\b"
    r"|\b(?:cannot|could\s+not|failed\s+to|unable\s+to)\s+"
    r"(?:parse|decode)\b.{0,80}\bjson\b"
    r"|\bjsondecodeerror\b"
    r"|\btb2[^\n]{0,80}\bparsing\b"
    r"|\bopenclaw\s+mcporter\b"
    r"|json\s*(?:解析失败|不合法|格式错误|无法解析|错误))",
    _re.IGNORECASE | _re.DOTALL,
)


# ============================================================
# F5-3 主参数
# ============================================================

# 连续 ≥2 次 user/tool/system msg 命中 → 触发。
# 设计 rationale：单次 harness 喊话 FP 多（tool stdout 含 JSON 错关键词、
# 用户引用文档等）；卡死 loop 才是真正值得 alert 的 P0/P1 故障。
F5_3_MIN_CONSECUTIVE = 2


# ============================================================
# 上溯被指控 assistant
# ============================================================

def _find_prev_assistant_turn(
    messages: List[Dict[str, Any]],
    user_idx: int,
) -> Optional[int]:
    """user/tool msg 的报错通常是在抱怨上一条 assistant 消息——回溯找最近的
    assistant turn 作为被指控方。"""
    for i in range(user_idx - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    return None


__all__ = [
    "MAX_MSG_BYTES",
    "NON_ASSISTANT_ROLES",
    "SHELL_PROMPT_RE",
    "TRIPLE_BACKTICK",
    "_is_terminal_dump",
    "HARNESS_COMPLAINT_PATTERN",
    "JSON_ERROR_PATTERN",
    "F5_3_MIN_CONSECUTIVE",
    "_find_prev_assistant_turn",
]
