"""F5_json_format_error/_shared/_json_utils — JSON 解析 + content-as-JSON 启发式。

源头：搬自旧 ``detect_f5_json_format_error.py``；**逻辑、正则、阈值完全
保持不变**，仅迁移位置以便 F5-1 / F5-2 共用。
"""
from __future__ import annotations

import json as _json
import re as _re
from typing import Tuple


# ============================================================
# 长度阈值（A 路 assistant 自检通用）
# ============================================================

MIN_LEN = 30
HIGH_LEN = 200


# ============================================================
# JSON 解析
# ============================================================

def _try_parse(s: str) -> Tuple[bool, str]:
    """尝试 json.loads。返回 (ok, err_msg)。err_msg 仅在 ok=False 时有意义。"""
    try:
        _json.loads(s)
        return True, ""
    except _json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e.msg} at line {e.lineno} col {e.colno}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _err_kind(err: str) -> str:
    """把 json.loads 错误信息归到几个粗粒度类别，便于聚合统计。"""
    ml = err.lower()
    if "unterminated string" in ml:
        return "unterminated_string"
    if "expecting value" in ml:
        return "expecting_value"
    if "expecting property name" in ml:
        return "expecting_property"
    if "delimiter" in ml:
        return "expecting_delim"
    if "extra data" in ml:
        return "extra_data"
    if "control char" in ml:
        return "control_char"
    if "invalid \\escape" in ml or "invalid escape" in ml:
        return "invalid_escape"
    return "other"


# ============================================================
# content-as-JSON 启发式（仅 F5-2 用）
# ============================================================

# 同时识别 "key": 与 'key':（覆盖 JSON 双引号风格与 Python-dict 单引号风格）。
# 当 content 首尾是 `{...}` / `[...]` 时再要求里面真的有 quoted-key-colon，
# 才认为模型在认真写 JSON / dict；否则当作自然语言放过，避免
# ``[Recent Summary]`` / ``{hina bowing deeply}`` 这种 RP 标签被误判 F5-2。
_JSON_KV_RE = _re.compile(r'''["']\s*[A-Za-z_][\w\-\.]*\s*["']\s*:''')

# 已知的"非 JSON 但首尾形状对齐"前缀——hermes_agent 上下文压缩 / openclaw 系统
# 标签 / 通用日志前缀等。命中其一就直接跳过 content 分支，避免高频 FP。
# （这些 prefix 后面常常嵌入了来自历史 tool_call 的 `"key":` 片段，
#  二级判定挡不住。）
_NON_JSON_CONTENT_PREFIX_RE = _re.compile(
    r"^\s*\[\s*("
    r"Recent\s+Summary"
    r"|CONTEXT\s+COMPACTION"
    r"|ASSISTANT\s*\]"
    r"|TOOL[_ ]CALL"
    r"|TOOL[_ ]RESULT"
    r"|Expand\s+for\s+details"
    r"|Response\s+truncated"
    r"|System\s+reminder"
    r"|System\s+Notification"
    r"|reply_to_current"
    r"|原子操作"
    r"|HEARTBEAT"
    r")",
    _re.IGNORECASE,
)


def _looks_like_json(text: str) -> bool:
    """粗判一段字符串是否在尝试做 JSON / Python-dict 字面量。

    三层 gate：
      1. strip() 首尾是 ``{...}`` 或 ``[...]``
      2. 不是已知的"非 JSON 上下文压缩 / 系统标签"开头
         （hermes ``[Recent Summary]``、openclaw ``[HEARTBEAT]`` 等。这些
          prefix 后面常常嵌着来自历史 tool_call 的 ``"key":`` 片段，把
          二级判定骗过去——所以单独 deny-list）
      3. 头 4 KB 里至少出现一次 quoted-key-colon（``"k":`` 或 ``'k':``）

    第 2、3 条用来挡住 ``[Recent Summary]\n[ASSISTANT]: ...`` 这种压缩头、
    ``{hina bowing deeply with worried eyes} ...`` 这种 RP narrative、
    ``[原子操作] 调用 write_to_file ...`` 这种 agent 内部 log 标签——
    它们"首尾对齐了但根本不是结构化数据"。
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    first, last = stripped[0], stripped[-1]
    if not ((first == "{" and last == "}") or (first == "[" and last == "]")):
        return False
    if _NON_JSON_CONTENT_PREFIX_RE.match(stripped):
        return False
    return bool(_JSON_KV_RE.search(stripped[:4096]))


# ============================================================
# 通用工具
# ============================================================

def severity_for_len(n: int) -> str:
    """A 路 (F5-1 / F5-2) 公用：按字串长度区分 severity。

    长 payload 错位影响更大，给 high；短的给 medium。
    """
    return "high" if n >= HIGH_LEN else "medium"


def make_snippet(text: str, max_len: int = 600) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + " …(truncated)"


__all__ = [
    "MIN_LEN",
    "HIGH_LEN",
    "_try_parse",
    "_err_kind",
    "_looks_like_json",
    "_JSON_KV_RE",
    "_NON_JSON_CONTENT_PREFIX_RE",
    "severity_for_len",
    "make_snippet",
]
