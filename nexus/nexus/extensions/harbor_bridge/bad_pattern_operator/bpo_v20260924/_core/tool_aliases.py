"""Tool-name canonicalization + tool-call args parsing.

Ported from ``bpo.base_detector`` per §11.B of the design doc — the
spike only ports what F11 + H1 actually need (full table for
discoverability, only ``parse_args_dict``/``get_command``/``get_path``/
``get_url``/``ci_get`` used by F11; H1 uses ``parse_args_dict`` indirectly
through ``ToolCall.args``).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Tuple


# ─── Tool aliases ────────────────────────────────────────────────────
# Kept identical to bpo.base_detector.TOOL_ALIASES so detectors keep
# their behavior across the rebrand.

TOOL_ALIASES: Dict[str, str] = {
    # shell family
    "bash": "shell", "exec": "shell", "terminal": "shell", "powershell": "shell",
    "execute_command": "shell", "execute_shell_command": "shell",
    "execute_code": "shell", "run_bash_command": "shell", "process": "shell",
    "cronjob": "shell", "run_terminal_cmd": "shell", "shell_exec": "shell",
    # read family
    "read": "read", "read_file": "read", "read_files": "read", "view_file": "read",
    # edit / write family
    "edit": "edit", "write": "edit", "write_file": "edit",
    "write_to_file": "edit", "replace_in_file": "edit", "patch": "edit",
    "create_or_rewrite_file": "edit", "str_replace": "edit",
    "str_replace_editor": "edit", "apply_patch": "edit", "insert": "edit",
    # search family
    "grep": "search", "search_content": "search",
    "search_files": "search", "search_file": "search",
    # list family
    "list_dir": "list_dir", "ls": "list_dir", "tree": "list_dir",
    # web
    "web_search": "web_search", "websearch": "web_search",
    "web_fetch": "web_fetch", "webfetch": "web_fetch", "fetch": "web_fetch",
}


# Multi-command editor tools: ``str_replace_editor`` / ``text_editor`` (the
# Anthropic editor tool) dispatch on a ``command`` arg — ``view`` is a file
# READ, ``str_replace`` / ``create`` / ``insert`` are edits, ``undo_edit`` is
# neither. Mapping the whole tool to ``"edit"`` (the static table) miscounts
# views as edits — the dominant false-positive source for the edit-family
# detectors (CODE_402/403/404) on harnesses that use this tool (on
# SWE-bench-Verified ~62% of its calls are ``view``). ``canonicalize_tool``
# resolves the real action when given the call's args.
_MULTI_COMMAND_EDITORS = {
    "str_replace_editor", "str_replace_based_edit_tool", "text_editor", "editor",
}
_EDITOR_COMMAND_CANON = {
    "view": "read",
    "str_replace": "edit", "create": "edit", "insert": "edit", "write": "edit",
    "undo_edit": "undo_edit",
}


def canonicalize_tool(name: Optional[str], args: Optional[Dict[str, Any]] = None) -> str:
    """Map a raw tool name (``function.name``) to the canonical alias.

    Returns ``"unknown"`` for empty / None input.

    For **multi-command editor** tools (``str_replace_editor`` & friends) the
    canonical family depends on the ``command`` arg, so pass ``args`` to get the
    real action (``view`` → ``read``, edit-commands → ``edit``). Without
    ``args`` (legacy callers) it falls back to the static name mapping, which
    treats the whole tool as ``"edit"`` — preserving prior behaviour.
    """
    if not name:
        return "unknown"
    n = str(name).strip().lower()
    if n in _MULTI_COMMAND_EDITORS and isinstance(args, dict):
        cmd = (get_command(args) or "").strip().lower()
        mapped = _EDITOR_COMMAND_CANON.get(cmd)
        if mapped:
            return mapped
        # unknown / missing command → treat as edit (conservative: the old
        # behaviour, so a real edit is never reclassified away).
        return "edit"
    return TOOL_ALIASES.get(n, n)


# ─── Args parsing helpers ────────────────────────────────────────────

COMMAND_KEYS: Tuple[str, ...] = (
    "command", "cmd", "code", "script", "shell",
    "input", "keystrokes", "shell_command", "bash_command",
)
PATH_KEYS: Tuple[str, ...] = (
    "path", "file_path", "filepath", "file", "filename", "file_name",
    "target", "target_file", "abs_path", "absolute_path",
)
URL_KEYS: Tuple[str, ...] = (
    "url", "uri", "endpoint", "address", "host", "request_url",
)


def parse_args_dict(args_raw: Any) -> Dict[str, Any]:
    """Coerce ``tool_calls[*].function.arguments`` into a dict.

    On JSON parse failure, falls back to ``{"_raw": <string>}`` so
    detectors can still scan the raw payload.
    """
    if isinstance(args_raw, dict):
        return args_raw
    if not isinstance(args_raw, str):
        return {"_raw": "" if args_raw is None else str(args_raw)}
    s = args_raw.strip()
    if not s:
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {"_raw": s}
    except (json.JSONDecodeError, ValueError):
        return {"_raw": s}


def ci_get(args: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    """Case-insensitive lookup of the first non-empty value in *args*."""
    if not isinstance(args, dict):
        return None
    lower_map = {str(k).lower(): k for k in args.keys()}
    for k in keys:
        if k in lower_map:
            v = args[lower_map[k]]
            if v is not None and v != "":
                return v
    return None


def get_command(args: Dict[str, Any]) -> str:
    """Pull the shell-command string out of a tool-call args dict."""
    v = ci_get(args, COMMAND_KEYS)
    if isinstance(v, str) and v.strip():
        return v
    raw = args.get("_raw") if isinstance(args, dict) else None
    return raw if isinstance(raw, str) else ""


# Shell idioms that WRITE a file — so a content-only harness (terminal_bench)
# that edits via `cat > f << EOF` / `sed -i` / `tee` / `> f` instead of an edit
# tool still surfaces as an EDIT to the edit-family detectors (CODE_403/404),
# not just an opaque shell call. Ordered most- to least-specific; the first
# match wins and yields the target path.
_FILE_WRITE_IDIOMS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = ()


def _file_write_idioms() -> Tuple[Tuple[str, Any], ...]:
    global _FILE_WRITE_IDIOMS
    if _FILE_WRITE_IDIOMS:
        return _FILE_WRITE_IDIOMS
    import re as _re
    _FILE_WRITE_IDIOMS = (
        # heredoc: `cat > path << EOF` / `cat >> path <<-'EOF'` / `tee path <<EOF`
        ("heredoc", _re.compile(r"(?:>>?|tee\s+(?:-a\s+)?)\s*([^\s<>|&;]+)\s*<<", _re.I)),
        # in-place edit: `sed -i ... path`  (path = last bare token)
        ("sed_i", _re.compile(r"\bsed\s+-i\b.*?\s([^\s<>|&;]+\.\w+)\s*$", _re.I)),
        # redirect to a file with an extension: `... > path.ext` / `>> path.ext`
        ("redirect", _re.compile(r">>?\s*([^\s<>|&;]+\.[A-Za-z0-9_]+)\b")),
        # tee to a file: `... | tee path`
        ("tee", _re.compile(r"\btee\s+(?:-a\s+)?([^\s<>|&;]+\.[A-Za-z0-9_]+)\b", _re.I)),
    )
    return _FILE_WRITE_IDIOMS


def shell_file_write_target(command: str) -> Optional[str]:
    """If a shell command WRITES a file (heredoc / sed -i / redirect / tee),
    return the target path; else ``None``. Lets the edit-family detectors see
    file edits made via the shell on content-only harnesses (terminal_bench).

    Only the COMMAND LINE is scanned (the text up to the first heredoc body or
    newline), so a ``>`` / token inside an embedded heredoc body — e.g. the R/
    Python source written by ``cat > f << EOF … EOF`` or run by ``python3 <<
    EOF`` — never false-matches as a redirect target."""
    if not command or not isinstance(command, str):
        return None
    # Scan only up to the heredoc delimiter (`<<`) or first newline — the body
    # below it is file CONTENT, not part of the command's redirect syntax.
    head = command
    cut = head.find("<<")
    if cut != -1:
        head = head[:cut]
    head = head.split("\n", 1)[0]
    for _name, pat in _file_write_idioms():
        # heredoc idiom needs the `<<` we just trimmed → re-test on full command
        target = command if _name == "heredoc" else head
        m = pat.search(target)
        if m:
            p = m.group(1).strip().strip("'\"")
            if p and p not in (".", "..", "/") and "\n" not in p:
                return p
    return None


def get_path(args: Dict[str, Any]) -> str:
    v = ci_get(args, PATH_KEYS)
    return v.strip() if isinstance(v, str) else ""


def get_url(args: Dict[str, Any]) -> str:
    v = ci_get(args, URL_KEYS)
    return v.strip() if isinstance(v, str) else ""


__all__ = [
    "TOOL_ALIASES",
    "COMMAND_KEYS",
    "PATH_KEYS",
    "URL_KEYS",
    "canonicalize_tool",
    "ci_get",
    "get_command",
    "get_path",
    "get_url",
    "parse_args_dict",
]
