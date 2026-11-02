"""Bridge: normalize code_mt TEXT_PARSE trajectories into the tool_call-shaped
OpenAI-chat messages that the structured-only monitors expect.

Why this exists
---------------
code_mt has two agent loops (nexus.extensions.projects.code_mt.agent.agent):
  * tool_call_loop  — assistant messages carry native ``tool_calls`` + the tool
                      observation arrives as a ``role='tool'`` message.
  * text_parse_loop — assistant messages are RAW NL text with shell commands
                      embedded in ``<keystrokes>`` (XML parser) or a JSON
                      ``{"commands":[{"keystrokes": ...}]}`` block, and the
                      terminal observation is the IMMEDIATELY FOLLOWING
                      ``role='user'`` message (chat.chat appends the next prompt,
                      which is that turn's terminal_output).

The SWE anti-hack detector (``parse_nexus_messages``) and the baseline
``trailing_tool_failure`` monitor read ONLY structured ``tool_calls`` /
``role='tool'`` messages, so on a text_parse trajectory they see zero actions and
are silently INERT — online-answer-fetch on a terminal task would never be
flagged. This module reconstructs the tool_call view WITHOUT modifying the
read-only SWE package (same design contract as ``str_replace_channels.py``: reuse
the reference detectors verbatim, own only the code_mt-specific format bridge).

Flag-only: the reconstructed view is fed to detectors that never change the score
themselves; masking is still decided downstream by
``trainer.additional_terminate_reason`` in the run script.
"""

from __future__ import annotations

import json
import re

from typing import Any


# <keystrokes duration="0.1">shell command text</keystrokes>  (XML parser channel)
_KEYSTROKES_RE = re.compile(r'<keystrokes[^>]*>(.*?)</keystrokes>', re.DOTALL)
# "keystrokes": "shell command text"  (JSON parser channel; captures escaped body)
_JSON_KEYSTROKES_RE = re.compile(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _g(o: Any, k: str, d: Any = None) -> Any:
    return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)


def _commands_from_content(content: Any) -> list[str]:
    """Recover the shell command strings issued in one assistant text message.

    Tries the XML ``<keystrokes>`` channel first, then the JSON ``keystrokes``
    channel. Returns the raw keystroke strings (which for terminal tasks are the
    shell command text the fetch signals scan). Empty list when the message
    issued no command (pure analysis / task_complete-only turn).
    """
    if not isinstance(content, str) or not content:
        return []

    xml = [k for k in _KEYSTROKES_RE.findall(content) if k.strip()]
    if xml:
        return xml

    cmds: list[str] = []
    for raw in _JSON_KEYSTROKES_RE.findall(content):
        try:
            cmds.append(json.loads('"' + raw + '"'))  # JSON-unescape \n \" etc.
        except (json.JSONDecodeError, ValueError):
            cmds.append(raw)
    return [c for c in cmds if isinstance(c, str) and c.strip()]


def normalize_text_parse_messages(messages: list[dict] | None) -> list[dict] | None:
    """text_parse OpenAI-chat messages -> a tool_call-shaped equivalent.

    For each assistant message, its ``<keystrokes>``/JSON commands become
    ``tool_calls`` (a synthetic ``bash`` call per command) and the FOLLOWING
    ``role='user'`` terminal output becomes a ``role='tool'`` observation linked
    by ``tool_call_id`` — exactly the shape ``parse_nexus_messages`` /
    ``detect_trailing_tool_failure`` consume. The assistant ``content`` is
    preserved so the text-reading monitors still work if the caller reuses this.

    Returns ``None`` (caller should fall back to the raw messages) when:
      * any assistant already carries structured ``tool_calls`` — this is a
        tool_call-loop trajectory, already fully covered; OR
      * no assistant message yields any extractable command — nothing to bridge.
    """
    msgs = messages or []

    # Already structured (tool_call loop): the monitors work as-is.
    for m in msgs:
        if _g(m, 'role') == 'assistant' and _g(m, 'tool_calls'):
            return None

    out: list[dict] = []
    made_any = False
    cid = 0
    n = len(msgs)
    for i, m in enumerate(msgs):
        role = _g(m, 'role')
        if role != 'assistant':
            content = _g(m, 'content', '')
            out.append({'role': role or 'user', 'content': content})
            continue

        content = _g(m, 'content', '')
        cmds = _commands_from_content(content)

        # The turn's terminal observation is the next role='user' message
        # (chat.chat appends it as the following prompt). Absent on the final
        # turn if the run ended on task_complete — then obs stays ''.
        obs = ''
        if i + 1 < n and _g(msgs[i + 1], 'role') == 'user':
            nxt = _g(msgs[i + 1], 'content', '')
            obs = nxt if isinstance(nxt, str) else ''

        asst: dict[str, Any] = {'role': 'assistant', 'content': content if isinstance(content, str) else ''}
        tool_msgs: list[dict] = []
        if cmds:
            tool_calls = []
            for c in cmds:
                cid += 1
                tid = f'tp_{cid}'
                tool_calls.append({
                    'id': tid,
                    'type': 'function',
                    'function': {'name': 'bash', 'arguments': json.dumps({'command': c})},
                })
                # Attach the (shared) turn observation to each command so the
                # per-fetch effective verdict has its downstream output.
                tool_msgs.append({'role': 'tool', 'tool_call_id': tid, 'content': obs})
            asst['tool_calls'] = tool_calls
            made_any = True

        out.append(asst)
        out.extend(tool_msgs)

    return out if made_any else None


__all__ = ['normalize_text_parse_messages']
