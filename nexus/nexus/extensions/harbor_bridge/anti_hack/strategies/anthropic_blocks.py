"""Bridge: normalize Anthropic-block trajectories into the tool_call-shaped
OpenAI-chat messages that the anti-hack detectors expect.

Why this exists
---------------
The detectors in this package read shell commands from ``message['tool_calls']``
and their observations from ``role='tool'`` messages -- the OpenAI-chat shape that
in-process Agents (code_mt, terminus2 after ``text_parse``) produce natively.

HyCode does not produce that shape. It is an external CLI, so it exposes no
``main_trajectory``; ``extract_messages`` falls back to parsing its stdout, which
is Claude-Code's Anthropic event stream. Those messages carry actions as
``content`` BLOCKS instead:

    {'role': 'assistant', 'content': [{'type': 'tool_use',   'id': .., 'name': 'Bash',
                                       'input': {'command': ..}}]}
    {'role': 'user',      'content': [{'type': 'tool_result', 'tool_use_id': .., 'content': ..}]}

``message.get('tool_calls')`` is absent on every one of them, so the scan loop
iterated zero times and the strategy reported ``detected=False`` with
``status='completed'`` -- a clean verdict it had no evidence for. Measured on the
0916 opus48 batch: 21 HyCode runs, 2920 messages, 657 real Bash calls, and the
detectors saw 0 commands. That silence is what this module removes.

This mirrors ``text_parse.py`` exactly: reuse the detectors verbatim, own only the
format bridge. No detector is modified, and no new detection channel is
introduced -- HyCode simply becomes visible to the ones already running.
"""

from __future__ import annotations

import json

from typing import Any


def _tool_result_text(content: Any) -> str:
    """Flatten one ``tool_result`` payload into the observation text.

    HyCode writes a plain string today (verified: 1050/1050 blocks on the 0916
    batch), but the Anthropic schema also permits a list of content blocks, and a
    dict slips through from some providers. All three are accepted because an
    observation lost here does not fail loudly -- it silently weakens the
    downstream ``fetch_effective`` verdict, which is the class of bug this whole
    module exists to remove.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get('text')
                parts.append(text if isinstance(text, str) else json.dumps(block, ensure_ascii=False))
            elif isinstance(block, str):
                parts.append(block)
        return '\n'.join(parts)
    if content is None:
        return ''
    return json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else str(content)


def _has_anthropic_blocks(messages: list[dict]) -> bool:
    """Whether any message carries ``tool_use`` / ``tool_result`` content blocks."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get('type') in ('tool_use', 'tool_result'):
                return True
    return False


def has_tool_activity(messages: list[dict] | None) -> bool:
    """Whether a trajectory shows the Agent invoking tools in ANY known shape.

    Used to tell "this Agent took no tool actions" apart from "this Agent's shape
    was not understood". Downstream the two are indistinguishable -- both scan
    zero commands -- but only the second is a defect, and conflating them is what
    let the HyCode blindness read as a clean result for a whole training run.

    A text-only trajectory (an Agent that simply answered, which the RepoQA and
    DreamingTask families legitimately do) has no tool activity and must NOT be
    reported as a parse failure.
    """
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if message.get('tool_calls') or message.get('tool_call_id'):
            return True
        if message.get('role') == 'tool':
            return True
    return _has_anthropic_blocks(messages or [])


def normalize_anthropic_block_messages(messages: list[dict] | None) -> list[dict] | None:
    """Anthropic-block OpenAI-chat messages -> a ``tool_calls``-shaped equivalent.

    Each ``tool_use`` block becomes one entry in the assistant's ``tool_calls``,
    keeping the block's own ``id`` as ``tool_call_id`` so the observation pairs by
    id rather than by position. Each ``tool_result`` block becomes a
    ``role='tool'`` message. Assistant ``text`` blocks are concatenated into
    ``content`` and ``thinking`` blocks into ``reasoning_content`` so both
    visible-text and reasoning-quality monitors keep working.

    Returns ``None`` (caller should keep the raw messages) when:
      * any assistant already carries structured ``tool_calls`` -- an in-process
        trajectory, already fully covered; OR
      * no ``tool_use`` / ``tool_result`` block is present -- nothing to bridge.

    Returning ``None`` rather than a rebuilt copy matters: it keeps this bridge
    strictly additive, so a shape it does not recognize is passed through
    untouched instead of being degraded by a partial conversion.
    """
    source = messages or []

    for message in source:
        if isinstance(message, dict) and message.get('role') == 'assistant' and message.get('tool_calls'):
            return None

    if not _has_anthropic_blocks(source):
        return None

    converted: list[dict[str, Any]] = []
    for message in source:
        if not isinstance(message, dict):
            continue
        content = message.get('content')
        role = message.get('role')

        if not isinstance(content, list):
            converted.append({'role': role or 'user', 'content': content if isinstance(content, str) else ''})
            continue

        texts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get('type')
            if block_type == 'text':
                text = block.get('text')
                if isinstance(text, str) and text:
                    texts.append(text)
            elif block_type == 'thinking':
                thinking = block.get('thinking')
                if isinstance(thinking, str) and thinking:
                    thinking_parts.append(thinking)
            elif block_type == 'tool_use':
                arguments = block.get('input')
                tool_calls.append({
                    'id': block.get('id'),
                    'type': 'function',
                    'function': {
                        'name': block.get('name'),
                        'arguments': json.dumps(arguments if isinstance(arguments, dict) else {}, ensure_ascii=False),
                    },
                })
            elif block_type == 'tool_result':
                observation = {
                    'role': 'tool',
                    'tool_call_id': block.get('tool_use_id'),
                    'content': _tool_result_text(block.get('content')),
                }
                # Preserve harness-authored outcome metadata for downstream
                # quality detectors instead of reducing a definite error to an
                # unclassified text observation.
                for key in ('is_error', 'status', 'tool_result_status', 'inferred_outcome', 'exit_code'):
                    if key in block:
                        observation[key] = block[key]
                observations.append(observation)

        provider_fields = message.get('provider_specific_fields')
        existing_reasoning = (
            message.get('reasoning_content')
            or message.get('_reasoning_content')
            or (
                provider_fields.get('reasoning_content') or provider_fields.get('reasoning')
                if isinstance(provider_fields, dict)
                else None
            )
        )
        reasoning_text = (
            existing_reasoning
            if isinstance(existing_reasoning, str) and existing_reasoning
            else '\n'.join(thinking_parts)
        )

        # A tool_result arrives on a role='user' message; emitting the (empty) user
        # turn as well would insert a non-tool message between an assistant's
        # tool_calls and their observations, which is exactly what makes
        # tool_observation_maps drop its pending positional pairings.
        if tool_calls or texts or reasoning_text or not observations:
            assistant_like = {'role': role or 'assistant', 'content': '\n'.join(texts)}
            if reasoning_text:
                assistant_like['reasoning_content'] = reasoning_text
            if tool_calls:
                assistant_like['tool_calls'] = tool_calls
            converted.append(assistant_like)
        converted.extend(observations)

    return converted


__all__ = ['has_tool_activity', 'normalize_anthropic_block_messages']
