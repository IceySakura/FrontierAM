"""Post-iteration hook for the ``yb_vlm_oneagent`` task.

The hook fires after every tool round (i.e. before the next LLM call)
and performs two things, surfaced as a single
``PostIterationHookResult``:

1. **Apply the doc-level token budget** to ``yb_web_search`` outputs
   in this round. Whole docs are dropped (never truncated mid-content)
   so ``[web_doc N begin]`` / ``[web_doc N end]`` brackets always
   remain paired. Implementation reuses ``nexus.extensions.projects.yb_vlm_oneagent.context_manage``.
2. **Force-answer at the penultimate iteration.** When the
   just-completed iteration index equals ``max_iterations - 1`` (so
   the next iteration is the last one), surface a sampling override
   on the result so the agent loop passes ``bad_words`` (covering
   ``<tool_calls>``, ``<tool_call>``, ``<tool_sep>``, ``</tool_calls>``
   and ``</tool_call>``) via ``sampling_params`` for the next
   ``create_completion`` call. This prevents the model from emitting
   another tool call on the final turn and forces it to commit to a
   final answer.

Per-tool-message framing (Chinese ``当前时间：`` line, the per-tool
"未能检索到相关信息" line, and the "通过其它search query获取的检索信
息如下：" separator) is emitted by the search tools themselves
(``yb_vision_search`` / ``yb_web_search``), NOT here. The hook only
touches whole-message content when the budget step decides to drop a
``yb_web_search`` doc.

Per-round metadata (iteration index, ``max_iterations`` and the
assistant message for this round) is read from
``PostIterationHookContext``.
"""

from __future__ import annotations

import logging

from typing import Any

from nexus.extensions.agents.main.agent import PostIterationHookContext, PostIterationHookResult
from nexus.extensions.projects.yb_vlm_oneagent.context_manage import (
    apply_tool_round_budget,
    split_web_doc_segments,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Tokens that must NOT appear on the final (force-answer) turn. Surfaced
# to the agent loop via ``next_completion_kwargs['sampling_params']['bad_words']``
# so vLLM blocks every opening / closing variant of the tool-call framing
# during sampling.
FORCE_ANSWER_BAD_WORDS: tuple[str, ...] = (
    '<tool_calls>',
    '<tool_call>',
    '<tool_sep>',
    '</tool_calls>',
    '</tool_call>',
)

# Chinese notice appended to the last ``role=tool`` message of the
# penultimate iteration, so the model sees an explicit user-visible
# instruction in addition to the sampling-level ``bad_words`` block.
FORCE_ANSWER_NOTICE: str = ''

# ``ctx.tool_state`` keys this hook reads.
ENABLED_KEY = 'yb_vlm_oneagent.force_answer_enabled'  # written by task

# Optional override for the per-round token cap; falls through to the
# library default when unset.
# (Image-token accounting is done entirely inside ``yb_vision_search``,
# which seeds ``yb_search.round_image_tokens``; the hook only consumes
# that and never needs the pixel budgets directly.)
_TOOL_ROUND_MAX_TOKENS_KEY = 'yb_search.tool_round_max_tokens'
# Optional override for the protocol-level special-token reservation
# (chat-template wrapping around the merged tool message).
_TOOL_ROUND_SPECIAL_RESERVE_KEY = 'yb_search.tool_round_special_reserve_tokens'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_iteration(hook_context: PostIterationHookContext) -> tuple[int, int]:
    """Return ``(current_iteration, max_iterations)`` from the hook context."""
    return int(hook_context.iteration), int(hook_context.max_iterations)


def _extract_text_from_content(content: Any) -> str:
    """Flatten a content field (str | list[parts]) into one text string."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text' and isinstance(part.get('text'), str):
                    chunks.append(part['text'])
            elif isinstance(part, str):
                chunks.append(part)
        return '\n'.join(chunks)
    try:
        import json as _json

        return _json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _is_web_search_message(text: str) -> bool:
    """Heuristic: a tool message is from ``yb_web_search`` iff it contains a ``[web_doc`` block."""
    return '[web_doc ' in text


def _extract_last_assistant_text(hook_context: PostIterationHookContext) -> str:
    for msg in reversed(hook_context.history):
        if not isinstance(msg, dict) or msg.get('role') != 'assistant':
            continue
        return _extract_text_from_content(msg.get('content'))
    return ''


def _append_force_answer_notice(tool_messages: list[dict[str, Any]]) -> None:
    """Append :data:`FORCE_ANSWER_NOTICE` to the last ``role=tool`` message.

    Mutates ``tool_messages`` in place. Handles both ``content: str`` and
    ``content: list[part]`` shapes; for the list form the notice is
    merged into the trailing text part when present, otherwise a new
    ``{'type': 'text', 'text': ...}`` part is appended so non-text parts
    (e.g. images) stay intact. No-op if there is no tool message.
    """
    last_idx = -1
    for idx in range(len(tool_messages) - 1, -1, -1):
        msg = tool_messages[idx]
        if isinstance(msg, dict) and msg.get('role') == 'tool':
            last_idx = idx
            break
    if last_idx < 0:
        return

    msg_obj = dict(tool_messages[last_idx])
    content = msg_obj.get('content')
    notice = FORCE_ANSWER_NOTICE

    if content is None or content == '':
        msg_obj['content'] = [{'type': 'text', 'text': notice}]
    elif isinstance(content, str):
        sep = '' if content.endswith('\n') else '\n'
        msg_obj['content'] = content + sep + notice
    elif isinstance(content, list):
        new_parts = list(content)
        # Find trailing text part (preserve order of any non-text parts).
        trailing_text_idx = -1
        for j in range(len(new_parts) - 1, -1, -1):
            part = new_parts[j]
            if isinstance(part, dict) and part.get('type') == 'text':
                trailing_text_idx = j
                break
        if trailing_text_idx >= 0:
            old_part = dict(new_parts[trailing_text_idx])
            old_text = old_part.get('text') or ''
            sep = '' if old_text.endswith('\n') else '\n'
            old_part['text'] = old_text + sep + notice
            new_parts[trailing_text_idx] = old_part
        else:
            new_parts.append({'type': 'text', 'text': notice})
        msg_obj['content'] = new_parts
    else:
        # Unknown content shape -- fall back to a fresh text part list.
        msg_obj['content'] = [{'type': 'text', 'text': notice}]

    tool_messages[last_idx] = msg_obj


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------


async def budget_and_finish_hook(
    ctx,
    hook_context: PostIterationHookContext,
) -> PostIterationHookResult | None:
    """Per-round budget + force-answer pipeline.

    See module docstring for the full contract.
    """
    tool_messages = list(hook_context.tool_messages)
    if not tool_messages:
        return None

    state = ctx.tool_state

    # Image accounting (consume + reset accumulator regardless of branch).
    # ``yb_vision_search`` accumulates the per-image token cost into
    # ``yb_search.round_image_tokens`` so we can include it in the
    # round-level budget. We pop here so the next round starts clean.
    round_image_tokens = int(state.pop('yb_search.round_image_tokens', 0) or 0)

    # ----- (1) Apply doc-level token budget on this round's web outputs -----
    #
    # Walk the tool messages once to collect the web-search ones. We
    # truncate at doc granularity inside each web tool message (never
    # mid-content) so the ``[web_doc N begin/end]`` brackets stay
    # paired. Non-web messages (vision_search, anything else) are
    # passed through untouched -- their text is still counted toward
    # ``fixed_extra_text`` so the budget knows how much room is left.
    new_tool_messages: list[dict[str, Any]] = list(tool_messages)
    web_indices: list[int] = []
    web_raw_texts: list[str] = []
    # Header lines that appear *before* the first ``[web_doc N begin]`` block
    # in each web_search tool message (e.g. ``当前时间：...``,
    # ``使用web_search检索工具，"qx"未能检索到相关信息；``,
    # ``通过其它search query获取的检索信息如下：``).
    # ``split_web_doc_segments`` only returns the doc bodies, so we need to
    # remember the prefix here and prepend it back when we rewrite the
    # message content -- otherwise the model loses the per-call header
    # injected by ``yb_web_search``.
    web_headers: list[str] = []
    fixed_extra_chunks: list[str] = []

    for idx, msg in enumerate(tool_messages):
        if not isinstance(msg, dict) or msg.get('role') != 'tool':
            continue
        text_body = _extract_text_from_content(msg.get('content'))
        if _is_web_search_message(text_body):
            web_indices.append(idx)
            web_raw_texts.append(text_body)
            prefix_end = text_body.find('[web_doc ')
            web_headers.append(text_body[:prefix_end].rstrip('\n') if prefix_end > 0 else '')
        elif text_body:
            fixed_extra_chunks.append(text_body)

    if web_indices:
        per_call_segments: list[list[str]] = [split_web_doc_segments(t) for t in web_raw_texts]
        fixed_extra_text = '\n'.join(s for s in fixed_extra_chunks if s)

        try:
            budget_kwargs: dict[str, Any] = {
                'tool_call_text': _extract_last_assistant_text(hook_context),
                'web_doc_segments': per_call_segments,
                'fixed_extra_text': fixed_extra_text,
                'image_tokens': round_image_tokens,
                # TokenCounter 由 task 经 tool_state 注入，必须显式传入。
                'token_counter': state['token_counter'],
            }
            if state.get(_TOOL_ROUND_MAX_TOKENS_KEY):
                budget_kwargs['tool_round_max_tokens'] = int(state[_TOOL_ROUND_MAX_TOKENS_KEY])
            if state.get(_TOOL_ROUND_SPECIAL_RESERVE_KEY) is not None:
                budget_kwargs['special_reserve_tokens'] = int(state[_TOOL_ROUND_SPECIAL_RESERVE_KEY])
            budget = apply_tool_round_budget(**budget_kwargs)
        except Exception:
            logger.warning(
                'apply_tool_round_budget failed; leaving web_search outputs untrimmed',
                exc_info=True,
            )
            budget = None

        truncated_segments = (budget or {}).get('truncated_web_segments')
        if truncated_segments is not None:
            for nm_idx, header, kept_docs in zip(
                web_indices,
                web_headers,
                truncated_segments,
                strict=False,
            ):
                docs_body = '\n'.join(kept_docs)
                # Prepend the original per-call header (e.g. ``当前时间：...``)
                # so it survives the doc-level truncation rewrite.
                rebuilt_body = f'{header}\n{docs_body}' if header and docs_body else header or docs_body
                # Replace in place (preserve role / tool_call_id / etc.).
                msg_obj = dict(new_tool_messages[nm_idx])
                msg_obj['content'] = [{'type': 'text', 'text': rebuilt_body}]
                new_tool_messages[nm_idx] = msg_obj

    # ----- (2) Force-answer at the penultimate iteration -----
    next_completion_kwargs: dict[str, Any] | None = None
    if state.get(ENABLED_KEY, False):
        current, max_iter = _resolve_iteration(hook_context)
        if max_iter > 0 and current == max_iter - 1:
            logger.warning(
                '[yb_vlm_oneagent] iter %d/%d — disabling tool calls on final turn via bad_words=%r',
                current,
                max_iter,
                list(FORCE_ANSWER_BAD_WORDS),
            )
            next_completion_kwargs = {
                'sampling_params': {
                    'bad_words': list(FORCE_ANSWER_BAD_WORDS),
                },
            }
            _append_force_answer_notice(new_tool_messages)

    return PostIterationHookResult(
        tool_messages=new_tool_messages,
        next_completion_kwargs=next_completion_kwargs,
    )
