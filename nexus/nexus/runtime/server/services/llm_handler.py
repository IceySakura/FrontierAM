"""LLM Request Handler.

Handles LLM requests from the runtime side via Socket.IO.
This enables installed agents running in sandboxed runtimes to
make LLM calls through the host-side OpenAI client.
"""

import asyncio
import contextlib
import contextvars
import copy
import json
import logging
import os
import re
import time
import weakref

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from nexus.core.openai_client import (
    CONTEXT_LENGTH_EXCEEDED_CODE,
    UNPARSED_TOOL_CALL_FIELD,
    ContextLengthExceededError,
    OpenAICompatibleClient,
)
from nexus.runtime.server.rpc_replay_guard import RPCReplayGuard
from nexus.runtime.server.services.rsio_llm_bridge import (
    NEXUS_AGENT_COMPACTION_FIELD,
    NEXUS_COMPACTION_DETECTOR_FIELD,
    RSIOLLMConfig,
)
from nexus.trace import span_context


if TYPE_CHECKING:
    from nexus.runtime import Runtime


logger = logging.getLogger(__name__)

NoThinkingRequestArgsMode = Literal['replace', 'inherit']

_HEARTBEAT_FAILURE_TIMEOUT = 5 * 60.0
# Keep a just-emitted result long enough to replay to a resend. A successful emit is
# not a worker ACK: if the frame was lost, the worker resends only after
# ``RSIOLLMConfig.report_timeout`` (45s) of silence, so the grace must outlast it.
_FINAL_REPLAY_GRACE_SECONDS = float(os.environ.get('NEXUS_FINAL_REPLAY_GRACE_SECONDS') or 90.0)
_UNBOUND_LLM_CHANNEL_CODE = 'UNBOUND_LLM_CHANNEL'
_NON_ALNUM = re.compile(r'[^A-Za-z0-9]')
# A scaffold-truncated id must keep at least this many chars to be matched by prefix.
_MIN_TRUNCATED_TOOL_ID_LEN = 16
_UNSUPPORTED_COMPLETION_FIELDS = frozenset({
    # wb's quick mode sends these; the host-side OpenAI SDK rejects them as unexpected kwargs.
    'cache_retention',
    'extra_vars',
    'metadata',
    # wb sends this; older host-side OpenAI SDKs reject it as an unexpected kwarg.
    'reasoning_summary',
    'service_tier',
    'store',
    'stream_options',
})


def _normalize_tool_schema(schema: Any) -> Any:
    """Make one JSON schema safe for strict function-calling backends.

    Vertex/gemini refuses ``{'type': 'object', 'properties': {}}`` with
    "functionDeclaration parameters schema should be of type OBJECT" and drops
    the *whole* request, so one zero-argument tool anywhere in the list stops
    the agent taking a single step. A bare ``{'type': 'object'}`` is accepted.

    Playwright MCP emits the empty-map shape for every zero-argument tool
    (``browser_snapshot``, ``browser_close``, ...). The runtime-side proxies
    convert Anthropic tools to OpenAI format inside the container, i.e. from a
    *published runtime image*, so that conversion cannot be relied on to have
    this fix. Normalising here covers every proxy (anthropic, gemini, openai)
    regardless of the image version the runtime happens to be running.
    """
    if not isinstance(schema, dict):
        return schema

    result = {}
    for key, value in schema.items():
        if key == 'properties' and isinstance(value, dict):
            if not value:
                continue
            result[key] = {k: _normalize_tool_schema(v) for k, v in value.items()}
        elif isinstance(value, dict):
            result[key] = _normalize_tool_schema(value)
        elif isinstance(value, list):
            result[key] = [_normalize_tool_schema(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value

    return result


def _normalize_tools(tools: Any) -> Any:
    """Normalise the ``parameters`` schema of every function tool in a request."""
    if not isinstance(tools, list):
        return tools

    normalized = []
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get('function'), dict):
            normalized.append(tool)
            continue

        fn = dict(tool['function'])
        fn.pop('strict', None)
        params = _normalize_tool_schema(fn.get('parameters'))
        # A tool with no parameters at all must still declare an object type,
        # or the same "should be of type OBJECT" rejection applies.
        if not isinstance(params, dict):
            params = {'type': 'object'}
        elif not params.get('type'):
            params = {**params, 'type': 'object'}
        fn['parameters'] = params
        normalized.append({**tool, 'function': fn})

    return normalized


def sanitize_openai_params(data: dict[str, Any]) -> dict[str, Any]:
    """Drop OpenAI-only request fields unsupported by generic host-side clients."""
    sanitized = dict(data)
    dropped = sorted(key for key in _UNSUPPORTED_COMPLETION_FIELDS if key in sanitized)
    for key in dropped:
        sanitized.pop(key, None)

    if dropped:
        logger.debug('Dropping unsupported completion fields from runtime request: %s', ', '.join(dropped))

    if sanitized.get('tools'):
        sanitized['tools'] = _normalize_tools(sanitized['tools'])

    return sanitized


class LLMRequestHandler:
    """Handles LLM requests from the runtime side.

    This class manages the Socket.IO event handling for LLM requests
    coming from proxy servers running in the runtime container.
    """

    def __init__(
        self,
        runtime: 'Runtime',
        openai_client: OpenAICompatibleClient,
        no_thinking_request_args: dict[str, Any] | None = None,
        no_thinking_request_args_mode: NoThinkingRequestArgsMode = 'replace',
        haiku_model: str | None = None,
        preserve_thinking: bool = False,
        on_tool_call: Callable[[str], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.openai_client = openai_client
        # Socket.IO dispatch runs in its own task, not necessarily the caller
        # task that installed a context-local ChatTracker. Create each request
        # task from this captured context so spans keep the correct processor.
        self._request_context = contextvars.copy_context()
        self._active_requests = 0
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self.no_thinking_request_args = no_thinking_request_args
        if no_thinking_request_args_mode not in ('replace', 'inherit'):
            raise ValueError(
                f'no_thinking_request_args_mode must be replace or inherit, got {no_thinking_request_args_mode!r}'
            )
        self.no_thinking_request_args_mode = no_thinking_request_args_mode
        self.haiku_model = haiku_model
        self.preserve_thinking = preserve_thinking
        # Optional host-side observer of tool_call names. Used by kilo to
        # detect 'suggest' tool calls and abort the long-running invoke early
        # (see installed_common.call_runtime_agent and kilo/agent.py).
        self.on_tool_call = on_tool_call
        # Cache of past assistant turns' thinking_blocks and any message-level
        # thought signatures. Each entry carries stable surface identifiers from
        # the assistant message, so re-injection can avoid positional guesses
        # when a scaffold rewrites or drops turns.
        self._thinking_cache: list[dict[str, Any]] = []
        # Same idea as _thinking_cache, but for OpenAI Responses reasoning
        # items carried on chat-like messages by litellm's openai/responses
        # bridge. Runtime chat-proxy scaffolds often drop these non-standard
        # fields, so we conservatively restore them before the next host call.
        self._reasoning_items_cache: list[dict[str, Any]] = []
        self.pending_requests: dict[str, asyncio.Future] = {}
        self._guard = RPCReplayGuard(ttl_seconds=300.0)
        self._report_interval = RSIOLLMConfig().report_interval
        self._handler_registered: bool = False
        self._default_router: LLMRequestRouter | None = None
        # Outcome of every request this handler settled, keyed by request_id and ordered by
        # first completion: error code for a failure, None for a success.
        #
        # An agent whose loop runs inside the sandbox cannot `except ContextLengthExceededError`
        # the way in-process agents (mini_swe_agent, tau2, ace) do: the error crosses the
        # Socket.IO boundary and reaches the agent as an opaque HTTP 500, so it can only report a
        # generic failure. Only the host still knows the true cause.
        #
        # This must stay per-request rather than a handler-wide flag: one handler serves every
        # request of a single call_runtime_agent, so a run where an early call is refused on
        # budget and a later one succeeds would otherwise report the whole run as truncated.
        self._request_outcomes: dict[str, str | None] = {}
        # Same keys and order as _request_outcomes, describing the first choice of a successful
        # completion. A failed request records None and False.
        self._request_finish_reasons: dict[str, str | None] = {}
        self._request_unparsed_tool_calls: dict[str, bool] = {}

    @property
    def request_outcomes(self) -> dict[str, str | None]:
        """Per-request outcomes in completion order: request_id -> error code, or None on success."""
        return dict(self._request_outcomes)

    @property
    def request_finish_reasons(self) -> dict[str, str | None]:
        """Per-request first-choice finish_reason, keyed and ordered like ``request_outcomes``."""
        return dict(self._request_finish_reasons)

    @property
    def request_unparsed_tool_calls(self) -> dict[str, bool]:
        """Per-request unparsed-tool-call flags, keyed and ordered like ``request_outcomes``."""
        return dict(self._request_unparsed_tool_calls)

    @property
    def last_request_error_code(self) -> str | None:
        """Error code of the most recently settled request, or None if it succeeded.

        A caller classifying why a sandboxed agent exited should prefer evidence from the agent's
        own event stream, which names the call it actually died on. This is the host-side
        approximation for agents that report nothing usable: the last call to settle is the one
        the agent was on when it gave up. It can miss under concurrency — a call refused on budget
        may be followed by an unrelated success — so it is a fallback, never an override.
        """
        if not self._request_outcomes:
            return None
        return next(reversed(self._request_outcomes.values()))

    @property
    def last_request_finish_reason(self) -> str | None:
        """finish_reason of the most recently settled request, or None if it failed or had none.

        A sandboxed agent that receives a ``length``-cut reply without tool calls usually treats it
        as its final message and exits cleanly; this lets the caller see that the answer was cut.
        Same last-settled approximation as ``last_request_error_code``.
        """
        if not self._request_finish_reasons:
            return None
        return next(reversed(self._request_finish_reasons.values()))

    @property
    def last_request_left_tool_call_unparsed(self) -> bool:
        """Whether the most recently settled request began a tool call the parser could not extract.

        Such a reply reaches the agent as plain text without tool calls, so a sandboxed agent takes
        the raw tool-call markup as its final answer and exits cleanly.
        """
        if not self._request_unparsed_tool_calls:
            return False
        return next(reversed(self._request_unparsed_tool_calls.values()))

    def _record_outcome(self, request_id: str, error_code: str | None, response: Any = None) -> None:
        # Keyed by request_id so a resend of an already-settled call neither duplicates an entry
        # nor reorders the one already recorded.
        if request_id not in self._request_outcomes:
            self._request_outcomes[request_id] = error_code
            choice = self._first_choice(response)
            finish_reason = choice.get('finish_reason')
            self._request_finish_reasons[request_id] = finish_reason if isinstance(finish_reason, str) else None
            provider_fields = choice.get('provider_specific_fields')
            self._request_unparsed_tool_calls[request_id] = isinstance(provider_fields, dict) and bool(
                provider_fields.get(UNPARSED_TOOL_CALL_FIELD)
            )

    @staticmethod
    def _first_choice(response: Any) -> dict[str, Any]:
        choices = response.get('choices') if isinstance(response, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return {}
        return choices[0]

    @staticmethod
    def _first_tool_call_id(msg: dict[str, Any]) -> str | None:
        for tc in msg.get('tool_calls') or []:
            tc_id = (tc or {}).get('id')
            if tc_id:
                return str(tc_id)
        return None

    @staticmethod
    def _tool_call_ids(msg: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for tc in msg.get('tool_calls') or []:
            tc_id = (tc or {}).get('id')
            if tc_id:
                ids.append(str(tc_id))
        return ids

    @staticmethod
    def _normalize_tool_call_arguments(arguments: Any) -> str:
        """Canonicalize a tool_call's arguments for cross-scaffold comparison.

        Chat-proxy scaffolds (opencode, kilo) round-trip tool_calls through their
        own JSON encoders, which preserve the semantic arguments but reorder object
        keys. Parse and re-dump with sorted keys so two semantically identical
        argument strings compare equal; fall back to the raw string when it is not
        valid JSON.
        """
        if arguments is None:
            return ''
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except (ValueError, TypeError):
                return arguments
        else:
            parsed = arguments
        try:
            return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(arguments)

    @classmethod
    def _tool_calls_for_match(cls, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Return ordered, normalized tool_calls for conservative state matching.

        Tool call order is part of the replay surface. Do not sort or compare
        as a set: if a scaffold reorders calls, skip injection rather than
        risk attaching opaque reasoning state to the wrong assistant turn.

        Each entry is normalized to ``{id, name, arguments}`` so that fields a
        scaffold may add or drop on the round-trip (e.g. the streaming ``index``)
        and JSON key ordering inside ``arguments`` do not defeat the match. The
        anchor remains the globally-unique tool ``id``; name/arguments are a
        secondary cross-check, not a positional guess.
        """
        tool_calls: list[dict[str, Any]] = []
        for tc in msg.get('tool_calls') or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get('function') or {}
            tool_calls.append({
                'id': str(tc['id']) if tc.get('id') else None,
                'name': fn.get('name') if isinstance(fn, dict) else None,
                'arguments': cls._normalize_tool_call_arguments(fn.get('arguments') if isinstance(fn, dict) else None),
            })
        return tool_calls

    @staticmethod
    def _plaintext_content(msg: dict[str, Any]) -> str | None:
        """Extract comparable assistant-visible text from OpenAI-style content."""
        content = msg.get('content')
        if content is None:
            return None
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                text = part.get('text') or part.get('input_text')
                if isinstance(text, str):
                    parts.append(text)
            return ''.join(parts) if parts else None
        return None

    @staticmethod
    def _thinking_entry_has_payload(entry: dict[str, Any]) -> bool:
        """Whether a cache entry still carries something worth replaying."""
        return bool(entry.get('blocks')) or bool(entry.get('provider_specific_fields'))

    def _find_cached_thinking_entry(
        self,
        *,
        msg: dict[str, Any],
        used: list[bool],
    ) -> dict[str, Any] | None:
        """Find a cached thinking entry without falling back to turn position."""
        tc_id = self._first_tool_call_id(msg)
        content = self._plaintext_content(msg)
        if tc_id:
            matches = [
                (j, entry)
                for j, entry in enumerate(self._thinking_cache)
                if not used[j]
                and self._thinking_entry_has_payload(entry)
                and entry.get('tool_call_id') == tc_id
                and entry.get('content') == content
            ]
            # Some scaffolds (OpenClaw) persist provider ids without ``_``.
            if len(matches) != 1:
                stripped = tc_id.replace('_', '')
                matches = [
                    (j, entry)
                    for j, entry in enumerate(self._thinking_cache)
                    if not used[j]
                    and self._thinking_entry_has_payload(entry)
                    and entry.get('tool_call_id')
                    and str(entry['tool_call_id']).replace('_', '') == stripped
                    and entry.get('content') == content
                ]
            if len(matches) != 1:
                return None

            j, entry = matches[0]
            used[j] = True
            return entry

        if not content:
            return None

        matches = [
            (j, entry)
            for j, entry in enumerate(self._thinking_cache)
            if not used[j]
            and not entry.get('tool_call_id')
            and entry.get('content') == content
            and self._thinking_entry_has_payload(entry)
        ]
        if len(matches) != 1:
            return None

        j, entry = matches[0]
        used[j] = True
        return entry

    def _rewrite_stripped_tool_ids(self, messages: list[dict[str, Any]]) -> int:
        """Write host-issued tool ids back when a scaffold sanitized them.

        OpenClaw keeps only ``[A-Za-z0-9]`` and may truncate, so the id it replays no longer matches the thinking cache. An incoming id is restored only when it maps to exactly one cached id.
        """
        cached_ids = [entry.get('tool_call_id') for entry in self._thinking_cache]
        for entry in self._reasoning_items_cache:
            cached_ids.extend(entry.get('tool_call_ids') or [])
        alnum_keys = {str(orig): _NON_ALNUM.sub('', str(orig)) for orig in cached_ids if orig}

        def resolve(incoming: str) -> str | None:
            if incoming in alnum_keys:
                return None
            if _NON_ALNUM.search(incoming):
                matches = {orig for orig in alnum_keys if orig.replace('_', '') == incoming}
            else:
                matches = {
                    orig
                    for orig, key in alnum_keys.items()
                    if key == incoming or (len(incoming) >= _MIN_TRUNCATED_TOOL_ID_LEN and key.startswith(incoming))
                }
            return matches.pop() if len(matches) == 1 else None

        rewritten = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            incoming = msg.get('tool_call_id')
            mapped = resolve(str(incoming)) if incoming else None
            if mapped:
                msg['tool_call_id'] = mapped
                rewritten += 1
            for tc in msg.get('tool_calls') or []:
                if not isinstance(tc, dict) or not tc.get('id'):
                    continue
                mapped = resolve(str(tc['id']))
                if mapped:
                    tc['id'] = mapped
                    rewritten += 1
        return rewritten

    def _inject_thinking_blocks(self, messages: list[dict[str, Any]]) -> int:
        """Restore thinking state (blocks and/or message-level signatures).

        Returns the number of assistant messages that received an injection.
        Matching is intentionally conservative: tool-call turns require both
        tool_call_id and exact plaintext content to match. Plain assistant turns
        without tool calls can match by unique exact plaintext content. Missing
        matches are skipped rather than falling back to turn position, because a
        wrong signed thinking block is worse than none.
        """
        if not self._thinking_cache:
            return 0
        injected = 0
        used = [False] * len(self._thinking_cache)
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            # Only replayable (signed/redacted) blocks are authoritative. Blocks a
            # scaffold echoes back with an empty signature (anthropic_proxy emits
            # ``signature=''`` when the provider signed at message level, e.g.
            # Gemini) carry no state, so they must not block restoring it.
            if self._has_replayable_thinking(msg):
                continue

            entry = self._find_cached_thinking_entry(msg=msg, used=used)
            if entry is None:
                continue

            restored = False
            if entry.get('blocks'):
                msg['thinking_blocks'] = [dict(b) for b in entry['blocks']]
                restored = True
            if entry.get('provider_specific_fields') and not msg.get('provider_specific_fields'):
                msg['provider_specific_fields'] = copy.deepcopy(entry['provider_specific_fields'])
                restored = True
            if restored:
                injected += 1
        return injected

    @staticmethod
    def _message_thought_signatures(msg: dict[str, Any]) -> list[str]:
        """Gemini's message-level thought signatures, if this response carries any.

        Anthropic-style providers sign each thinking block. Gemini instead leaves
        ``thinking_blocks[].signature`` unset and returns opaque signatures for the
        non-functionCall parts under ``provider_specific_fields.thought_signatures``
        (tool-call signatures ride on the tool_call itself). Both have to be replayed
        verbatim or the model loses the reasoning state behind the turn.
        """
        provider_fields = msg.get('provider_specific_fields')
        if not isinstance(provider_fields, dict):
            return []
        signatures = provider_fields.get('thought_signatures')
        if not isinstance(signatures, list):
            return []
        return [str(s) for s in signatures if s]

    @staticmethod
    def _is_replayable_thinking_block(block: Any) -> bool:
        """Whether a thinking block can be replayed to the provider as reasoning state.

        ``thinking`` blocks need a non-empty signature; ``redacted_thinking`` blocks
        carry their state in ``data``. Anything else is not a thinking block and is
        left alone.
        """
        if not isinstance(block, dict):
            return True
        block_type = block.get('type')
        if block_type == 'thinking':
            return bool(block.get('signature'))
        if block_type == 'redacted_thinking':
            return bool(block.get('data'))
        return True

    @classmethod
    def _has_replayable_thinking(cls, msg: dict[str, Any]) -> bool:
        blocks = msg.get('thinking_blocks')
        if not isinstance(blocks, list):
            return False
        return any(
            isinstance(b, dict)
            and b.get('type') in ('thinking', 'redacted_thinking')
            and cls._is_replayable_thinking_block(b)
            for b in blocks
        )

    @classmethod
    def _drop_unreplayable_thinking_blocks(cls, messages: list[dict[str, Any]]) -> int:
        """Remove thinking blocks that carry no signature before they reach litellm.

        litellm treats any non-None signature as valid: Gemini turns
        ``{thinking, signature: ''}`` into a plain text part (the summary is replayed
        as visible model output), and Anthropic turns unsigned blocks into text
        blocks. An unsigned block never restores reasoning state, so it is dropped
        for every provider. Returns the number of blocks removed.
        """
        dropped = 0
        for msg in messages:
            if not isinstance(msg, dict) or msg.get('role') != 'assistant':
                continue
            blocks = msg.get('thinking_blocks')
            if not isinstance(blocks, list):
                continue
            kept = [b for b in blocks if cls._is_replayable_thinking_block(b)]
            if len(kept) == len(blocks):
                continue
            dropped += len(blocks) - len(kept)
            if kept:
                msg['thinking_blocks'] = kept
            else:
                msg.pop('thinking_blocks', None)
        return dropped

    def _cache_thinking_from_response(self, response_data: dict[str, Any]) -> tuple[int, str | None]:
        """Cache this assistant turn's thinking state for later replay.

        Signed blocks are cached as-is. When the provider signs at the message level
        instead, the unsigned blocks are kept alongside those signatures so the pair
        can be restored together — dropping either half loses the reasoning state.
        """
        choices = response_data.get('choices') or []
        if not choices:
            return 0, None
        msg = choices[0].get('message') or {}
        blocks_raw = [b for b in msg.get('thinking_blocks') or [] if isinstance(b, dict)]
        blocks = [dict(b) for b in blocks_raw if b.get('signature')]

        signatures = self._message_thought_signatures(msg)
        provider_fields = {'thought_signatures': signatures} if signatures else None
        if provider_fields and not blocks:
            blocks = [dict(b) for b in blocks_raw]

        if not blocks and not provider_fields:
            return 0, None

        tc_id: str | None = None
        for tc in msg.get('tool_calls') or []:
            if (tc or {}).get('id'):
                tc_id = str(tc['id'])
                break
        self._thinking_cache.append({
            'tool_call_id': tc_id,
            'content': self._plaintext_content(msg),
            'blocks': blocks,
            'provider_specific_fields': provider_fields,
        })
        return len(blocks) or len(signatures), tc_id

    @staticmethod
    def _normalize_reasoning_item(item: dict[str, Any]) -> dict[str, Any] | None:
        """Copy a Responses reasoning item without leaking invalid/null fields."""
        if not isinstance(item, dict):
            return None

        encrypted_content = item.get('encrypted_content')
        item_id = item.get('id')
        if encrypted_content and not item_id:
            logger.warning('preserve_reasoning: skipping encrypted reasoning item without id')
            return None

        normalized: dict[str, Any] = {'type': item.get('type') or 'reasoning'}
        if item_id:
            normalized['id'] = str(item_id)

        summary = item.get('summary')
        if isinstance(summary, list):
            normalized['summary'] = copy.deepcopy(summary)
        elif summary is not None:
            normalized['summary'] = summary

        content = item.get('content')
        if isinstance(content, list):
            normalized['content'] = copy.deepcopy(content)
        elif content is not None:
            normalized['content'] = content

        if encrypted_content:
            normalized['encrypted_content'] = str(encrypted_content)

        status = item.get('status')
        if status:
            normalized['status'] = status

        return normalized

    def _match_tool_reasoning_entry(
        self,
        *,
        tool_call_ids: list[str],
        tool_calls: list[dict[str, Any]],
        content: str | None,
        used: list[bool],
    ) -> int | None:
        """Index of the unique cached entry for this assistant turn's tool calls.

        Anchored on the ordered, globally-unique tool ``id`` list; the normalized
        tool_calls (name + canonical arguments, see ``_tool_calls_for_match``) are a
        secondary cross-check so a scaffold's added ``index`` field or reordered
        argument keys do not defeat the match. When the cached entry also carries
        plaintext content it must equal the message's content. Returns None when the
        match is absent or ambiguous (never guesses by position).
        """
        matches: list[int] = []
        for j, entry in enumerate(self._reasoning_items_cache):
            if used[j] or not entry.get('items'):
                continue
            if entry.get('tool_call_ids') != tool_call_ids:
                continue
            if entry.get('tool_calls') != tool_calls:
                continue
            entry_content = entry.get('content')
            if entry_content and content and entry_content != content:
                continue
            matches.append(j)
        if len(matches) == 1:
            return matches[0]

        # Same fallback as thinking: scaffolds may persist ids without ``_``.
        stripped_ids = [item.replace('_', '') for item in tool_call_ids]
        stripped_calls = [{**call, 'id': (call.get('id') or '').replace('_', '')} for call in tool_calls]
        matches = []
        for j, entry in enumerate(self._reasoning_items_cache):
            if used[j] or not entry.get('items'):
                continue
            cached_ids = [item.replace('_', '') for item in (entry.get('tool_call_ids') or [])]
            if cached_ids != stripped_ids:
                continue
            cached_calls = [
                {**call, 'id': (call.get('id') or '').replace('_', '')} for call in (entry.get('tool_calls') or [])
            ]
            if cached_calls != stripped_calls:
                continue
            entry_content = entry.get('content')
            if entry_content and content and entry_content != content:
                continue
            matches.append(j)
        return matches[0] if len(matches) == 1 else None

    def _match_content_reasoning_entry(
        self,
        *,
        content: str | None,
        used: list[bool],
        exclude: int | None = None,
    ) -> int | None:
        """Index of the unique content-only cached entry matching ``content``.

        A chat-proxy scaffold (e.g. opencode) collapses a multi-choice Responses
        reply -- a text choice plus a separate function_call choice -- into one
        assistant message carrying both content and tool_calls. The reasoning that
        rode on the text choice was cached as a content-only entry (no tool ids), so
        it must be recovered by exact plaintext content rather than tool id. Returns
        None when absent or ambiguous.
        """
        if not content:
            return None
        matches = [
            j
            for j, entry in enumerate(self._reasoning_items_cache)
            if not used[j]
            and j != exclude
            and not entry.get('tool_call_ids')
            and entry.get('content') == content
            and entry.get('items')
        ]
        return matches[0] if len(matches) == 1 else None

    def _take_cached_reasoning_entry(
        self,
        idx: int,
        used: list[bool],
        collected: list[dict[str, Any]],
        output_items: list[dict[str, Any]],
    ) -> None:
        used[idx] = True
        entry = self._reasoning_items_cache[idx]
        collected.extend(entry['items'])
        extra = entry.get('output_items')
        if extra:
            output_items.extend(copy.deepcopy(extra))

    def _collect_cached_reasoning(
        self, *, msg: dict[str, Any], used: list[bool]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Gather cached reasoning_items for one assistant message, in replay order.

        A single scaffold-rewritten assistant message can need reasoning from more
        than one cached turn: the text choice's reasoning (matched by content) and
        the tool-call choice's reasoning (matched by tool id). They are returned in
        temporal order -- content reasoning first, then tool-call reasoning -- so
        litellm's Responses bridge replays them in the same order it emitted them
        (all reasoning_items lower ahead of the function_calls). Matched entries are
        marked used so they are not double-injected.

        Also returns the Polarist ``responses_output_items`` snapshot from those
        same cache entries (empty when the host response did not carry PSF).
        """
        tool_call_ids = self._tool_call_ids(msg)
        content = self._plaintext_content(msg)
        collected: list[dict[str, Any]] = []
        output_items: list[dict[str, Any]] = []

        if tool_call_ids:
            tool_idx = self._match_tool_reasoning_entry(
                tool_call_ids=tool_call_ids,
                tool_calls=self._tool_calls_for_match(msg),
                content=content,
                used=used,
            )
            # Only pull a separate content-only entry for the merged multi-choice
            # shape, where the tool entry carries no content of its own. If the tool
            # entry already owns this content (HY-style single choice), a same-text
            # content-only entry would be a coincidental different turn -- skip it.
            tool_entry = self._reasoning_items_cache[tool_idx] if tool_idx is not None else None
            if content and not (tool_entry and tool_entry.get('content')):
                content_idx = self._match_content_reasoning_entry(content=content, used=used, exclude=tool_idx)
                if content_idx is not None:
                    self._take_cached_reasoning_entry(content_idx, used, collected, output_items)
            if tool_idx is not None:
                self._take_cached_reasoning_entry(tool_idx, used, collected, output_items)
            return collected, output_items

        content_idx = self._match_content_reasoning_entry(content=content, used=used)
        if content_idx is not None:
            self._take_cached_reasoning_entry(content_idx, used, collected, output_items)
        return collected, output_items

    def _item_by_encrypted_blob(self) -> dict[str, dict[str, Any]]:
        """Map unique cached ``encrypted_content`` blobs to their provider item.

        Duplicate blobs that disagree are dropped so we never guess an ``rs_*`` id.
        """
        found: dict[str, dict[str, Any]] = {}
        clash: set[str] = set()
        for entry in self._reasoning_items_cache:
            for cached in entry.get('items') or []:
                if not isinstance(cached, dict):
                    continue
                blob = cached.get('encrypted_content')
                if not blob:
                    continue
                blob = str(blob)
                prev = found.get(blob)
                if prev is not None and prev != cached:
                    clash.add(blob)
                else:
                    found[blob] = cached
        for blob in clash:
            found.pop(blob, None)
        return found

    def _align_reasoning_ids(self, items: list[Any]) -> list[Any]:
        """Replace items with the cached provider item when encrypted_content uniquely matches."""
        by_blob = self._item_by_encrypted_blob()
        aligned: list[Any] = []
        for item in items:
            blob = item.get('encrypted_content') if isinstance(item, dict) else None
            cached = by_blob.get(str(blob)) if blob else None
            aligned.append(copy.deepcopy(cached) if cached is not None else item)
        return aligned

    @staticmethod
    def _restore_responses_output_items(msg: dict[str, Any], output_items: list[dict[str, Any]]) -> bool:
        """Write the cached Polarist output[] onto the assistant message PSF."""
        if not output_items:
            return False
        fields = dict(msg.get('provider_specific_fields') or {})
        if fields.get('responses_output_items') == output_items:
            return False
        fields['responses_output_items'] = output_items
        msg['provider_specific_fields'] = fields
        return True

    def _inject_reasoning_items(self, messages: list[dict[str, Any]]) -> int:
        """Restore cached Responses reasoning_items onto assistant messages."""
        if not self._reasoning_items_cache:
            return 0

        injected = 0
        used = [False] * len(self._reasoning_items_cache)
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            existing_items = msg.get('reasoning_items')
            # Responses proxies may preserve encrypted_content while rewriting
            # its provider-bound item id. For an exact tool-call match, restore
            # the authoritative item returned by the host.
            # Content-only turns often already carry reasoning_items: Codex
            # keeps encrypted_content but rewrites the provider-bound rs_* id.
            # Align by blob instead of skipping the whole turn.
            if existing_items and not self._tool_call_ids(msg):
                aligned = self._align_reasoning_ids(existing_items)
                if aligned != existing_items:
                    msg['reasoning_items'] = aligned
                    injected += 1
                continue

            items, output_items = self._collect_cached_reasoning(msg=msg, used=used)
            if not items:
                continue
            touched = False
            if items != existing_items:
                msg['reasoning_items'] = copy.deepcopy(items)
                touched = True
            if self._restore_responses_output_items(msg, output_items):
                touched = True
            if touched:
                injected += 1
        return injected

    def _cache_reasoning_items_from_response(self, response_data: dict[str, Any]) -> tuple[int, str | None]:
        """Cache this assistant turn's Responses reasoning_items for replay."""
        total_items = 0
        first_key: str | None = None

        for choice in response_data.get('choices') or []:
            msg = (choice or {}).get('message') or {}
            raw_items = msg.get('reasoning_items') or []
            if not isinstance(raw_items, list):
                continue

            items = [
                normalized
                for item in raw_items
                if isinstance(item, dict)
                for normalized in [self._normalize_reasoning_item(item)]
                if normalized is not None
            ]
            if not items:
                continue

            tool_call_ids = self._tool_call_ids(msg)
            tool_calls = self._tool_calls_for_match(msg)
            content = self._plaintext_content(msg)
            fields = msg.get('provider_specific_fields')
            raw_output_items = fields.get('responses_output_items') if isinstance(fields, dict) else None
            output_items = (
                copy.deepcopy(raw_output_items) if isinstance(raw_output_items, list) and raw_output_items else None
            )
            self._reasoning_items_cache.append({
                'tool_call_ids': tool_call_ids,
                'tool_calls': tool_calls,
                'content': content,
                'items': items,
                'output_items': output_items,
            })
            total_items += len(items)
            if first_key is None:
                first_key = ','.join(tool_call_ids) if tool_call_ids else '<content>'

        return total_items, first_key

    @staticmethod
    def _maybe_tamper_for_test(messages: list[dict[str, Any]]) -> int | None:
        """Optional fault injector: corrupt the Nth assistant turn's signature.

        Activated by env var ``NEXUS_TAMPER_THINKING_INDEX`` (0-based index over
        assistant messages that carry thinking_blocks). Used to verify that
        Bedrock validates each historical signature independently and reports
        the precise message index. Returns the assistant index that was tampered,
        or None if disabled / not applicable.
        """
        raw = os.environ.get('NEXUS_TAMPER_THINKING_INDEX')
        if raw is None:
            return None
        try:
            target = int(raw)
        except ValueError:
            return None
        seen = 0
        for msg in messages:
            if msg.get('role') != 'assistant':
                continue
            tbs = msg.get('thinking_blocks') or []
            if not tbs:
                continue
            if seen == target:
                first = tbs[0]
                if isinstance(first, dict) and first.get('signature'):
                    sig = first['signature']
                    # Flip one base64 char so the signature is still well-formed
                    # base64 but cryptographically invalid.
                    flip = 'B' if sig[0] != 'B' else 'C'
                    first['signature'] = flip + sig[1:]
                    return target
            seen += 1
        return None

    async def _produce_response(
        self, request_id: str, data: dict[str, Any], *, rsio_mode: bool = True
    ) -> dict[str, Any]:
        """Run the host-side LLM call and return the response dump.

        ``data`` has already been copied and had ``rsio`` / ``request_id``
        stripped by :meth:`handle_llm_request`. Raises on failure. The periodic
        ``running`` heartbeat is emitted only in ``rsio_mode`` — an old worker
        cannot digest it.
        """
        operation = data.pop('_nexus_operation', None)
        if operation == 'responses_compact':
            request_body = data.pop('request_body', None)
            compact = getattr(self.openai_client, 'compact_response', None)
            supports_native = bool(getattr(self.openai_client, 'supports_native_responses_compaction', False))
            if not supports_native or not callable(compact):
                raise RuntimeError(
                    'configured OpenAI client does not support provider-native Responses compaction; '
                    'HyTuner training must use local summary compaction'
                )
            if not isinstance(request_body, dict) or data:
                raise ValueError('invalid Responses compaction bridge payload')
            heartbeat = asyncio.create_task(self._run_heartbeat(request_id)) if rsio_mode else None
            try:
                with span_context(
                    'agent.compaction',
                    attributes={
                        'type': 'agent_compaction',
                        'compacted': False,
                        'source': 'provider_native',
                        'trainable': False,
                    },
                ) as compaction_span:
                    response = await compact(request_body)
                    output = response.get('output') if isinstance(response, dict) else None
                    compaction_items = (
                        [item for item in output if isinstance(item, dict) and item.get('type') == 'compaction']
                        if isinstance(output, list)
                        else []
                    )
                    if (
                        not isinstance(output, list)
                        or len(compaction_items) != 1
                        or not isinstance(compaction_items[0].get('id'), str)
                        or not isinstance(compaction_items[0].get('encrypted_content'), str)
                    ):
                        raise ValueError(
                            'native Responses compaction must return exactly one compaction item among '
                            'the output items, with string id and encrypted_content'
                        )
                    compaction_span.update(attributes={'compacted': True})
                    return response
            finally:
                if heartbeat is not None:
                    heartbeat.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await heartbeat
        if operation is not None:
            raise ValueError(f'unsupported Nexus LLM operation: {operation}')

        # Proxies own compaction classification. Consume their transport-only metadata before
        # sanitization and provider dispatch; message text must not override that classification.
        is_agent_compaction = data.pop(NEXUS_AGENT_COMPACTION_FIELD, False) is True
        runtime_detector = data.pop(NEXUS_COMPACTION_DETECTOR_FIELD, None)
        compaction_detector = runtime_detector if isinstance(runtime_detector, str) else 'runtime_marker'
        client_supports_request_meta = getattr(self.openai_client, 'supports_request_meta', False)
        if is_agent_compaction and client_supports_request_meta:
            request_meta = data.get('request_meta')
            if request_meta is None:
                request_meta = {}
            elif not isinstance(request_meta, dict):
                raise TypeError('Compaction request_meta must be a dictionary')
            data['request_meta'] = {
                **request_meta,
                'invocation_kind': 'summary_inference',
            }

        requested_model = data.get('model')
        host_request_args = self.openai_client.config.request_args
        host_model = host_request_args.get('model')
        if host_model:
            data.pop('model', None)
        # Preserve the historical handler contract: request_args.reasoning_effort
        # is host-pinned, while max_tokens remains a default unless the client
        # explicitly declares it under enforced_request_args.
        if host_request_args.get('reasoning_effort') is not None:
            data.pop('reasoning_effort', None)

        try:
            client_has_thinking = data.pop('_client_has_thinking', None)
            client_requests_haiku = data.pop('_client_requests_haiku', None)

            if client_has_thinking is False and self.no_thinking_request_args is not None:
                no_think_args = copy.deepcopy(self.no_thinking_request_args)
                if self.no_thinking_request_args_mode == 'inherit':
                    no_think_overrides = no_think_args
                    no_think_args = copy.deepcopy(host_request_args)
                    no_think_args.update(no_think_overrides)
                    for field in ('thinking', 'reasoning_effort', 'output_config'):
                        if field not in no_think_overrides:
                            no_think_args.pop(field, None)
                    extra_body = no_think_args.get('extra_body')
                    override_extra_body = no_think_overrides.get('extra_body')
                    if isinstance(extra_body, dict):
                        extra_body = dict(extra_body)
                        for field in ('thinking', 'output_config'):
                            if not isinstance(override_extra_body, dict) or field not in override_extra_body:
                                extra_body.pop(field, None)
                        if extra_body:
                            no_think_args['extra_body'] = extra_body
                        else:
                            no_think_args.pop('extra_body', None)
                if client_requests_haiku and self.haiku_model:
                    no_think_args.pop('model', None)
                data['_request_args_override'] = no_think_args

            if client_requests_haiku and self.haiku_model:
                data['model'] = self.haiku_model

            if 'request_meta' in data and not client_supports_request_meta:
                data.pop('request_meta', None)

            final_model = data.get('model') or host_model or requested_model
            thinking_str = 'no_thinking' if client_has_thinking is False else 'thinking'
            haiku_str = ' [haiku]' if client_requests_haiku else ''
            logger.debug(
                f'Handling LLM request: {request_id}, '
                f'requested_model={requested_model}, final_model={final_model}, '
                f'{thinking_str}{haiku_str}'
            )

            request_data = sanitize_openai_params(data)

            if self.preserve_thinking and isinstance(request_data.get('messages'), list):
                rewritten = self._rewrite_stripped_tool_ids(request_data['messages'])
                if rewritten:
                    logger.info(
                        'preserve_thinking: rewrote %d scaffold-stripped tool id(s) [%s]',
                        rewritten,
                        request_id,
                    )
                injected = self._inject_thinking_blocks(request_data['messages'])
                if injected:
                    logger.debug(
                        'preserve_thinking: injected thinking_blocks into %d assistant message(s) [%s]',
                        injected,
                        request_id,
                    )
                reasoning_injected = self._inject_reasoning_items(request_data['messages'])
                if reasoning_injected:
                    logger.debug(
                        'preserve_reasoning: injected reasoning_items into %d assistant message(s) [%s]',
                        reasoning_injected,
                        request_id,
                    )
                tampered = self._maybe_tamper_for_test(request_data['messages'])
                if tampered is not None:
                    logger.warning(
                        'preserve_thinking: TAMPERED signature on assistant turn #%d for diagnostics [%s]',
                        tampered,
                        request_id,
                    )

            if isinstance(request_data.get('messages'), list):
                unsigned_dropped = self._drop_unreplayable_thinking_blocks(request_data['messages'])
                if unsigned_dropped:
                    logger.debug(
                        'dropped %d unsigned thinking block(s) before replay [%s]',
                        unsigned_dropped,
                        request_id,
                    )

            heartbeat = asyncio.create_task(self._run_heartbeat(request_id)) if rsio_mode else None
            try:
                if is_agent_compaction:
                    with span_context(
                        'agent.compaction',
                        attributes={
                            'type': 'agent_compaction',
                            'compacted': False,
                            'source': 'runtime_marker',
                            'detector': compaction_detector,
                        },
                    ) as compaction_span:
                        response = await self.openai_client.create_completion(**request_data)
                        compaction_span.update(attributes={'compacted': True})
                else:
                    response = await self.openai_client.create_completion(**request_data)
            finally:
                if heartbeat is not None:
                    heartbeat.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await heartbeat

            response_dump = response.model_dump() if hasattr(response, 'model_dump') else response

            if self.preserve_thinking and isinstance(response_dump, dict):
                cached_n, cached_key = self._cache_thinking_from_response(response_dump)
                if cached_n:
                    logger.debug(
                        'preserve_thinking: cached %d thinking block(s) (key=%s) [%s]',
                        cached_n,
                        cached_key or '<content>',
                        request_id,
                    )
                reasoning_cached_n, reasoning_cached_key = self._cache_reasoning_items_from_response(response_dump)
                if reasoning_cached_n:
                    logger.debug(
                        'preserve_reasoning: cached %d reasoning item(s) (key=%s) [%s]',
                        reasoning_cached_n,
                        reasoning_cached_key or '<unknown>',
                        request_id,
                    )

            # Notify host-side tool_call observer (used by kilo to detect 'suggest').
            if self.on_tool_call is not None and isinstance(response_dump, dict):
                for choice in response_dump.get('choices') or []:
                    message = (choice or {}).get('message') or {}
                    for tc in message.get('tool_calls') or []:
                        fn = (tc or {}).get('function') or {}
                        fn_name = fn.get('name') or ''
                        if fn_name:
                            try:
                                self.on_tool_call(fn_name)
                            except Exception:
                                logger.exception('on_tool_call callback raised; ignoring')

            if self.openai_client.config.client_args.get('log_llm_usage'):
                rd = response_dump
                usage = (rd.get('usage') or {}) if isinstance(rd, dict) else {}
                cache_read = usage.get('cache_read_input_tokens', 0) or 0
                cache_creation = usage.get('cache_creation_input_tokens', 0) or 0
                input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0) or 0
                output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0) or 0
                total_cached = cache_read + cache_creation
                hit_rate = (cache_read / total_cached * 100) if total_cached > 0 else 0
                prompt_details = usage.get('prompt_tokens_details') or {}
                input_details = usage.get('input_tokens_details') or {}
                cached_tokens = 0
                if isinstance(prompt_details, dict):
                    cached_tokens = (
                        prompt_details.get('gpu_cached_tokens', 0) or prompt_details.get('cached_tokens', 0) or 0
                    )
                if not cached_tokens and isinstance(input_details, dict):
                    cached_tokens = (
                        input_details.get('gpu_cached_tokens', 0) or input_details.get('cached_tokens', 0) or 0
                    )
                cached_input_rate = (
                    (cached_tokens / input_tokens * 100) if input_tokens > 0 and cached_tokens > 0 else 0
                )
                logger.debug(
                    'LLM usage [%s]: input=%d output=%d cache_read=%d cache_creation=%d hit_rate=%.1f%% cached_tokens=%d cached_input_rate=%.1f%%',
                    request_id,
                    input_tokens,
                    output_tokens,
                    cache_read,
                    cache_creation,
                    hit_rate,
                    cached_tokens,
                    cached_input_rate,
                )

            return response_dump

        except Exception as e:
            logger.error(f'LLM request {request_id} failed: {e}')
            raise

    async def _emit_running(self, request_id: str, *, kind: str = 'heartbeat') -> bool:
        """Emit one ``running`` report for *request_id*.

        ``kind`` is only for the local log line (``ack`` vs ``heartbeat``); the
        wire payload is identical. Both submit acknowledgements and periodic
        heartbeats stay at DEBUG to avoid flooding normal logs. Only ever
        called in RSIO mode — an old worker
        cannot digest a ``running`` report (it has no ``success`` field) and
        would mistake it for a failed final.

        Delivery is best-effort: if the socket is currently disconnected the
        report is dropped silently. The worker's rsio driver will resend the
        original request when it stops hearing heartbeats, and the replay
        guard will short-circuit the duplicate to the cached final anyway.
        """
        logger.debug('rsio.host: running %s request_id=%s', kind, request_id)
        try:
            await self.runtime.emit(
                'llm_response',
                {'request_id': request_id, 'rsio': {'kind': 'running'}},
            )
        except Exception as exc:
            logger.warning('rsio.host: running %s emit dropped request_id=%s (%s)', kind, request_id, exc)
            return False
        return True

    async def _run_heartbeat(self, request_id: str) -> None:
        """Periodically report that this request is still being processed."""
        failure_started_at: float | None = None
        try:
            while True:
                await asyncio.sleep(self._report_interval)
                if await self._emit_running(request_id):
                    failure_started_at = None
                    continue

                now = time.monotonic()
                if failure_started_at is None:
                    failure_started_at = now
                elif now - failure_started_at >= _HEARTBEAT_FAILURE_TIMEOUT:
                    logger.warning(
                        'rsio.host: heartbeat stopped after %.0fs of consecutive emit failures request_id=%s',
                        now - failure_started_at,
                        request_id,
                    )
                    return
        except asyncio.CancelledError:
            raise

    async def _emit_final(
        self,
        request_id: str,
        *,
        success: bool,
        response: dict[str, Any] | None = None,
        error: str | None = None,
        error_code: str | None = None,
        rsio_mode: bool = True,
    ) -> bool:
        """Emit the terminal ``llm_response`` for *request_id*.

        Delivery is best-effort: if the socket is disconnected right now, we
        log and swallow the exception. The worker will resend the request and
        the replay guard turns that into a fresh ``_emit_final`` call — which
        can then land on a reconnected socket. Raising here instead would
        break the enclosing ``handle_llm_request`` and leave the worker
        without an eventual final anyway. A return value of ``True`` means the
        Socket.IO emit completed without raising; it is not an end-to-end ACK.
        """
        payload: dict[str, Any] = {'request_id': request_id, 'success': success}
        if rsio_mode:
            # Old workers ignore an unknown ``rsio`` sub-field on the final, so
            # it is harmless to include — but we omit it in legacy mode to keep
            # the wire identical to the pre-RSIO response.
            payload['rsio'] = {'kind': 'final'}
        if success:
            payload['response'] = response
        else:
            payload['error'] = error or 'Unknown error'
            if error_code:
                # Machine-readable companion to the free-text ``error``. The worker turns it back
                # into a typed failure so the in-sandbox proxy can name the cause to its agent,
                # which otherwise only ever sees an opaque HTTP 500. Old workers ignore it.
                payload['error_code'] = error_code
        logger.debug('rsio.host: emit final request_id=%s success=%s rsio=%s', request_id, success, rsio_mode)
        try:
            await self.runtime.emit('llm_response', payload)
        except Exception as exc:
            logger.warning('rsio.host: final emit dropped request_id=%s (%s); worker will resend', request_id, exc)
            return False
        return True

    async def handle_llm_request(self, data: dict[str, Any]) -> None:
        """Handle one frame inside the context that created this channel handler."""
        request_context = self._request_context.copy()
        if self._active_requests == 0:
            self._idle_event.clear()
        self._active_requests += 1
        task = request_context.run(
            asyncio.create_task,
            self._handle_llm_request(data),
        )
        try:
            await task
        finally:
            self._active_requests -= 1
            if self._active_requests == 0:
                self._idle_event.set()

    def refresh_request_context(self) -> None:
        """Capture the caller's turn-local processors for subsequent requests."""
        if self._active_requests:
            raise RuntimeError('Cannot change LLM handler context while requests are in flight')
        self._request_context = contextvars.copy_context()

    async def wait_until_idle(self, *, timeout: float) -> None:
        """Wait until all model calls already accepted by this handler settle."""
        if self._active_requests == 0:
            return
        await asyncio.wait_for(self._idle_event.wait(), timeout=timeout)

    async def _handle_llm_request(self, data: dict[str, Any]) -> None:
        """Handle an LLM request from the runtime side.

        Dedups overlapping retries by ``request_id`` (same host process), emits
        ``running`` heartbeats while the model call runs, and always attempts a
        terminal ``llm_response``. A completed replay result is retained until
        that final emit completes without raising.

        Backward compatibility (new host + old worker): the inbound request
        carries an ``rsio`` sub-field only from a new worker. When it is absent
        the caller is an old worker that cannot digest ``running`` heartbeats
        and does not expect ``rsio`` on the response — so we skip the heartbeat
        and emit a plain legacy final.
        """
        data = dict(data)
        rsio = data.pop('rsio', None)  # protocol envelope — never forward to the provider
        rsio_mode = rsio is not None  # absent ⇒ old worker ⇒ legacy responses
        request_id = data.pop('request_id', None) or '<no-request-id>'
        attempt = (rsio or {}).get('attempt')
        logger.debug(
            'rsio.host: recv llm_request request_id=%s attempt=%s mode=%s',
            request_id,
            attempt,
            'rsio' if rsio_mode else 'legacy',
        )

        ran = False

        async def once() -> dict[str, Any]:
            nonlocal ran
            ran = True
            # Ack with a `running` report before starting the model call, and
            # only on the owning frame (guard runs `once` once per request_id).
            # This tells the caller "received, working on it" so its probe window
            # switches from ack_timeout to the longer report_timeout — a normal
            # multi-second LLM call is no longer mistaken for a lost request and
            # needlessly resent. In-flight/cached resends don't re-ack: the owner
            # already did, and a resend gets the final directly anyway.
            # Skip entirely for an old worker (see method docstring).
            if rsio_mode:
                await self._emit_running(request_id, kind='ack')
            return await self._produce_response(request_id, data, rsio_mode=rsio_mode)

        try:
            result = await self._guard.run('llm_request', request_id, once)
        except Exception as e:
            error_code = CONTEXT_LENGTH_EXCEEDED_CODE if isinstance(e, ContextLengthExceededError) else None
            self._record_outcome(request_id, error_code)
            logger.warning('rsio.host: request_id=%s failed (code=%s): %s', request_id, error_code, e)
            delivered = await self._emit_final(
                request_id,
                success=False,
                error=str(e),
                error_code=error_code,
                rsio_mode=rsio_mode,
            )
            if delivered:
                await self._guard.discard('llm_request', request_id, delay=_FINAL_REPLAY_GRACE_SECONDS)
            return
        self._record_outcome(request_id, None, result)
        if not ran:
            # Guard replayed an in-flight/cached result: this is a resend of a
            # request already being handled or already completed. The model call
            # was NOT re-executed — we just re-emit the final (a prior final may
            # have been the frame that got lost and triggered this resend).
            logger.debug('rsio.host: dedup replay request_id=%s attempt=%s (no model re-call)', request_id, attempt)
        # Re-emit final on every attempt (cache hit or fresh run): a prior final
        # may have been the message that got lost and triggered the resend.
        delivered = await self._emit_final(request_id, success=True, response=result, rsio_mode=rsio_mode)
        if delivered:
            await self._guard.discard('llm_request', request_id, delay=_FINAL_REPLAY_GRACE_SECONDS)

    def register(self) -> None:
        """Register the LLM request handler with the runtime.

        Registers as the runtime's *default* handler: it answers every request
        that carries no ``channel_id``. Channel-scoped callers bind through
        :class:`LLMRequestRouter` instead.
        """
        if self._handler_registered:
            return

        # NOTE(wallyshi): Requests run in the registration context, which may hold an agent span entered after
        # construction; per-turn callers refresh it through refresh_request_context.
        self._request_context = contextvars.copy_context()
        # The router registry holds routers weakly, so keep the router that owns our listener until unregister.
        self._default_router = LLMRequestRouter.for_runtime(self.runtime)
        self._default_router.bind_default(self)
        self._handler_registered = True
        logger.debug('LLM request handler registered on runtime')

    def unregister(self) -> None:
        """Unregister the LLM request handler."""
        if not self._handler_registered:
            return

        router, self._default_router = self._default_router, None
        if router is not None:
            router.unbind_default(self)
        self._handler_registered = False
        logger.debug('LLM request handler unregistered')


class LLMRequestRouter:
    """Own the single ``llm_request`` listener of a runtime, fan out by channel.

    Host runtimes keep a *list* of listeners per event, so two handlers
    registered for one runtime would each answer every request. The router
    registers once and dispatches every request to the handler bound to its
    ``channel_id``; requests without a channel go to the default handler.

    A request whose channel has no bound handler is rejected rather than
    answered by the default one: routing it to another session's client would
    silently mix trajectories.
    """

    # Weak on both sides: the runtime's listener list and the default handler keep a router alive, so
    # runtime -> router -> handler -> runtime stays an unrooted cycle the GC can collect even when a
    # caller never unbinds.
    _by_runtime: 'weakref.WeakKeyDictionary[Any, weakref.ReferenceType[LLMRequestRouter]]' = weakref.WeakKeyDictionary()

    def __init__(self, runtime: Any) -> None:
        self._runtime_ref = weakref.ref(runtime)
        self._by_channel: dict[str, LLMRequestHandler] = {}
        self._default: LLMRequestHandler | None = None
        self._listening = False

    @classmethod
    def for_runtime(cls, runtime: Any) -> 'LLMRequestRouter':
        """Return the router of *runtime*, creating it on first use."""
        router_ref = cls._by_runtime.get(runtime)
        router = router_ref() if router_ref is not None else None
        if router is None:
            router = cls(runtime)
            cls._by_runtime[runtime] = weakref.ref(router)
        return router

    def get(self, channel_id: str) -> LLMRequestHandler | None:
        """Return the handler currently bound to *channel_id*."""
        return self._by_channel.get(channel_id)

    def bind(self, channel_id: str, handler: LLMRequestHandler) -> None:
        """Route *channel_id* to *handler* until :meth:`unbind`."""
        if not channel_id:
            raise ValueError('channel_id must be a non-empty string')
        owner = self._by_channel.get(channel_id)
        if owner is not None and owner is not handler:
            raise ValueError(f'LLM channel {channel_id!r} is already bound to another handler')
        self._by_channel[channel_id] = handler
        self._listen()

    def unbind(self, channel_id: str, handler: LLMRequestHandler | None = None) -> None:
        """Release *channel_id*; no-op when *handler* is not its owner."""
        owner = self._by_channel.get(channel_id)
        if owner is None:
            return
        if handler is not None and owner is not handler:
            return
        self._by_channel.pop(channel_id, None)
        self._stop_listening_if_idle()

    def bind_default(self, handler: LLMRequestHandler) -> None:
        """Route every unchanneled request to *handler*."""
        if self._default is not None and self._default is not handler:
            raise ValueError('A default LLM handler is already bound to this runtime')
        handler_channel = next((key for key, value in self._by_channel.items() if value is handler), None)
        if handler_channel is not None:
            raise ValueError(f'LLM handler is already bound to channel {handler_channel!r}')
        self._default = handler
        self._listen()

    def unbind_default(self, handler: LLMRequestHandler) -> None:
        """Release the default route; no-op when *handler* is not its owner."""
        if self._default is not handler:
            return
        self._default = None
        self._stop_listening_if_idle()

    def _listen(self) -> None:
        if self._listening:
            return
        runtime = self._runtime_ref()
        if runtime is None:
            raise RuntimeError('Cannot bind an LLM handler after its runtime was released')
        runtime.on('llm_request', cast(Any, self.dispatch))
        self._listening = True

    def _stop_listening_if_idle(self) -> None:
        if not self._listening or self._default is not None or self._by_channel:
            return
        runtime = self._runtime_ref()
        if runtime is not None:
            runtime.off('llm_request', cast(Any, self.dispatch))
        self._listening = False

    async def dispatch(self, data: dict[str, Any]) -> None:
        handler = self._resolve(data)
        if handler is None:
            await self._reject_unbound(data)
            return
        await handler.handle_llm_request(data)

    async def _reject_unbound(self, data: Any) -> None:
        """Fail a routed request immediately instead of making RSIO retry until timeout."""
        if not isinstance(data, dict):
            return
        request_id = data.get('request_id')
        if not request_id:
            return
        rsio = data.get('rsio')
        channel_id = rsio.get('channel_id') if isinstance(rsio, dict) else None
        payload: dict[str, Any] = {
            'request_id': request_id,
            'success': False,
            'error': f'No LLM handler is bound for channel {channel_id!r}',
            'error_code': _UNBOUND_LLM_CHANNEL_CODE,
        }
        if isinstance(rsio, dict):
            payload['rsio'] = {'kind': 'final'}
        runtime = self._runtime_ref()
        if runtime is not None:
            await runtime.emit('llm_response', payload)

    def _resolve(self, data: dict[str, Any]) -> LLMRequestHandler | None:
        if not isinstance(data, dict):
            logger.warning('rsio.host: llm_request payload is not an object, ignoring')
            return None

        rsio = data.get('rsio')
        channel_id = rsio.get('channel_id') if isinstance(rsio, dict) else None
        handler = self._by_channel.get(channel_id) if channel_id else None
        if handler is None and not channel_id:
            handler = self._default
        if handler is None:
            logger.warning('rsio.host: no handler bound for channel=%s; rejecting llm_request', channel_id)
        return handler
