"""Merge the ChatTracker snapshot with the handed-in trajectory into one detection view.

Why this exists
---------------
Detection wants every branch the Agent ran, because these rules are **not monotone in
their input**: ``ancestor_commits`` earns its exemption whitelist *from the trajectory*,
so dropping the branch that ran a ``git log`` turns the read it excused into a commit of
unknown provenance. Handing the detector less makes it report more, which is why the
snapshot -- the only source that carries a subagent's branches -- has to be read.

But the snapshot is the *wider* source, not the *more complete* one.
``HarborAgentTrajectoryTracker`` rebuilds the conversation from LLM request/response
pairs, so a tool **result** only materializes once it appears in the NEXT request. A run
that stops on a turn limit therefore leaves its last tool call answered in the handed-in
trajectory and unanswered in the snapshot -- and an unanswered ``curl`` reads as a fetch
nobody can show was blocked. Reproduced with a real tracker: a ``curl`` whose observation
plainly said ``Could not resolve host`` was clean on the full trajectory and became
``online_answer_fetch`` once the snapshot cut off at that response.

So "the snapshot is non-empty" is not "the snapshot is more complete", and replacing one
list with the other loses whichever half the chosen source was missing. This module takes
both: the snapshot's branch coverage AND every observation either source recorded, matched
by tool-call identity rather than by position, because position does not survive a merge
that interleaves branches.

The property it maintains: **the merged view is never poorer in tool activity than either
input**. That is what the non-monotonicity argument above needs -- a rule that reports more
when handed less must never be handed less than what was recorded. The property is enforced
rather than argued: the merge counts the calls and answered calls it is about to return, and
if that count regressed against either input it returns the richer input as-is instead. So a
shape this module guesses wrong about costs the merge, never an observation, and never the
snapshot-only view the previous code handed the detectors.

Strictly additive, like the other bridges here: it owns no detection logic, adds no channel,
and with either side absent it returns the other unchanged.
"""

from __future__ import annotations

import json
import logging

from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from nexus.extensions.harbor_bridge.anti_hack.strategies.anthropic_blocks import (
    has_tool_activity,
    normalize_anthropic_block_messages,
)
from nexus.extensions.harbor_bridge.anti_hack.text_parse import normalize_text_parse_messages


logger = logging.getLogger(__name__)


# The two encodings a trajectory may use for tool activity. ``tool_calls`` is the
# OpenAI-chat shape (in-process Agents, code_mt); ``blocks`` is the Anthropic content-block
# shape (HyCode). An observation synthesized below is written in the shape of the message
# whose call it answers, so the downstream normalize chain still recognizes the list.
_TOOL_BLOCK_TYPES = ('tool_use', 'tool_result')


@dataclass
class _Pairing:
    """Where each tool call sits in one message list, and what answered it.

    ``calls_by_message`` maps an assistant's index to its ``(id, ordinal)`` calls in issue
    order; ``result_messages_by_call_message`` maps the same index to the indices of the
    messages that carried its observations, so a turn can be re-emitted as a unit. The two
    observation maps are keyed by whichever handle could identify the call: ids are
    authoritative when present, and the RL path mints none (HyTuner never goes through an
    OpenAI server), so the ordinal is the only fallback left there.
    """

    calls_by_message: dict[int, list[tuple[str, str]]] = field(default_factory=dict)
    result_messages_by_call_message: dict[int, list[int]] = field(default_factory=dict)
    by_id: dict[str, Any] = field(default_factory=dict)
    by_ordinal: dict[str, Any] = field(default_factory=dict)
    ordinal_by_id: dict[str, str] = field(default_factory=dict)
    id_by_ordinal: dict[str, str] = field(default_factory=dict)

    def calls(self) -> list[tuple[int, str, str]]:
        """Every call as ``(message index, id, ordinal)``, in message order."""
        return [
            (index, call_id, ordinal)
            for index in sorted(self.calls_by_message)
            for call_id, ordinal in self.calls_by_message[index]
        ]


def _signature(name: Any, arguments: Any) -> str:
    """A call's own identity: its tool name plus its normalized arguments.

    Arguments arrive as a JSON string in the OpenAI shape and as a dict in the Anthropic
    one, so a string is parsed back before being re-dumped with sorted keys. Without that
    the same call signs differently in the two shapes and only the id can match it.
    """
    value = arguments
    if isinstance(value, str):
        with suppress(json.JSONDecodeError, TypeError):
            value = json.loads(value)
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    return f'{name or ""}\x00{text}'


def _tool_calls(message: dict[str, Any]) -> list[tuple[str, str]]:
    """``(id, signature)`` for every tool call one message issues, in either encoding."""
    calls: list[tuple[str, str]] = []
    for call in message.get('tool_calls') or []:
        if not isinstance(call, dict):
            continue
        function = call.get('function')
        function = function if isinstance(function, dict) else {}
        calls.append((str(call.get('id') or ''), _signature(function.get('name'), function.get('arguments'))))
    content = message.get('content')
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_use':
                calls.append((str(block.get('id') or ''), _signature(block.get('name'), block.get('input'))))
    return calls


def _tool_results(message: dict[str, Any]) -> list[tuple[str, Any]]:
    """``(answered call id, observation)`` for every result one message carries.

    The observation is kept verbatim rather than flattened to text: the normalize bridges
    downstream already accept every shape a provider writes, and rewriting it here would
    change what they see.
    """
    if message.get('role') == 'tool':
        return [(str(message.get('tool_call_id') or ''), message.get('content'))]
    content = message.get('content')
    if not isinstance(content, list):
        return []
    return [
        (str(block.get('tool_use_id') or ''), block.get('content'))
        for block in content
        if isinstance(block, dict) and block.get('type') == 'tool_result'
    ]


def _pair_tools(messages: list[dict[str, Any]]) -> _Pairing:
    """Index every tool call in a message list together with the result that answered it.

    Pending calls are dropped at any message carrying neither a call nor a result, so a turn
    whose observation never arrived cannot shift the pairings of the turns after it. That is
    the same rule ``parse_nexus_messages`` applies, deliberately: a merge that paired
    differently from the parser would backfill an observation onto the wrong call.
    """
    pairing = _Pairing()
    occurrences: Counter[str] = Counter()
    pending: list[tuple[str, str]] = []
    pending_owner = -1
    for index, message in enumerate(messages):
        own: list[tuple[str, str]] = []
        for call_id, signature in _tool_calls(message):
            occurrences[signature] += 1
            # The nth identical call is a different call. Without the ordinal a repeated
            # command collapses to one key and every repeat reads the first observation.
            ordinal = f'{signature}#{occurrences[signature]}'
            own.append((call_id, ordinal))
            pairing.id_by_ordinal[ordinal] = call_id
            if call_id:
                pairing.ordinal_by_id[call_id] = ordinal
        results = _tool_results(message)
        if own:
            pairing.calls_by_message[index] = own
            pending = list(own)
            pending_owner = index
        elif not results:
            pending = []
            pending_owner = -1
        if results and pending_owner >= 0 and not own:
            pairing.result_messages_by_call_message.setdefault(pending_owner, []).append(index)
        for call_id, observation in results:
            if call_id:
                pairing.by_id.setdefault(call_id, observation)
            if pending:
                _pending_id, ordinal = pending.pop(0)
                pairing.by_ordinal.setdefault(ordinal, observation)
    return pairing


def _is_blank(observation: Any) -> bool:
    """Whether an observation says nothing, which a missing one and a placeholder both do."""
    if observation is None:
        return True
    if isinstance(observation, str):
        return not observation.strip()
    if isinstance(observation, (list, dict, tuple)):
        return not observation
    return False


def _observation_for(pairing: _Pairing, call_id: str, ordinal: str) -> Any:
    """The observation a pairing holds for one call: id first, then the ordinal."""
    if call_id:
        if call_id in pairing.by_id:
            return pairing.by_id[call_id]
        matching_ordinal = pairing.ordinal_by_id.get(call_id)
        if matching_ordinal is not None:
            return pairing.by_ordinal.get(matching_ordinal)
        if pairing.id_by_ordinal.get(ordinal):
            return None
    return pairing.by_ordinal.get(ordinal)


def _has_call(pairing: _Pairing, call_id: str, ordinal: str) -> bool:
    """Whether pairing contains this call, using ordinal only when an ID is absent."""
    if call_id and call_id in pairing.ordinal_by_id:
        return True
    paired_id = pairing.id_by_ordinal.get(ordinal)
    if paired_id is None:
        return False
    return not (call_id and paired_id)


def _observation_message(call_message: dict[str, Any], call_id: str, observation: Any) -> dict[str, Any]:
    """One observation, written in the shape of the call it answers.

    Shape matters because the normalize chain in the strategy is all-or-nothing: an
    Anthropic-block list that suddenly contains an OpenAI ``role='tool'`` message still
    converts, but the converter has no field to put ``tool_call_id`` in, so the pairing this
    function exists to restore is lost again on the way out.
    """
    if call_message.get('tool_calls') or not isinstance(call_message.get('content'), list):
        return {'role': 'tool', 'tool_call_id': call_id, 'content': observation}
    return {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call_id, 'content': observation}]}


def _without_tool_results(message: dict[str, Any]) -> dict[str, Any] | None:
    """The same message with its ``tool_result`` blocks removed, or None if nothing is left.

    A rewritten turn re-emits its observations in call order, which means the original result
    messages are replaced rather than kept. In the Anthropic shape those messages are
    ``role='user'`` turns that may also carry text the harness wrote back, and that text is
    what the prose-reading monitors see, so it is carried over separately instead of dropped.
    """
    content = message.get('content')
    if not isinstance(content, list):
        return None
    kept = [block for block in content if not (isinstance(block, dict) and block.get('type') == 'tool_result')]
    if not kept:
        return None
    return {**message, 'content': kept}


def _tool_shape(messages: list[dict[str, Any]]) -> str:
    """Which tool encoding a message list uses, or ``''`` when it shows no tool activity."""
    for message in messages:
        if message.get('tool_calls') or message.get('role') == 'tool':
            return 'tool_calls'
        content = message.get('content')
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get('type') in _TOOL_BLOCK_TYPES for block in content
        ):
            return 'blocks'
    return ''


def _turn_messages(messages: list[dict[str, Any]], pairing: _Pairing, index: int, source: _Pairing) -> list[dict]:
    """One assistant turn re-emitted with an observation for each of its calls, in call order.

    The whole observation run is rebuilt rather than patched, because these lists are read
    positionally downstream when ids are absent: inserting a recovered observation next to
    the ones already there would pair it with whichever call happens to sit at that offset.
    Rebuilding in call order is the only placement that survives either pairing rule.
    """
    call_message = messages[index]
    rebuilt: list[dict[str, Any]] = [call_message]
    for call_id, ordinal in pairing.calls_by_message.get(index) or []:
        observation = _observation_for(pairing, call_id, ordinal)
        if _is_blank(observation):
            observation = _observation_for(source, call_id, ordinal)
        if not _is_blank(observation):
            rebuilt.append(_observation_message(call_message, call_id, observation))
    for result_index in pairing.result_messages_by_call_message.get(index) or []:
        remainder = _without_tool_results(messages[result_index])
        if remainder is not None:
            rebuilt.append(remainder)
    return rebuilt


def _needs_backfill(pairing: _Pairing, index: int, source: _Pairing) -> bool:
    """Whether any call in one turn is unanswered here but answered in the other source.

    This is the reviewer's case: the snapshot holds the call, the handed-in trajectory holds
    its result, and only together do they say whether the fetch returned anything.
    """
    for call_id, ordinal in pairing.calls_by_message.get(index) or []:
        if not _is_blank(_observation_for(pairing, call_id, ordinal)):
            continue
        if not _is_blank(_observation_for(source, call_id, ordinal)):
            return True
    return False


def _unrecorded_turns(
    base: list[dict[str, Any]],
    base_pairing: _Pairing,
    snapshot: list[dict[str, Any]],
    snapshot_pairing: _Pairing,
) -> list[dict[str, Any]]:
    """Base turns whose calls the snapshot never recorded at all, with their observations.

    A turn is carried over only when *none* of its calls is known to the snapshot;
    re-emitting a partially known turn would duplicate the calls the snapshot already has.

    Both lists are in one encoding by the time this runs (see ``_unify``), so a carried turn
    can be appended verbatim without handing the normalize chain downstream a mixed list.
    """
    if not base_pairing.calls_by_message or _tool_shape(base) != _tool_shape(snapshot):
        return []

    carried: list[dict[str, Any]] = []
    for index in sorted(base_pairing.calls_by_message):
        calls = base_pairing.calls_by_message[index]
        if any(_has_call(snapshot_pairing, call_id, ordinal) for call_id, ordinal in calls):
            continue
        carried.extend(_turn_messages(base, base_pairing, index, snapshot_pairing))
    return carried


def _normalize_text_parse_for_merge(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose TEXT_PARSE calls while keeping their locally minted IDs non-authoritative."""
    normalized = normalize_text_parse_messages(messages)
    if normalized is None:
        return messages

    rewritten: list[dict[str, Any]] = []
    for message in normalized:
        item = dict(message)
        if item.get('tool_calls'):
            item['tool_calls'] = [{**call, 'id': ''} for call in item['tool_calls'] if isinstance(call, dict)]
        if item.get('role') == 'tool':
            item['tool_call_id'] = ''
        rewritten.append(item)
    return rewritten


def _unify(
    base: list[dict[str, Any]],
    snapshot: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Put both lists in one tool encoding, so a turn can cross from one to the other.

    HyCode is the case that needs this. Its handed-in trajectory is parsed from Claude Code's
    stdout and keeps the Anthropic ``tool_use``/``tool_result`` blocks, but the same run
    reaches the LLM through the anthropic proxy, which rewrites those blocks into OpenAI
    ``tool_calls`` *before* the ``openai_completion`` span the tracker listens on. So the two
    sources describe one conversation in two encodings, and a merge that compares shapes sees
    a mismatch on every HyCode run.

    Appending across encodings is not an option: the strategy's normalize chain is
    all-or-nothing, and a list holding both shapes makes each bridge decline it, which loses
    the whole trajectory rather than one turn. Converting is, because the block bridge keeps
    each block's own ``id`` as the ``tool_call_id`` -- the same id the proxy preserves -- so
    pairing by identity still works across the seam.

    TEXT_PARSE sides are normalized first, with their locally minted IDs cleared so they
    remain ordinal fallbacks rather than false cross-source identities. After that, only a
    block side is rewritten, and only when the other side is already in the ``tool_calls``
    shape this module's output is read in. Returns inputs untouched when no bridge applies.
    """
    base = _normalize_text_parse_for_merge(base)
    snapshot = _normalize_text_parse_for_merge(snapshot)
    base_shape = _tool_shape(base)
    snapshot_shape = _tool_shape(snapshot)
    if not base_shape or not snapshot_shape or base_shape == snapshot_shape:
        return base, snapshot
    if base_shape == 'blocks' and snapshot_shape == 'tool_calls':
        return normalize_anthropic_block_messages(base) or base, snapshot
    if snapshot_shape == 'blocks' and base_shape == 'tool_calls':
        return base, normalize_anthropic_block_messages(snapshot) or snapshot
    return base, snapshot


def _activity(messages: list[dict[str, Any]]) -> tuple[int, int]:
    """How much tool activity a list shows, as ``(calls, non-blank observations)``.

    This is the quantity the merge must never reduce, counted the way the detectors read it:
    a call they can scan, and an observation that can clear it.
    """
    calls = 0
    observations = 0
    for message in messages:
        calls += len(_tool_calls(message))
        observations += sum(1 for _call_id, value in _tool_results(message) if not _is_blank(value))
    return calls, observations


def merge_trajectory_messages(
    base: list[dict[str, Any]] | None,
    snapshot: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """The snapshot's branches plus every observation the handed-in trajectory recorded.

    ``base`` is the conversation handed to the strategy (authoritative on observations, one
    branch only); ``snapshot`` is the flattened ChatTracker view (every branch, but a result
    may not have landed yet). Either side absent returns the other unchanged, so a run without
    the processor registered behaves exactly as it did; with a processor the result is a
    superset of both sides, and where it cannot be, the richer side is returned untouched.
    """
    base_messages = [message for message in (base or []) if isinstance(message, dict)]
    snapshot_messages = [message for message in (snapshot or []) if isinstance(message, dict)]
    if not snapshot_messages:
        return base_messages
    if not base_messages:
        return snapshot_messages
    if has_tool_activity(base_messages) and not has_tool_activity(snapshot_messages):
        # A snapshot that recorded no tool action is not a detection view at all: these rules
        # read commands, so adopting it would trade every observation for nothing. This is the
        # reviewer's case in the limit, a tracker holding only the prompts. The floor below
        # reaches the same verdict by counting -- this says it up front because it is the case
        # the review was filed about, and because "no tool activity" is recognized in shapes the
        # counter does not itemize. Nothing detectable is given up: the snapshot dropped here is,
        # by the test just applied, a snapshot with no tool activity in it to lose.
        return base_messages

    # The originals are kept aside. Everything below works on the unified pair, but a merge
    # that ends up declining itself must hand back what it was given, not a re-encoded copy of
    # it: converting an encoding is only worth doing in exchange for a turn that crosses
    # between the two, and with no such turn the conversion is a rewrite of the detection view
    # for nothing.
    base_source, snapshot_source = base_messages, snapshot_messages
    base_messages, snapshot_messages = _unify(base_messages, snapshot_messages)
    base_pairing = _pair_tools(base_messages)
    snapshot_pairing = _pair_tools(snapshot_messages)

    rewritten = {
        index: _turn_messages(snapshot_messages, snapshot_pairing, index, base_pairing)
        for index in snapshot_pairing.calls_by_message
        if _needs_backfill(snapshot_pairing, index, base_pairing)
    }
    carried = _unrecorded_turns(base_messages, base_pairing, snapshot_messages, snapshot_pairing)
    if not rewritten and not carried:
        # Nothing crossed. The snapshot already answered every call it recorded and the
        # handed-in trajectory held no turn it had missed, so there is no merge to make and the
        # snapshot goes back untouched -- the same object the previous code handed the
        # detectors. This is the common case for every single-branch Agent, and it is what makes
        # the new path inert unless it has something to contribute.
        return snapshot_source

    replaced = {
        result_index
        for index in rewritten
        for result_index in snapshot_pairing.result_messages_by_call_message.get(index) or []
    }

    merged: list[dict[str, Any]] = []
    for index, message in enumerate(snapshot_messages):
        if index in rewritten:
            merged.extend(rewritten[index])
        elif index not in replaced:
            merged.append(message)
    merged.extend(carried)

    # The floor. Everything above is a best effort to keep both sources' activity, and each
    # step is guarded, but the guards reason about shapes and ids that a future provider is
    # free to write differently. So the invariant is also *checked*, not merely argued: the
    # merge counts what it is about to hand the detectors and abandons itself if that count
    # regressed. The cost of the floor is a merge declined; the cost of not having it is a rule
    # that reports more because it was handed less, which is the failure this module exists to
    # prevent.
    #
    # It is measured against BOTH inputs, and falls back to whichever of them is richer, which
    # is what keeps this drop-in for either caller. Against the handed-in trajectory it is the
    # non-monotonicity guarantee. Against the snapshot it is backward compatibility: returning
    # the snapshot wholesale is what the previous code did whenever a tracker was registered,
    # so a run where the merge declines and the snapshot is the richer source still sees the
    # behaviour it saw before this module existed -- never a view poorer than that. Only a
    # merge that beats both replaces either. The source handed back is the original, not the
    # unified copy, for the same reason the early return above hands back one: a declined merge
    # should leave no trace, and re-encoding bought nothing once no turn crossed.
    base_activity, snapshot_activity = _activity(base_messages), _activity(snapshot_messages)
    floor = max(base_activity, snapshot_activity)
    if _activity(merged) < floor:
        richer = base_activity >= snapshot_activity
        logger.debug(
            'Trajectory merge would lose tool activity (%s < %s); keeping the richer source unmerged (%s)',
            _activity(merged),
            floor,
            'trajectory' if richer else 'snapshot',
        )
        return base_source if richer else snapshot_source
    return merged


__all__ = ['merge_trajectory_messages']
