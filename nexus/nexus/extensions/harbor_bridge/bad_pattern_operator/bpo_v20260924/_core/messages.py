"""Frozen dataclasses for messages, tool calls, tool responses, and the
four typed views (TrajectoryView / SFTView / PreferenceView /
CompletionView).

Per the design doc §3, all views are frozen — detectors should not be
able to mutate the sample mid-pipeline. The helper functions
``iter_tool_calls`` and ``pair_tool_calls_with_responses`` are ports of
the ``bpo.base_detector`` helpers, retyped to accept
``Sequence[Message]`` rather than ``List[Dict[str, Any]]`` (per design
review I4 default: "port the helpers in core/messages.py to accept
Sequence[Message]"). The dict-typed legacy variants are not needed in
the spike — only F11 and H1 consume these helpers, both via
``Message`` objects.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

from .tool_registry import ToolRegistryTimeline

from .tool_aliases import (
    canonicalize_tool,
    get_command,
    get_path,
    get_url,
    parse_args_dict,
    shell_file_write_target,
)

Role = Literal["system", "user", "assistant", "tool", "function"]


# ─── Tool call / response ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation produced by an assistant message.

    Mirrors ``bpo.base_detector.ToolCall`` — rename is not a redesign.
    The ``args_raw`` field is the un-parsed ``function.arguments``
    string (used by H1 to test JSON validity); ``args`` is the parsed
    dict (with a ``{"_raw": ...}`` fallback on parse failure).
    """

    turn_index: int
    tool_call_id: Optional[str]
    raw_name: str
    name: str
    args: Dict[str, Any]
    args_raw: str = ""        # original arguments string before parse_args_dict
    command: str = ""
    path: str = ""
    url: str = ""


@dataclass(frozen=True, slots=True)
class ToolResponse:
    turn_index: int
    tool_call_id: Optional[str]
    text: str
    is_error: bool = False
    exit_code: Optional[int] = None
    # Source-derived failure class (see bpo.core.tool_signals). Set by the
    # ingest annotation layer (corpus.tool_outcome) when a catalog signal
    # fires; ``None`` means "unknown / not classified", never "succeeded".
    error_class: Optional[str] = None
    # Source annotation is deliberately three-valued.  Missing annotation is
    # ``unknown`` and must never be read as success.
    outcome: Literal["success", "failure", "unknown"] = "unknown"
    # True only when the raw message carried ``inferred_outcome=failure``.
    # This keeps an ingest-derived ``is_error=True`` distinguishable from a
    # provider/harness-authored flag after conversion to the typed view.
    outcome_inferred: bool = False


# ─── Message ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Message:
    """One conversation element. Compatible with OpenAI messages but
    flattens multimodal blocks: ``content`` is the string form, original
    blocks live in ``raw_content``.
    """

    role: Role
    content: str
    turn_index: int = 0
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""
    tool_response: Optional[ToolResponse] = None
    raw_content: Optional[List[Dict[str, Any]]] = None
    # Per-message generation terminator when the source exports one (for
    # example ``stop`` / ``length`` / ``tool_calls``). Distinct from the
    # record-level TrajectoryView.finish_reason fallback. Kept last so adding
    # the observational field does not shift existing positional arguments.
    finish_reason: str = ""


# ─── Helpers ─────────────────────────────────────────────────────────


# ``<keystrokes …>CMD</keystrokes>`` — the terminal_bench XML protocol's unit
# of action. Each is one shell command typed at the terminal.
_KEYSTROKES_RE = re.compile(r"<keystrokes\b[^>]*>([\s\S]*?)</keystrokes>", re.IGNORECASE)


def _extract_keystroke_commands(content: str) -> List[str]:
    """Pull the shell command out of each ``<keystrokes>…</keystrokes>`` block
    in an assistant message's content (terminal_bench protocol). Trailing
    newline (the literal Enter the model types) is stripped; empty blocks are
    dropped. Returns [] when the content carries no keystrokes."""
    out: List[str] = []
    for m in _KEYSTROKES_RE.finditer(content):
        cmd = m.group(1).strip()
        if cmd:
            out.append(cmd)
    return out


# terminal_bench feeds the result of the keystrokes back in the NEXT user
# message as ``New Terminal Output: …``. That's a tool RESPONSE in disguise —
# converting it to a tool-role Message lets pair_tool_calls_with_responses pair
# it with the synthesized keystroke calls (so edit→run-error rounds, tool
# errors, etc. are visible). These error markers flag a failed command in the
# echoed terminal text.
_TERMINAL_OUTPUT_MARKER = "New Terminal Output"
_TERMINAL_ERROR_RE = re.compile(
    r"(command not found|not recognized|No such file or directory|Traceback|"
    r"\bError\b|\bERROR\b|Exception|FAILED|FAIL\b|Segmentation fault|"
    r"permission denied|cannot |unable to |fatal:|exit code [1-9]|"
    r"returned non-zero|SyntaxError|NameError|TypeError|ModuleNotFoundError)",
)


def _terminal_output_is_error(text: str) -> bool:
    return bool(text) and bool(_TERMINAL_ERROR_RE.search(text[:4000]))


def _msg_text_from_dict(msg: Any) -> str:
    """Flatten OpenAI tool-calling ``message.content`` to a single string.

    Accepts: str → returned as-is; list[{type:text, text:...}] → joined;
    anything else → "". Mirrors ``bpo.base_detector.msg_text``.
    """
    if not isinstance(msg, dict):
        return ""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: List[str] = []
        for x in c:
            if isinstance(x, dict):
                t = x.get("text") or x.get("content")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(x, str):
                parts.append(x)
        return "\n".join(parts)
    return ""


def _raw_blocks(msg: Any) -> Optional[List[Dict[str, Any]]]:
    """If content was a list of blocks, preserve it for raw_content."""
    if not isinstance(msg, dict):
        return None
    c = msg.get("content")
    return list(c) if isinstance(c, list) else None


def reasoning_from_message(msg: Any) -> str:
    """Extract an assistant turn's reasoning text, wherever the backend stashed it.

    Inference backends disagree on where the reasoning/thinking channel lands in
    an exported chat message, so reading only ``reasoning_content`` silently
    drops it for some of them. Observed on the same checkpoint
    (``exp163_step80``) across three serving backends:

    - **sglang** / **baseline** → top-level ``reasoning_content`` (baseline also
      duplicates it into ``provider_specific_fields.reasoning``).
    - **vLLM** → ONLY ``provider_specific_fields.reasoning``; top-level
      ``reasoning_content`` is absent and ``_reasoning_content`` is ``null``.

    Reading only ``reasoning_content`` therefore lost 100% of the reasoning for
    the vLLM export (0 chars vs 172K across 10 trajectories) — invisible to
    every reasoning-dependent detector. This tries each known location in
    precedence order. Mirrors the live-API path's hoist in
    ``bpo.platform.taiji_pull`` so detector semantics match across the
    export-file and live pulls.
    """
    if not isinstance(msg, dict):
        return ""
    psf = msg.get("provider_specific_fields") or {}
    reasoning = (
        msg.get("reasoning_content")                                          # sglang, baseline
        or msg.get("_reasoning_content")                                      # dead slot in practice, harmless
        or (psf.get("reasoning_content") if isinstance(psf, dict) else None)  # some export vintages
        or (psf.get("reasoning") if isinstance(psf, dict) else None)          # vLLM
        or ""
    )
    return reasoning if isinstance(reasoning, str) else ""


def message_from_dict(msg: Dict[str, Any], turn_index: int) -> Message:
    """Build a ``Message`` from a raw OpenAI-style dict.

    Tool calls and tool responses are decoded inline; nothing in the
    raw record is mutated. The same converter is used by both the
    trajectory adapter and the SFT adapter, so the `Message` semantics
    are identical regardless of source format.
    """
    role = msg.get("role") or "user"
    content = _msg_text_from_dict(msg)
    raw = _raw_blocks(msg)
    reasoning = reasoning_from_message(msg)
    raw_finish_reason = msg.get("finish_reason")
    finish_reason = raw_finish_reason if isinstance(raw_finish_reason, str) else ""

    tool_calls: List[ToolCall] = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        raw_name = str(fn.get("name") or tc.get("name") or "")
        args_raw_obj = fn.get("arguments")
        if args_raw_obj is None:
            args_raw_obj = tc.get("arguments")
        # Stringify args_raw for H1; parse_args_dict tolerates non-str.
        if isinstance(args_raw_obj, str):
            args_raw_str = args_raw_obj
        elif args_raw_obj is None:
            args_raw_str = ""
        else:
            # If upstream already parsed args to a dict, JSON-roundtrip
            # so H1 can still apply the "is this a valid JSON string?"
            # test deterministically.
            try:
                import json as _json
                args_raw_str = _json.dumps(args_raw_obj, ensure_ascii=False)
            except (TypeError, ValueError):
                args_raw_str = str(args_raw_obj)
        args = parse_args_dict(args_raw_obj)
        tool_calls.append(ToolCall(
            turn_index=turn_index,
            tool_call_id=tc.get("id"),
            raw_name=raw_name,
            name=canonicalize_tool(raw_name, args),
            args=args,
            args_raw=args_raw_str,
            command=get_command(args),
            path=get_path(args),
            url=get_url(args),
        ))

    # Content-embedded actions: some harnesses (terminal_bench's XML keystrokes
    # protocol) carry NO structured tool_calls — the model's shell commands live
    # in the assistant ``content`` as ``<keystrokes>…</keystrokes>`` inside a
    # ``<commands>`` block. Every tool-call-dependent detector is blind to those
    # (445/445 terminal_bench trajectories have 0 tool_calls). When an assistant
    # message has no structured calls, synthesise shell ToolCalls from any
    # keystrokes so detectors see the real actions. Only fires on the pattern,
    # so ordinary records are untouched.
    if role == "assistant" and not tool_calls and content and "<keystrokes" in content:
        for ks_turn, cmd in enumerate(_extract_keystroke_commands(content)):
            args = {"command": cmd}
            # A keystroke that WRITES a file (cat > f << EOF, sed -i, > f, tee)
            # is the model EDITING a file — surface it as an edit (with the
            # target path) so the edit-family detectors see it, since
            # terminal_bench has no edit tool. Other keystrokes stay shell.
            write_target = shell_file_write_target(cmd)
            if write_target:
                args["path"] = write_target
            tool_calls.append(ToolCall(
                turn_index=turn_index,
                tool_call_id=None,
                raw_name="keystrokes",
                name="edit" if write_target else "shell",
                args=args,
                # ``args_raw`` is the un-parsed ``function.arguments`` — for a
                # real tool_call that's a JSON string. A synthesized keystroke
                # call must honour that contract: serialize the args dict, NOT
                # store the bare shell command. Otherwise
                # ``tool_call_args_parse_fail`` json.loads()-es "apt-get install …"
                # and false-flags every keystroke (7088 FPs on one terminal_bench
                # bucket). It must only see the serialized structured-arguments
                # contract here.
                # ``args``/``command``/``path`` still carry the parsed form.
                args_raw=json.dumps(args, ensure_ascii=False),
                command=cmd,
                path=write_target,
                url=None,
            ))

    tool_response: Optional[ToolResponse] = None
    if role == "tool":
        # is_error / exit_code: read whatever the record carries. Native OpenAI
        # logs rarely set these, but our ingest layer stamps a heuristic
        # success verdict onto tool messages (error / ok / non-deterministic);
        # promoting them here lets every downstream consumer (detectors,
        # corpus read --outline action summary) see per-turn outcome. Absent →
        # defaults (is_error=False, exit_code=None), i.e. "unknown", and callers
        # must not read that as success.
        _is_err = msg.get("is_error")
        _exit = msg.get("exit_code")
        _err_class = msg.get("error_class")
        _raw_inferred_outcome = msg.get("inferred_outcome")
        _outcome = _raw_inferred_outcome
        if _outcome not in {"success", "failure", "unknown"}:
            if _is_err is True or (isinstance(_exit, int) and _exit != 0):
                _outcome = "failure"
            else:
                _outcome = "unknown"
        tool_response = ToolResponse(
            turn_index=turn_index,
            tool_call_id=msg.get("tool_call_id") or msg.get("id"),
            text=content,
            is_error=bool(_is_err) if _is_err is not None else False,
            exit_code=_exit if isinstance(_exit, int) else None,
            error_class=_err_class if isinstance(_err_class, str) else None,
            outcome=_outcome,
            outcome_inferred=_raw_inferred_outcome == "failure",
        )

    return Message(
        role=role,  # type: ignore[arg-type]
        content=content,
        turn_index=turn_index,
        tool_calls=tool_calls,
        reasoning_content=reasoning if isinstance(reasoning, str) else "",
        finish_reason=finish_reason,
        tool_response=tool_response,
        raw_content=raw,
    )


def messages_from_record(record: Dict[str, Any]) -> List[Message]:
    """Pull messages out of a record, normalising both
    ``record["messages"]`` and ``record["request"]["messages"]`` shapes.
    """
    if not isinstance(record, dict):
        return []
    req = record.get("request") if isinstance(record.get("request"), dict) else {}
    raw_msgs = req.get("messages") if isinstance(req, dict) else None
    if not isinstance(raw_msgs, list):
        raw_msgs = record.get("messages") or []
    if not isinstance(raw_msgs, list):
        return []
    out: List[Message] = []
    for i, m in enumerate(raw_msgs):
        if isinstance(m, dict):
            out.append(message_from_dict(m, turn_index=i))

    # terminal_bench round-trip: when an assistant message carried synthesized
    # keystroke tool_calls (raw_name="keystrokes") and the NEXT message is a
    # user "New Terminal Output", re-cast that user message as a tool RESPONSE
    # so pair_tool_calls_with_responses can pair it (enabling edit→run-error
    # rounds, tool-error detection, etc. on this content-only harness). Only
    # fires on the synthesized pattern — ordinary trajectories are untouched.
    for i in range(len(out) - 1):
        cur, nxt = out[i], out[i + 1]
        if (cur.role == "assistant"
                and any(tc.raw_name == "keystrokes" for tc in cur.tool_calls)
                and nxt.role == "user"
                and _TERMINAL_OUTPUT_MARKER in (nxt.content or "")):
            out[i + 1] = Message(
                role="tool",
                content=nxt.content,
                turn_index=nxt.turn_index,
                tool_calls=[],
                reasoning_content="",
                tool_response=ToolResponse(
                    turn_index=nxt.turn_index,
                    tool_call_id=None,
                    text=nxt.content,
                    is_error=_terminal_output_is_error(nxt.content or ""),
                    exit_code=None,
                    error_class=None,
                ),
                raw_content=nxt.raw_content,
                finish_reason=nxt.finish_reason,
            )
    return out


def iter_tool_calls(messages: Sequence[Message]) -> Iterator[ToolCall]:
    """Yield every ``ToolCall`` carried by every assistant ``Message``.

    The Sequence[Message] signature is intentional (per I4 default in
    the design doc): once messages are typed, helpers should take the
    typed shape rather than re-walking dicts.
    """
    for m in messages:
        if m.role != "assistant":
            continue
        for tc in m.tool_calls:
            yield tc


def pair_tool_calls_with_responses(
    messages: Sequence[Message],
) -> List[Tuple[ToolCall, Optional[ToolResponse]]]:
    """Pair each assistant ``ToolCall`` with the matching ``ToolResponse``.

    Algorithm follows ``bpo.base_detector.pair_tool_calls_with_responses``:
    prefer matching by ``tool_call_id``; fall back to "first unused
    response strictly after the call's turn".
    """
    calls = list(iter_tool_calls(messages))
    responses: List[ToolResponse] = []
    for m in messages:
        if m.role == "tool" and m.tool_response is not None:
            responses.append(m.tool_response)

    pairs: List[Tuple[ToolCall, Optional[ToolResponse]]] = []
    used_ids = set()
    for c in calls:
        match: Optional[ToolResponse] = None
        if c.tool_call_id:
            for r in responses:
                if r.tool_call_id == c.tool_call_id and id(r) not in used_ids:
                    match = r
                    used_ids.add(id(r))
                    break
        if match is None:
            for r in responses:
                if (
                    r.tool_call_id is None
                    and r.turn_index > c.turn_index
                    and id(r) not in used_ids
                ):
                    match = r
                    used_ids.add(id(r))
                    break
        pairs.append((c, match))
    return pairs


# ─── Views ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TrajectoryView:
    """Multi-turn agent trajectory: full message + tool conversation.

    Per §3 of the design, ``end_status`` and ``n_tool_calls`` are
    record-level metadata pre-computed by the adapter so detectors
    that consume them (D7 in the full migration) don't have to walk
    the message list again.
    """

    sample_id: str
    messages: List[Message]
    model: str = ""
    tools_available: List[Dict[str, Any]] = field(default_factory=list)
    tool_registry: ToolRegistryTimeline = field(default_factory=ToolRegistryTimeline)
    finish_reason: str = ""
    end_status: str = ""
    n_tool_calls: int = 0
    # Platform/eval-engine failure tags lifted from the record's ``extra`` by
    # the adapter (set by taiji_pull ingest for timeout / recursion-dump that
    # don't survive into ``messages``). Lets INFRA_101 surface them as a hit
    # without re-reading the raw record. Empty/False for normal records.
    infra_failure: bool = False
    infra_failure_reason: str = ""
    infra_failure_detail: str = ""
    infra_failure_evidence: Dict[str, Any] = field(default_factory=dict)
    # Exact evaluator-budget provenance. These remain unset unless the raw
    # record explicitly carries all/part of ``extra.sampling`` and
    # ``extra.eval_detail``; detectors must never substitute message count,
    # pass/fail, or the generic record-level end_status.
    configured_max_iterations: Optional[int] = None
    actual_iterations: Optional[int] = None
    evaluator_exit_status: str = ""
    # Which turn THIS model just produced, as opposed to the conversation it was
    # handed. Only set when the source shape carried the reply in its own field
    # (the online-log shapes), so ``None`` genuinely means "unknown" and must
    # never be back-filled with ``len(messages) - 1``. Kept last so adding it
    # does not shift existing positional arguments.
    generated_turn_index: Optional[int] = None


@dataclass(frozen=True, slots=True)
class SFTView:
    """Supervised fine-tuning sample.

    Wave 3a.5 extension (I5 resolution): the Hunyuan SFT corpus carries
    ``loss_mask`` / ``thinking_loss_mask`` / ``topic`` / ``sft_infos``
    bookkeeping that future SFT detectors (H2 role conformance, the
    eventual loss-mask audit) need. The generic ``sft_messages`` adapter
    leaves these at the defaults; the Hunyuan adapter populates them
    losslessly. Detectors that don't care just ignore the new fields.

    Fields:
        loss_mask          : per-token mask (1 = compute loss, 0 = skip).
                             Same length as the tokenized assistant
                             response. Empty when the source format
                             doesn't expose it.
        thinking_loss_mask : per-token mask for assistant
                             ``reasoning_content``. Empty when absent.
        topic              : Hunyuan taxonomy levels — typically a dict
                             with keys ``L1`` / ``L2`` / ``L3``.
        sft_metadata       : free-form dict of provider/task/language/
                             ori_file etc. — whatever ``sft_infos`` +
                             ``sft_extra_infos`` carry.
    """

    sample_id: str
    messages: List[Message]
    user_prompt: str = ""
    assistant_response: str = ""
    loss_mask: List[int] = field(default_factory=list)
    thinking_loss_mask: List[int] = field(default_factory=list)
    topic: Dict[str, str] = field(default_factory=dict)
    sft_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreferenceView:
    """DPO-style preference pair. Reserved for Wave 3; the spike only
    instantiates this if a detector ever requests it, which it won't."""

    sample_id: str
    prompt_messages: List[Message]
    chosen: List[Message]
    rejected: List[Message]
    chosen_score: Optional[float] = None
    rejected_score: Optional[float] = None


@dataclass(frozen=True, slots=True)
class CompletionView:
    """Free-text completion — no roles, no tool calls. Reserved for Wave 3."""

    sample_id: str
    prompt_text: str
    completion_text: str
    prompt_tokens: Optional[List[int]] = None
    completion_tokens: Optional[List[int]] = None


__all__ = [
    "Role",
    "ToolCall",
    "ToolResponse",
    "Message",
    "TrajectoryView",
    "SFTView",
    "PreferenceView",
    "CompletionView",
    "iter_tool_calls",
    "pair_tool_calls_with_responses",
    "message_from_dict",
    "messages_from_record",
]
