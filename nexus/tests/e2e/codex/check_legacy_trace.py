#!/usr/bin/env python3
"""Validate a real SWE Codex run through the legacy/default Nexus path."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_TOOL_ITEM_TYPES = {
    "collab_tool_call",
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "web_search",
}


class TraceValidationError(RuntimeError):
    """Raised when an E2E artifact violates the legacy trace contract."""


def _codex_item_type(item: dict[str, Any]) -> str | None:
    item_type = item.get("item_type") or item.get("type")
    return item_type if isinstance(item_type, str) else None


def _load_single_record(results_path: Path) -> dict[str, Any]:
    if not results_path.is_file():
        raise TraceValidationError(f"results JSONL does not exist: {results_path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TraceValidationError(f"invalid results JSON on line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise TraceValidationError(f"results line {line_number} is not a JSON object")
        records.append(record)
    if len(records) != 1:
        raise TraceValidationError(f"expected exactly one SWE result, found {len(records)}")
    return records[0]


def _parse_raw_events(agent_output: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    raw_events = agent_output.get("json_events_stdout")
    if not isinstance(raw_events, str) or not raw_events.strip():
        raise TraceValidationError("successful run did not retain raw Codex JSON events")

    events: list[dict[str, Any]] = []
    item_counts: Counter[str] = Counter()
    for line_number, line in enumerate(raw_events.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise TraceValidationError(f"invalid raw Codex JSON event on line {line_number}: {error}") from error
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise TraceValidationError(f"raw Codex event line {line_number} has no string type")
        events.append(event)
        item = event.get("item")
        if isinstance(item, dict) and (item_type := _codex_item_type(item)) is not None:
            item_counts[item_type] += 1

    event_types = [event["type"] for event in events]
    for required_type in ("thread.started", "turn.started", "turn.completed"):
        if required_type not in event_types:
            raise TraceValidationError(f"raw Codex event stream is missing {required_type}")
    if not (
        event_types.index("thread.started")
        < event_types.index("turn.started")
        < len(event_types) - 1 - event_types[::-1].index("turn.completed")
    ):
        raise TraceValidationError("raw Codex thread/turn lifecycle is out of order")
    if item_counts["reasoning"] == 0:
        raise TraceValidationError("raw Codex event stream has no reasoning item")
    if item_counts["agent_message"] == 0:
        raise TraceValidationError("raw Codex event stream has no final agent message")
    if not any(item_counts[item_type] for item_type in _TOOL_ITEM_TYPES):
        raise TraceValidationError("raw Codex event stream has no tool execution item")

    started_items: dict[tuple[str, str], int] = {}
    completed_items: set[tuple[str, str]] = set()
    for event_index, event in enumerate(events):
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        item_type = _codex_item_type(item)
        if not isinstance(item_id, str) or not isinstance(item_type, str):
            continue
        key = (item_id, item_type)
        if event["type"] == "item.started":
            if key in started_items and key not in completed_items:
                raise TraceValidationError(f"raw Codex item {item_id!r} started twice before completion")
            started_items[key] = event_index
        elif event["type"] == "item.completed":
            if key in started_items and started_items[key] >= event_index:
                raise TraceValidationError(f"raw Codex item {item_id!r} completed before it started")
            completed_items.add(key)
    unfinished = sorted(key for key in started_items if key not in completed_items)
    if unfinished:
        raise TraceValidationError(f"raw Codex items never completed: {unfinished}")
    return events, item_counts


def _trajectory_messages(trajectory: Any, index: int) -> list[dict[str, Any]]:
    messages = trajectory.get("messages") if isinstance(trajectory, dict) else trajectory
    if not isinstance(messages, list) or not messages:
        raise TraceValidationError(f"chat trajectory {index} has no messages")
    if not all(isinstance(message, dict) for message in messages):
        raise TraceValidationError(f"chat trajectory {index} contains a non-object message")
    return messages


def _is_spawn_tool_name(name: Any) -> bool:
    return isinstance(name, str) and (name == "spawn_agent" or name.endswith("__spawn_agent"))


def _validate_chat_trajectories(
    processor_results: dict[str, Any],
    *,
    require_multi_agent: bool,
) -> dict[str, Any]:
    if "gateway" in processor_results:
        raise TraceValidationError("legacy run unexpectedly contains a Gateway processor result")
    chat_result = processor_results.get("chat")
    trajectories = chat_result.get("trajectories") if isinstance(chat_result, dict) else None
    if not isinstance(trajectories, list) or not trajectories:
        raise TraceValidationError("chat processor did not dump any trajectories")

    candidates: list[tuple[tuple[bool, bool, int], list[dict[str, Any]], int, int, int]] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        messages = _trajectory_messages(trajectory, trajectory_index)
        announced_calls: set[str] = set()
        resolved_calls: set[str] = set()
        reasoning_messages = 0
        spawn_tool_calls = 0
        for message_index, message in enumerate(messages):
            role = message.get("role")
            if role not in {"assistant", "developer", "system", "tool", "user"}:
                raise TraceValidationError(
                    f"chat trajectory {trajectory_index} message {message_index} has invalid role {role!r}"
                )
            if role == "assistant" and (
                message.get("reasoning_content") or message.get("reasoning_items") or message.get("thinking_blocks")
            ):
                reasoning_messages += 1
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise TraceValidationError(
                    f"chat trajectory {trajectory_index} message {message_index} has invalid tool_calls"
                )
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    raise TraceValidationError(
                        f"chat trajectory {trajectory_index} message {message_index} has a tool call without an id"
                    )
                if tool_call_id in announced_calls:
                    raise TraceValidationError(f"chat trajectory {trajectory_index} repeats tool call {tool_call_id!r}")
                function = tool_call.get("function")
                if isinstance(function, dict) and _is_spawn_tool_name(function.get("name")):
                    spawn_tool_calls += 1
                announced_calls.add(tool_call_id)
            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id not in announced_calls:
                    raise TraceValidationError(
                        f"chat trajectory {trajectory_index} message {message_index} resolves unknown tool call "
                        f"{tool_call_id!r}"
                    )
                if tool_call_id in resolved_calls:
                    raise TraceValidationError(
                        f"chat trajectory {trajectory_index} resolves tool call {tool_call_id!r} more than once"
                    )
                resolved_calls.add(tool_call_id)

        final_message = messages[-1]
        is_closed = final_message.get("role") == "assistant" and not final_message.get("tool_calls")
        candidates.append(
            (
                (not require_multi_agent or spawn_tool_calls > 0, is_closed, len(messages)),
                messages,
                len(announced_calls),
                reasoning_messages,
                spawn_tool_calls,
            )
        )

    _, main_messages, tool_call_count, reasoning_message_count, spawn_tool_call_count = max(
        candidates,
        key=lambda candidate: candidate[0],
    )
    if main_messages[-1].get("role") != "assistant" or main_messages[-1].get("tool_calls"):
        raise TraceValidationError("selected main chat trajectory does not end with a final assistant message")

    announced_calls = {
        call["id"]
        for message in main_messages
        for call in message.get("tool_calls") or []
        if isinstance(call, dict) and isinstance(call.get("id"), str)
    }
    resolved_calls = {message.get("tool_call_id") for message in main_messages if message.get("role") == "tool"}
    unresolved_calls = sorted(announced_calls - resolved_calls)
    if unresolved_calls:
        raise TraceValidationError(f"main chat trajectory has unresolved tool calls: {unresolved_calls}")
    if tool_call_count == 0:
        raise TraceValidationError("main chat trajectory has no tool calls")
    if reasoning_message_count == 0:
        raise TraceValidationError("main chat trajectory has no assistant reasoning payload")
    if require_multi_agent and spawn_tool_call_count == 0:
        raise TraceValidationError("main chat trajectory has no spawn_agent tool call")
    if require_multi_agent and spawn_tool_call_count != 1:
        raise TraceValidationError(
            f"expected exactly one spawn_agent tool call in main chat trajectory, found {spawn_tool_call_count}"
        )
    return {
        "chat_trajectory_count": len(trajectories),
        "main_message_count": len(main_messages),
        "tool_call_count": tool_call_count,
        "reasoning_message_count": reasoning_message_count,
        "spawn_tool_call_count": spawn_tool_call_count,
    }


def _validate_multi_agent_events(
    events: list[dict[str, Any]],
    *,
    require_multi_agent: bool,
) -> dict[str, Any]:
    root_thread_ids = [
        event.get("thread_id")
        for event in events
        if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str)
    ]
    root_thread_id = root_thread_ids[0] if root_thread_ids else None
    started_spawn_ids: set[str] = set()
    completed_spawns: list[dict[str, Any]] = []
    completed_waits: list[dict[str, Any]] = []
    failed_collaboration_items: list[str] = []
    for event in events:
        item = event.get("item")
        if not isinstance(item, dict) or _codex_item_type(item) != "collab_tool_call":
            continue
        item_id = item.get("id")
        if item.get("status") == "failed" and isinstance(item_id, str):
            failed_collaboration_items.append(item_id)
        if event.get("type") == "item.started" and item.get("tool") == "spawn_agent":
            if isinstance(item_id, str):
                started_spawn_ids.add(item_id)
        if event.get("type") != "item.completed":
            continue
        if item.get("tool") == "spawn_agent" and item.get("status") == "completed":
            completed_spawns.append(item)
        elif item.get("tool") == "wait" and item.get("status") == "completed":
            completed_waits.append(item)

    child_thread_ids: list[str] = []
    for spawn in completed_spawns:
        item_id = spawn.get("id")
        if item_id not in started_spawn_ids:
            raise TraceValidationError(f"completed spawn_agent item {item_id!r} has no matching started event")
        if root_thread_id is not None and spawn.get("sender_thread_id") != root_thread_id:
            raise TraceValidationError(
                f"spawn_agent sender {spawn.get('sender_thread_id')!r} does not match root thread {root_thread_id!r}"
            )
        if not isinstance(spawn.get("prompt"), str) or not spawn["prompt"]:
            raise TraceValidationError(f"spawn_agent item {item_id!r} did not retain its child prompt")
        receiver_thread_ids = spawn.get("receiver_thread_ids")
        if not isinstance(receiver_thread_ids, list) or not receiver_thread_ids:
            raise TraceValidationError(f"spawn_agent item {item_id!r} did not retain a child thread id")
        child_thread_ids.extend(str(thread_id) for thread_id in receiver_thread_ids if thread_id is not None)

    if failed_collaboration_items:
        raise TraceValidationError(f"collaboration items failed: {sorted(failed_collaboration_items)}")
    if require_multi_agent and len(completed_spawns) != 1:
        raise TraceValidationError(f"expected exactly one completed spawn_agent, found {len(completed_spawns)}")
    unique_child_ids = sorted(set(child_thread_ids))
    if require_multi_agent and len(unique_child_ids) != 1:
        raise TraceValidationError(f"expected exactly one child thread, found {unique_child_ids}")

    completed_wait_child_ids: set[str] = set()
    for wait in completed_waits:
        agents_states = wait.get("agents_states")
        if not isinstance(agents_states, dict):
            continue
        for child_thread_id, state in agents_states.items():
            if (
                child_thread_id in unique_child_ids
                and isinstance(state, dict)
                and state.get("status") == "completed"
                and isinstance(state.get("message"), str)
                and state["message"]
            ):
                completed_wait_child_ids.add(child_thread_id)
    if require_multi_agent and completed_wait_child_ids != set(unique_child_ids):
        missing_child_ids = sorted(set(unique_child_ids) - completed_wait_child_ids)
        raise TraceValidationError(f"spawned child threads have no completed wait result: {missing_child_ids}")
    return {
        "root_thread_id": root_thread_id,
        "subagent_spawn_count": len(completed_spawns),
        "subagent_child_ids": unique_child_ids,
        "subagent_completed_wait_count": len(completed_wait_child_ids),
    }


def _load_trace_rows(database_path: Path, run_id: str) -> list[dict[str, Any]]:
    if not database_path.is_file():
        raise TraceValidationError(f"batch trace database does not exist: {database_path}")
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='run_trace_events'"
        ).fetchone()
        if table is None:
            raise TraceValidationError("batch database has no run_trace_events table")
        rows = connection.execute(
            """
            SELECT event_type, time_unix_nano, span_id, parent_span_id, event_data
            FROM run_trace_events
            WHERE run_id = ?
            ORDER BY time_unix_nano, id
            """,
            (run_id,),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise TraceValidationError(f"batch database has no trace events for run {run_id!r}")

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            event_data = json.loads(row["event_data"])
        except json.JSONDecodeError as error:
            raise TraceValidationError(f"batch trace row has invalid event_data: {error}") from error
        parsed_rows.append({**dict(row), "event_data": event_data})
    return parsed_rows


def _validate_trace_spans(
    database_path: Path,
    run_id: str,
    *,
    require_compaction: bool,
) -> dict[str, Any]:
    rows = _load_trace_rows(database_path, run_id)
    starts = {
        row["span_id"]: row for row in rows if row["event_type"] == "start" and isinstance(row["event_data"], dict)
    }
    ended_span_ids = {row["span_id"] for row in rows if row["event_type"] == "end"}
    span_end_times = {row["span_id"]: row["time_unix_nano"] for row in rows if row["event_type"] == "end"}
    codex_spans = [row for row in starts.values() if row["event_data"].get("name") == "agent.codex"]
    if len(codex_spans) != 1:
        raise TraceValidationError(f"expected exactly one agent.codex span, found {len(codex_spans)}")
    codex_span_id = codex_spans[0]["span_id"]
    if codex_span_id not in ended_span_ids:
        raise TraceValidationError("agent.codex span did not end")

    completion_spans = [row for row in starts.values() if row["event_data"].get("name") == "openai_completion"]
    if not completion_spans:
        raise TraceValidationError("batch trace contains no openai_completion span")
    unfinished_completions = sorted(row["span_id"] for row in completion_spans if row["span_id"] not in ended_span_ids)
    if unfinished_completions:
        raise TraceValidationError(f"openai_completion spans did not end: {unfinished_completions}")

    gateway_span_names = sorted(
        {
            str(row["event_data"].get("name"))
            for row in starts.values()
            if "gateway" in str(row["event_data"].get("name", "")).lower()
        }
    )
    if gateway_span_names:
        raise TraceValidationError(f"legacy trace unexpectedly contains Gateway spans: {gateway_span_names}")

    termination_reason = None
    for row in rows:
        if row["span_id"] != codex_span_id or row["event_type"] != "update":
            continue
        attributes = row["event_data"].get("attributes")
        if isinstance(attributes, dict) and attributes.get("hytuner.termination_reason"):
            termination_reason = attributes["hytuner.termination_reason"]
    if termination_reason != "ENV_DONE":
        raise TraceValidationError(f"agent.codex termination reason is {termination_reason!r}, expected 'ENV_DONE'")

    completed_compactions: list[tuple[dict[str, Any], int]] = []
    for compaction_start in starts.values():
        data = compaction_start["event_data"]
        attributes = data.get("attributes") if isinstance(data.get("attributes"), dict) else {}
        if (
            data.get("name") != "agent.compaction"
            or attributes.get("type") != "agent_compaction"
            or compaction_start["parent_span_id"] != codex_span_id
        ):
            continue
        span_id = compaction_start["span_id"]
        compacted = any(
            row["span_id"] == span_id
            and row["event_type"] == "update"
            and isinstance(row["event_data"].get("attributes"), dict)
            and row["event_data"]["attributes"].get("compacted") is True
            for row in rows
        )
        has_completion_child = any(
            row["parent_span_id"] == span_id and row["span_id"] in ended_span_ids for row in completion_spans
        )
        if compacted and span_id in ended_span_ids and has_completion_child:
            completed_compactions.append((attributes, span_end_times[span_id]))

    if require_compaction and not completed_compactions:
        raise TraceValidationError("forced-compaction run has no completed agent.compaction with an LLM child")
    if require_compaction:
        latest_compaction_end = max(end_time for _, end_time in completed_compactions)
        has_later_conversation_completion = any(
            completion["parent_span_id"] == codex_span_id
            and completion["span_id"] in ended_span_ids
            and completion["time_unix_nano"] > latest_compaction_end
            for completion in completion_spans
        )
        if not has_later_conversation_completion:
            raise TraceValidationError("completed compaction has no later conversation completion")
    return {
        "trace_event_count": len(rows),
        "openai_completion_count": len(completion_spans),
        "compaction_count": len(completed_compactions),
        "compaction_detectors": [attributes.get("detector") for attributes, _ in completed_compactions],
        "termination_reason": termination_reason,
    }


def validate_trace(
    results_path: str | Path,
    database_path: str | Path,
    *,
    require_compaction: bool,
    require_multi_agent: bool = False,
) -> dict[str, Any]:
    """Validate JSONL, raw Codex events, chat messages, and batch spans."""
    results_path = Path(results_path)
    database_path = Path(database_path)
    record = _load_single_record(results_path)
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise TraceValidationError("SWE result has no run_id")
    if record.get("status") != "success":
        raise TraceValidationError(f"SWE run {run_id!r} has status {record.get('status')!r}: {record.get('error')}")
    agent_output = record.get("agent_output")
    if not isinstance(agent_output, dict) or agent_output.get("success") is not True:
        raise TraceValidationError(f"Codex agent output for {run_id!r} is not successful")
    processor_results = record.get("processor_results")
    if not isinstance(processor_results, dict):
        raise TraceValidationError("SWE result has no processor_results object")

    raw_events, raw_item_counts = _parse_raw_events(agent_output)
    multi_agent_summary = _validate_multi_agent_events(raw_events, require_multi_agent=require_multi_agent)
    chat_summary = _validate_chat_trajectories(processor_results, require_multi_agent=require_multi_agent)
    if require_multi_agent and chat_summary["spawn_tool_call_count"] != multi_agent_summary["subagent_spawn_count"]:
        raise TraceValidationError(
            "raw/chat spawn_agent count mismatch: "
            f"raw={multi_agent_summary['subagent_spawn_count']} chat={chat_summary['spawn_tool_call_count']}"
        )
    span_summary = _validate_trace_spans(database_path, run_id, require_compaction=require_compaction)
    return {
        "run_id": run_id,
        "status": record["status"],
        "raw_event_count": len(raw_events),
        "raw_event_counts": dict(sorted(raw_item_counts.items())),
        **multi_agent_summary,
        **chat_summary,
        **span_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", type=Path)
    parser.add_argument("database_path", type=Path)
    parser.add_argument("--require-compaction", action="store_true")
    parser.add_argument("--require-multi-agent", action="store_true")
    args = parser.parse_args()
    try:
        summary = validate_trace(
            args.results_path,
            args.database_path,
            require_compaction=args.require_compaction,
            require_multi_agent=args.require_multi_agent,
        )
    except TraceValidationError as error:
        print(f"trace validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
