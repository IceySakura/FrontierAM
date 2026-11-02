#!/usr/bin/env python3
"""Validate a real SWE Codex run and its canonical Nexus Gateway trajectory."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from pathlib import Path
from typing import Any

from nexus.gateway.trajectory import MAIN_CHAIN_ID, Trajectory


_LEGACY_CHECKER_PATH = Path(__file__).with_name("check_legacy_trace.py")
_LEGACY_SPEC = importlib.util.spec_from_file_location("_codex_swe_legacy_trace", _LEGACY_CHECKER_PATH)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise RuntimeError(f"cannot load legacy trace checks from {_LEGACY_CHECKER_PATH}")
_legacy_checker = importlib.util.module_from_spec(_LEGACY_SPEC)
_LEGACY_SPEC.loader.exec_module(_legacy_checker)

TraceValidationError = _legacy_checker.TraceValidationError
_load_single_record = _legacy_checker._load_single_record
_parse_raw_events = _legacy_checker._parse_raw_events
_validate_chat_trajectories = _legacy_checker._validate_chat_trajectories
_validate_multi_agent_events = _legacy_checker._validate_multi_agent_events
_validate_trace_spans = _legacy_checker._validate_trace_spans

_REGISTERED_TOOL_NAME_PREFIXES = ("multi_agent_v1__",)


def _load_gateway_trajectory(processor_results: dict[str, Any], run_id: str) -> Trajectory:
    dumped = processor_results.get("gateway")
    if not isinstance(dumped, dict):
        raise TraceValidationError("Gateway processor did not dump the canonical trajectory")
    try:
        trajectory = Trajectory.from_dump(dumped)
    except (KeyError, TypeError, ValueError) as error:
        raise TraceValidationError(f"Gateway trajectory cannot be deserialized: {error}") from error
    if trajectory.dump() != dumped:
        raise TraceValidationError("Gateway trajectory changed during dump/from_dump round-trip")
    if trajectory.session_id != run_id:
        raise TraceValidationError(f"Gateway session_id {trajectory.session_id!r} does not match SWE run_id {run_id!r}")
    if trajectory.state != "completed":
        raise TraceValidationError(f"Gateway trajectory state is {trajectory.state!r}, expected 'completed'")
    if not trajectory.termination_reason:
        raise TraceValidationError("Gateway trajectory has no termination_reason")
    if MAIN_CHAIN_ID not in trajectory.chains:
        raise TraceValidationError("Gateway trajectory has no main chain")
    return trajectory


def _validate_main_messages(trajectory: Trajectory) -> dict[str, int]:
    messages = trajectory.to_messages(MAIN_CHAIN_ID)
    if not messages:
        raise TraceValidationError("Gateway main chain has no messages")

    announced_calls: set[str] = set()
    resolved_calls: set[str] = set()
    reasoning_message_count = 0
    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"assistant", "developer", "system", "tool", "user"}:
            raise TraceValidationError(f"Gateway main message {message_index} has invalid role {role!r}")
        if role == "assistant" and (
            message.get("reasoning_content") or message.get("reasoning_items") or message.get("thinking_blocks")
        ):
            reasoning_message_count += 1
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise TraceValidationError(f"Gateway main message {message_index} has invalid tool_calls")
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise TraceValidationError(f"Gateway main message {message_index} has a tool call without an id")
            if tool_call_id in announced_calls:
                raise TraceValidationError(f"Gateway main chain repeats tool call {tool_call_id!r}")
            function = tool_call.get("function")
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if not isinstance(arguments, str):
                raise TraceValidationError(
                    f"Gateway main tool call {tool_call_id!r} did not preserve its JSON argument string"
                )
            announced_calls.add(tool_call_id)
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id not in announced_calls:
                raise TraceValidationError(
                    f"Gateway main message {message_index} resolves unknown tool call {tool_call_id!r}"
                )
            if tool_call_id in resolved_calls:
                raise TraceValidationError(f"Gateway main chain resolves tool call {tool_call_id!r} more than once")
            resolved_calls.add(tool_call_id)

    final_message = messages[-1]
    if final_message.get("role") != "assistant" or final_message.get("tool_calls"):
        raise TraceValidationError("Gateway main chain does not end with a final assistant message")
    unresolved_calls = sorted(announced_calls - resolved_calls)
    if unresolved_calls:
        raise TraceValidationError(f"Gateway main chain has unresolved tool calls: {unresolved_calls}")
    if not announced_calls:
        raise TraceValidationError("Gateway main chain has no tool calls")
    if reasoning_message_count == 0:
        raise TraceValidationError("Gateway main chain has no assistant reasoning payload")
    return {
        "main_message_count": len(messages),
        "tool_call_count": len(announced_calls),
        "reasoning_message_count": reasoning_message_count,
    }


def _validate_gateway_facts(
    trajectory: Trajectory,
    *,
    expected_version: str,
    require_compaction: bool,
) -> dict[str, Any]:
    selected_llm_nodes = [
        node for node in trajectory.iter_nodes() if node.generated_by == "llm" and node.selection != "rejected_sibling"
    ]
    main_llm_nodes = [node for node in selected_llm_nodes if trajectory.node_chain(node.node_id) == MAIN_CHAIN_ID]
    if not main_llm_nodes:
        raise TraceValidationError("Gateway canonical trajectory has no selected main-chain LLM node")

    codex_timeline = [
        event
        for node in main_llm_nodes
        for event in node.timeline
        if isinstance(event, dict) and event.get("scaffold") == "codex"
    ]
    if not codex_timeline:
        raise TraceValidationError("Gateway LLM nodes have no Codex policy timeline facts")
    versions = sorted(
        {event["scaffold_version"] for event in codex_timeline if isinstance(event.get("scaffold_version"), str)}
    )
    if versions != [expected_version]:
        raise TraceValidationError(f"expected Codex scaffold version {expected_version!r}, found {versions}")

    request_ids = [event.get("request_id") for event in codex_timeline]
    if any(not isinstance(request_id, str) or not request_id for request_id in request_ids):
        raise TraceValidationError("Gateway Codex timeline contains an empty provider request_id")
    if len(request_ids) != len(set(request_ids)):
        raise TraceValidationError("Gateway Codex timeline reuses a provider request_id")

    compaction_nodes = [
        node
        for node in selected_llm_nodes
        if node.segment_boundary is not None
        and node.segment_boundary.kind in {"summary_inference", "compaction_resume"}
        and "compaction" in node.segment_boundary.causes
    ]
    if require_compaction and not compaction_nodes:
        raise TraceValidationError("forced-compaction run has no Gateway compaction boundary")

    post_compaction_count = 0
    main_compaction_nodes = [node for node in compaction_nodes if trajectory.node_chain(node.node_id) == MAIN_CHAIN_ID]
    if require_compaction and main_compaction_nodes:
        latest_compaction_node_id = max(node.node_id for node in main_compaction_nodes)
        post_compaction_count = sum(
            node.node_id > latest_compaction_node_id and node not in main_compaction_nodes for node in main_llm_nodes
        )
        if post_compaction_count == 0:
            raise TraceValidationError("Gateway compaction boundary has no later main-chain LLM completion")

    compaction_rules = sorted(
        {str(rule) for node in compaction_nodes for rule in node.segment_boundary.evidence.get("boundary_rules", [])}
    )
    metadata_sources = sorted(
        {
            event["codex_metadata_source"]
            for event in codex_timeline
            if isinstance(event.get("codex_metadata_source"), str)
        }
    )
    return {
        "gateway_session_id": trajectory.session_id,
        "gateway_state": trajectory.state,
        "gateway_termination_reason": trajectory.termination_reason,
        "gateway_node_count": len(trajectory.nodes),
        "gateway_chain_count": len(trajectory.chains),
        "main_llm_node_count": len(main_llm_nodes),
        "scaffold_versions": versions,
        "codex_metadata_sources": metadata_sources,
        "gateway_compaction_boundary_count": len(compaction_nodes),
        "gateway_main_compaction_boundary_count": len(main_compaction_nodes),
        "gateway_child_compaction_boundary_count": len(compaction_nodes) - len(main_compaction_nodes),
        "gateway_compaction_rules": compaction_rules,
        "post_compaction_llm_node_count": post_compaction_count,
    }


def _normalize_registered_tool_name(name: str) -> str:
    for prefix in _REGISTERED_TOOL_NAME_PREFIXES:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(message))
    normalized.pop("provider_specific_fields", None)
    for tool_call in normalized.get("tool_calls") or []:
        if isinstance(tool_call, dict):
            tool_call.pop("index", None)
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                function["name"] = _normalize_registered_tool_name(function["name"])
    return normalized


def _canonical_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_ledger(
    message_sequences: list[list[dict[str, Any]]],
    *,
    source: str,
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    calls: dict[str, tuple[str, str]] = {}
    results: dict[str, str] = {}
    for messages in message_sequences:
        for message in messages:
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    raise TraceValidationError(f"{source} contains a non-object tool call")
                tool_call_id = tool_call.get("id")
                function = tool_call.get("function")
                if not isinstance(tool_call_id, str) or not isinstance(function, dict):
                    raise TraceValidationError(f"{source} contains a tool call without stable identity")
                name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(name, str) or not isinstance(arguments, str):
                    raise TraceValidationError(f"{source} tool call {tool_call_id!r} lost its name or argument string")
                signature = (_normalize_registered_tool_name(name), arguments)
                previous = calls.get(tool_call_id)
                if previous is not None and previous != signature:
                    raise TraceValidationError(f"{source} tool call {tool_call_id!r} has conflicting payloads")
                calls[tool_call_id] = signature
            if message.get("role") != "tool":
                continue
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                raise TraceValidationError(f"{source} contains a tool result without a tool_call_id")
            result_signature = _canonical_value(message.get("content"))
            previous_result = results.get(tool_call_id)
            if previous_result is not None and previous_result != result_signature:
                raise TraceValidationError(f"{source} tool result {tool_call_id!r} has conflicting payloads")
            results[tool_call_id] = result_signature
    unresolved = sorted(set(calls) - set(results))
    orphan_results = sorted(set(results) - set(calls))
    if unresolved or orphan_results:
        raise TraceValidationError(
            f"{source} tool ledger is not closed: unresolved={unresolved} orphan_results={orphan_results}"
        )
    return calls, results


def _validate_chat_gateway_parity(
    trajectory: Trajectory,
    processor_results: dict[str, Any],
) -> dict[str, Any]:
    chat_result = processor_results.get("chat")
    chat_trajectories = chat_result.get("trajectories") if isinstance(chat_result, dict) else None
    if not isinstance(chat_trajectories, list) or not chat_trajectories:
        raise TraceValidationError("Gateway run has no Chat Tracker trajectories for parity validation")
    chat_message_sequences = [
        _legacy_checker._trajectory_messages(chat_trajectory, trajectory_index)
        for trajectory_index, chat_trajectory in enumerate(chat_trajectories)
    ]
    gateway_messages = [
        node.original_message
        for node in trajectory.iter_nodes()
        if node.selection != "rejected_sibling" and isinstance(node.original_message, dict)
    ]
    chat_calls, chat_results = _tool_ledger(chat_message_sequences, source="Chat Tracker")
    gateway_calls, gateway_results = _tool_ledger([gateway_messages], source="Gateway trajectory")
    if chat_calls != gateway_calls:
        missing = sorted(set(chat_calls) - set(gateway_calls))
        extra = sorted(set(gateway_calls) - set(chat_calls))
        mismatched = sorted(
            tool_call_id
            for tool_call_id in set(chat_calls) & set(gateway_calls)
            if chat_calls[tool_call_id] != gateway_calls[tool_call_id]
        )
        raise TraceValidationError(
            f"Chat/Gateway tool call ledger mismatch: missing={missing} extra={extra} mismatched={mismatched}"
        )
    if chat_results != gateway_results:
        missing = sorted(set(chat_results) - set(gateway_results))
        extra = sorted(set(gateway_results) - set(chat_results))
        mismatched = sorted(
            tool_call_id
            for tool_call_id in set(chat_results) & set(gateway_results)
            if chat_results[tool_call_id] != gateway_results[tool_call_id]
        )
        raise TraceValidationError(
            f"Chat/Gateway tool result ledger mismatch: missing={missing} extra={extra} mismatched={mismatched}"
        )

    chat_assistant_outputs = {
        _canonical_value(_normalize_message(message))
        for messages in chat_message_sequences
        for message in messages
        if message.get("role") == "assistant"
    }
    selected_llm_nodes = [
        node for node in trajectory.iter_nodes() if node.generated_by == "llm" and node.selection != "rejected_sibling"
    ]
    missing_llm_node_ids = [
        node.node_id
        for node in selected_llm_nodes
        if _canonical_value(_normalize_message(node.original_message)) not in chat_assistant_outputs
    ]
    if missing_llm_node_ids:
        raise TraceValidationError(
            f"Gateway LLM outputs are missing from Chat Tracker: node_ids={missing_llm_node_ids}"
        )

    latest_messages = trajectory.to_messages(MAIN_CHAIN_ID)
    normalized_latest_messages = [_normalize_message(message) for message in latest_messages]
    normalized_chat_trajectories = [
        [_normalize_message(message) for message in messages] for messages in chat_message_sequences
    ]
    latest_messages_match = normalized_latest_messages in normalized_chat_trajectories
    if not latest_messages_match:
        raise TraceValidationError("Gateway latest main messages do not match any Chat Tracker trajectory")
    return {
        "chat_gateway_tool_call_parity": True,
        "chat_gateway_tool_result_parity": True,
        "chat_tool_call_count": len(chat_calls),
        "gateway_tool_call_count": len(gateway_calls),
        "llm_output_parity_count": len(selected_llm_nodes),
        "latest_messages_match": True,
    }


def _validate_gateway_subagents(
    trajectory: Trajectory,
    multi_agent_summary: dict[str, Any],
    *,
    expected_version: str,
    require_multi_agent: bool,
) -> dict[str, Any]:
    child_chains = {chain_id: chain for chain_id, chain in trajectory.chains.items() if chain_id != MAIN_CHAIN_ID}
    child_chain_ids = sorted(str(chain_id) for chain_id in child_chains)
    if require_multi_agent:
        expected_child_ids = multi_agent_summary["subagent_child_ids"]
        if child_chain_ids != expected_child_ids:
            raise TraceValidationError(
                f"Gateway child chain IDs do not match raw Codex children: raw={expected_child_ids} "
                f"gateway={child_chain_ids}"
            )

    child_llm_node_count = 0
    for child_chain_id, chain in child_chains.items():
        if chain.parent_chain_id != MAIN_CHAIN_ID:
            raise TraceValidationError(
                f"Gateway child chain {child_chain_id!r} has parent {chain.parent_chain_id!r}, expected 'main'"
            )
        if chain.detector != "codex_turn_metadata":
            raise TraceValidationError(
                f"Gateway child chain {child_chain_id!r} has detector {chain.detector!r}, "
                "expected 'codex_turn_metadata'"
            )
        child_llm_nodes = [
            node
            for node in trajectory.iter_nodes()
            if node.generated_by == "llm"
            and node.selection != "rejected_sibling"
            and trajectory.node_chain(node.node_id) == child_chain_id
        ]
        if not child_llm_nodes:
            raise TraceValidationError(f"Gateway child chain {child_chain_id!r} has no selected LLM node")
        child_llm_node_count += len(child_llm_nodes)
        for node in child_llm_nodes:
            codex_events = [
                event for event in node.timeline if isinstance(event, dict) and event.get("scaffold") == "codex"
            ]
            if not any(
                event.get("agent_id") == child_chain_id
                and event.get("parent_agent_id") == multi_agent_summary["root_thread_id"]
                and event.get("is_sidechain") is True
                and event.get("scaffold_version") == expected_version
                for event in codex_events
            ):
                raise TraceValidationError(
                    f"Gateway child LLM node {node.node_id} lost child/parent/version policy facts"
                )
        child_messages = trajectory.to_messages(child_chain_id)
        if not child_messages or child_messages[-1].get("role") != "assistant":
            raise TraceValidationError(f"Gateway child chain {child_chain_id!r} has no terminal assistant message")
    if require_multi_agent and len(child_chains) != 1:
        raise TraceValidationError(f"expected exactly one Gateway child chain, found {len(child_chains)}")
    return {
        "gateway_child_chain_count": len(child_chains),
        "gateway_child_chain_ids": child_chain_ids,
        "gateway_child_llm_node_count": child_llm_node_count,
    }


def validate_trace(
    results_path: str | Path,
    database_path: str | Path,
    *,
    expected_version: str,
    require_compaction: bool,
    require_multi_agent: bool = False,
) -> dict[str, Any]:
    """Validate raw Codex events, batch spans, and the canonical Gateway dump."""
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
    chat_summary = _validate_chat_trajectories(
        {"chat": processor_results.get("chat")},
        require_multi_agent=require_multi_agent,
    )
    if require_multi_agent and chat_summary["spawn_tool_call_count"] != multi_agent_summary["subagent_spawn_count"]:
        raise TraceValidationError(
            "raw/chat spawn_agent count mismatch: "
            f"raw={multi_agent_summary['subagent_spawn_count']} chat={chat_summary['spawn_tool_call_count']}"
        )
    trajectory = _load_gateway_trajectory(processor_results, run_id)
    message_summary = _validate_main_messages(trajectory)
    gateway_summary = _validate_gateway_facts(
        trajectory,
        expected_version=expected_version,
        require_compaction=require_compaction,
    )
    parity_summary = _validate_chat_gateway_parity(trajectory, processor_results)
    subagent_summary = _validate_gateway_subagents(
        trajectory,
        multi_agent_summary,
        expected_version=expected_version,
        require_multi_agent=require_multi_agent,
    )
    span_summary = _validate_trace_spans(database_path, run_id, require_compaction=require_compaction)
    compaction_count_parity = span_summary["compaction_count"] == gateway_summary["gateway_compaction_boundary_count"]
    if not compaction_count_parity:
        raise TraceValidationError(
            "trace/Gateway compaction count mismatch: "
            f"trace={span_summary['compaction_count']} "
            f"gateway={gateway_summary['gateway_compaction_boundary_count']}"
        )
    return {
        "run_id": run_id,
        "status": record["status"],
        "raw_event_count": len(raw_events),
        "raw_event_counts": dict(sorted(raw_item_counts.items())),
        **multi_agent_summary,
        "chat_main_message_count": chat_summary["main_message_count"],
        "chat_spawn_tool_call_count": chat_summary["spawn_tool_call_count"],
        **message_summary,
        **gateway_summary,
        **parity_summary,
        **subagent_summary,
        **span_summary,
        "compaction_count_parity": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", type=Path)
    parser.add_argument("database_path", type=Path)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--require-compaction", action="store_true")
    parser.add_argument("--require-multi-agent", action="store_true")
    args = parser.parse_args()
    try:
        summary = validate_trace(
            args.results_path,
            args.database_path,
            expected_version=args.expected_version,
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
