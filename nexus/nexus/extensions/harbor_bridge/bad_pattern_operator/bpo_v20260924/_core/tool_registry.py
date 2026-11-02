"""Authoritative tool/schema timeline for one normalized trajectory.

The request's initial ``tools`` list is only one observation.  Some harnesses
mutate a per-trajectory tool registry through ordinary tool calls (list,
inspect, add, remove), while other exports omit the registry altogether.  This
module folds those observations into a conservative timeline:

* availability is decided only from an explicit active definition, a complete
  registry snapshot, a successful registry mutation, or the paired runtime
  response for the call itself;
* schemas retain their source and effective turn;
* missing/conflicting evidence stays ``unjudgeable`` instead of becoming a
  false hit or a false clean.

Registry controllers are discovered from their own name, description and
parameter schema.  There are deliberately no benchmark, model, or concrete
controller-name allowlists here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple


RegistryOperation = Literal["add", "remove", "list", "inspect"]
AvailabilityStatus = Literal["available", "unavailable", "undefined", "unjudgeable"]
ResultState = Literal["success", "failure", "unavailable", "returned", "missing"]
SchemaStatus = Literal["resolved", "unjudgeable"]


@dataclass(frozen=True, slots=True)
class ToolDefinitionEvidence:
    name: str
    schema: Optional[Dict[str, Any]]
    description: str
    source: str
    effective_turn: int = -1


@dataclass(frozen=True, slots=True)
class RegistryController:
    tool_name: str
    operation: RegistryOperation
    target_field: Optional[str] = None
    # A list controller can describe the active registry, the inactive tools
    # available to add, or an unspecified catalog.  Only the first two prove
    # call-time availability completeness.
    list_scope: Literal["active", "inactive", "unknown"] = "unknown"
    list_is_complete: bool = False


@dataclass(frozen=True, slots=True)
class RegistryEvent:
    effective_turn: int
    operation: RegistryOperation
    source_tool: str
    target: Optional[str] = None
    listed_names: Tuple[str, ...] = ()
    list_scope: Literal["active", "inactive", "unknown"] = "unknown"
    list_is_complete: bool = False


@dataclass(frozen=True, slots=True)
class ToolResultEvidence:
    state: ResultState
    response_turn: Optional[int]
    pattern: str


@dataclass(frozen=True, slots=True)
class ToolAvailability:
    status: AvailabilityStatus
    source: str
    reason: str
    registry_complete: bool


@dataclass(frozen=True, slots=True)
class ToolSchemaResolution:
    status: SchemaStatus
    schema: Optional[Dict[str, Any]]
    sources: Tuple[str, ...]
    reason: str = ""

    @property
    def source_label(self) -> str:
        return ", ".join(self.sources)


@dataclass(frozen=True, slots=True)
class ToolRegistryTimeline:
    initial_active: frozenset[str] = frozenset()
    definitions: Tuple[ToolDefinitionEvidence, ...] = ()
    controllers: Dict[str, RegistryController] = field(default_factory=dict)
    events: Tuple[RegistryEvent, ...] = ()
    registry_sources: Tuple[str, ...] = ()
    static_complete: bool = False
    dynamic: bool = False
    incomplete_reason: str = ""

    def schema_for(self, tool_name: str, turn_index: int) -> ToolSchemaResolution:
        historical = [
            item
            for item in self.definitions
            if item.name == tool_name
            and item.effective_turn < turn_index
            and isinstance(item.schema, dict)
        ]
        if not historical:
            sources = tuple(sorted({
                item.source
                for item in self.definitions
                if item.name == tool_name and item.effective_turn < turn_index
            }))
            return ToolSchemaResolution(
                status="unjudgeable",
                schema=None,
                sources=sources,
                reason="schema_missing",
            )

        # A later successful inspect/registration is a new call-time contract,
        # not a permanent conflict with an older definition. Conflicts remain
        # undecidable only among sources effective at the same latest turn.
        latest_turn = max(item.effective_turn for item in historical)
        candidates = [
            item for item in historical if item.effective_turn == latest_turn
        ]
        by_value: Dict[str, List[ToolDefinitionEvidence]] = {}
        for item in candidates:
            key = json.dumps(item.schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            by_value.setdefault(key, []).append(item)
        sources = tuple(sorted({item.source for item in candidates}))
        if len(by_value) != 1:
            return ToolSchemaResolution(
                status="unjudgeable",
                schema=None,
                sources=sources,
                reason="schema_conflict",
            )
        return ToolSchemaResolution(
            status="resolved",
            schema=candidates[-1].schema,
            sources=sources,
        )

    def availability_for(
        self,
        tool_name: str,
        turn_index: int,
        response: Any = None,
    ) -> ToolAvailability:
        result = classify_tool_result(response, tool_name)
        if result.state == "unavailable":
            return ToolAvailability(
                status="unavailable",
                source=f"runtime_response@turn={result.response_turn}",
                reason=result.pattern,
                registry_complete=True,
            )
        # A paired response which is not an unavailable-tool rejection proves
        # dispatch reached a real tool.  It does NOT prove the tool operation
        # succeeded; that distinction remains in ``result_state`` evidence.
        if result.state != "missing":
            return ToolAvailability(
                status="available",
                source=f"runtime_response@turn={result.response_turn}",
                reason="runtime_dispatched_call",
                registry_complete=True,
            )

        active = set(self.initial_active)
        inactive: set[str] = set()
        complete = self.static_complete
        completeness_source = "initial_tools" if complete else ""
        last_source: Dict[str, str] = {
            name: "initial_tools" for name in self.initial_active
        }

        for event in self.events:
            if event.effective_turn >= turn_index:
                break
            if event.operation == "add" and event.target:
                active.add(event.target)
                inactive.discard(event.target)
                last_source[event.target] = (
                    f"registry_add@turn={event.effective_turn}"
                )
            elif event.operation == "remove" and event.target:
                active.discard(event.target)
                inactive.add(event.target)
                last_source[event.target] = (
                    f"registry_remove@turn={event.effective_turn}"
                )
            elif event.operation == "list" and event.list_is_complete:
                listed = set(event.listed_names)
                if event.list_scope == "inactive":
                    inactive = listed
                    active.difference_update(inactive)
                elif event.list_scope == "active":
                    active = listed
                    inactive.difference_update(active)
                    for name in active:
                        last_source[name] = (
                            f"registry_list@turn={event.effective_turn}"
                        )
                complete = event.list_scope in {"active", "inactive"}
                if complete:
                    completeness_source = (
                        f"registry_list@turn={event.effective_turn}"
                    )

        if tool_name in active:
            return ToolAvailability(
                status="available",
                source=last_source.get(tool_name, "registry_timeline"),
                reason="tool_active_at_call",
                registry_complete=complete,
            )
        if tool_name in inactive:
            return ToolAvailability(
                status="unavailable",
                source=completeness_source or "registry_timeline",
                reason="tool_inactive_at_call",
                registry_complete=complete,
            )
        if complete:
            return ToolAvailability(
                status="undefined",
                source=completeness_source or "initial_tools",
                reason="tool_absent_from_complete_registry",
                registry_complete=True,
            )
        return ToolAvailability(
            status="unjudgeable",
            source=", ".join(self.registry_sources),
            reason=self.incomplete_reason or "registry_incomplete",
            registry_complete=False,
        )


_TOKEN_BREAK_RE = re.compile(r"[_./:-]+")
_ACTION_WORDS: Dict[RegistryOperation, Tuple[str, ...]] = {
    "add": ("add", "register", "enable", "activate"),
    "remove": ("remove", "unregister", "disable", "deactivate"),
    "list": ("list", "enumerate", "show"),
    "inspect": ("inspect", "describe", "get details", "get detailed"),
}


def _words(text: str) -> str:
    return " ".join(_TOKEN_BREAK_RE.sub(" ", str(text or "").lower()).split())


def _function_body(tool: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(tool, dict):
        return None
    fn = tool.get("function")
    if isinstance(fn, dict):
        return fn
    return tool


def _definition(tool: Any, *, source: str, effective_turn: int = -1) -> Optional[ToolDefinitionEvidence]:
    fn = _function_body(tool)
    if fn is None:
        return None
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    parameters = fn.get("parameters")
    return ToolDefinitionEvidence(
        name=name.strip(),
        schema=dict(parameters) if isinstance(parameters, dict) else None,
        description=str(fn.get("description") or ""),
        source=source,
        effective_turn=effective_turn,
    )


def _target_field(schema: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    if not isinstance(props, dict):
        return None
    ranked: List[Tuple[int, str]] = []
    required = set(schema.get("required") or [])
    for name, prop in props.items():
        if not isinstance(name, str) or not isinstance(prop, dict):
            continue
        types = prop.get("type")
        if types not in (None, "string") and not (
            isinstance(types, list) and "string" in types
        ):
            continue
        text = _words(name + " " + str(prop.get("description") or ""))
        score = 0
        if "tool name" in text or "function name" in text:
            score += 4
        elif "tool" in text or "function" in text:
            score += 2
        if name in required:
            score += 1
        if score:
            ranked.append((score, name))
    if ranked:
        return sorted(ranked, key=lambda x: (-x[0], x[1]))[0][1]
    string_fields = [
        str(name)
        for name, prop in props.items()
        if isinstance(prop, dict) and prop.get("type") == "string"
    ]
    return string_fields[0] if len(string_fields) == 1 else None


def _controller_for(item: ToolDefinitionEvidence) -> Optional[RegistryController]:
    name_text = _words(item.name)
    all_text = _words(item.name + " " + item.description)
    # Requiring an explicit tool/function/registry noun prevents ordinary
    # domain actions such as add_slide or remove_file from becoming registry
    # mutations.
    if not any(
        noun in name_text.split()
        for noun in ("tool", "tools", "function", "functions", "registry", "toolbelt")
    ):
        return None

    operation: Optional[RegistryOperation] = None
    name_tokens = set(name_text.split())
    # The controller name is stronger than prose.  For example a list
    # controller may say "tools you can add" in its description; choosing the
    # first arbitrary action word would misclassify it as an add controller.
    for candidate in ("add", "remove", "list", "inspect"):
        actions = _ACTION_WORDS[candidate]  # type: ignore[index]
        if any(
            action in name_tokens if " " not in action else action in name_text
            for action in actions
        ):
            operation = candidate  # type: ignore[assignment]
            break
    if operation is None:
        positions: List[Tuple[int, RegistryOperation]] = []
        for candidate in ("add", "remove", "list", "inspect"):
            for action in _ACTION_WORDS[candidate]:  # type: ignore[index]
                pos = all_text.find(action)
                if pos >= 0:
                    positions.append((pos, candidate))  # type: ignore[arg-type]
        if positions:
            operation = min(positions, key=lambda item: item[0])[1]
    if operation is None:
        return None

    target = None if operation == "list" else _target_field(item.schema)
    if operation != "list" and not target:
        return None

    list_scope: Literal["active", "inactive", "unknown"] = "unknown"
    complete = False
    if operation == "list":
        desc = _words(item.description)
        complete = "all" in desc.split()
        if any(phrase in desc for phrase in (
            "not currently", "not active", "available to add", "can add",
        )):
            list_scope = "inactive"
        elif any(phrase in desc for phrase in (
            "currently active", "currently registered", "active tools",
            "registered tools", "in your toolbelt", "in the registry",
        )):
            list_scope = "active"
        # A controller named list_tools with no complete-snapshot language is
        # discovery evidence only; it must not make absence decidable.
        if list_scope == "unknown":
            complete = False

    return RegistryController(
        tool_name=item.name,
        operation=operation,
        target_field=target,
        list_scope=list_scope,
        list_is_complete=complete,
    )


def _json_body(text: str) -> Any:
    try:
        return json.loads(str(text or "").strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _listed_names(value: Any) -> Optional[Tuple[str, ...]]:
    if isinstance(value, dict):
        for key in ("tools", "tool_names", "functions", "items", "data"):
            if key in value:
                return _listed_names(value[key])
        return None
    if not isinstance(value, list):
        return None
    names: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
        elif isinstance(item, dict):
            fn = _function_body(item) or {}
            name = fn.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return tuple(sorted(set(names)))


def _mutation_succeeded(response: Any) -> bool:
    if response is None:
        return False
    if getattr(response, "is_error", False):
        return False
    exit_code = getattr(response, "exit_code", None)
    if isinstance(exit_code, int) and exit_code != 0:
        return False
    outcome = getattr(response, "outcome", "unknown")
    if outcome == "failure":
        return False
    if outcome == "success":
        return True
    parsed = _json_body(getattr(response, "text", ""))
    return isinstance(parsed, dict) and parsed.get("success") is True


def _response_definitions(response: Any, *, source: str) -> List[ToolDefinitionEvidence]:
    if response is None:
        return []
    value = _json_body(getattr(response, "text", ""))
    rows: Iterable[Any]
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        nested = next((
            value.get(key)
            for key in ("tools", "functions", "items")
            if isinstance(value.get(key), list)
        ), None)
        rows = nested if isinstance(nested, list) else [value]
    else:
        return []
    out: List[ToolDefinitionEvidence] = []
    turn = int(getattr(response, "turn_index", -1))
    for idx, row in enumerate(rows):
        item = _definition(
            row,
            source=f"{source}[{idx}]",
            effective_turn=turn,
        )
        if item is not None:
            out.append(item)
    return out


def classify_tool_result(response: Any, called_name: str) -> ToolResultEvidence:
    if response is None:
        return ToolResultEvidence("missing", None, "no_paired_response")
    turn = getattr(response, "turn_index", None)
    text = str(getattr(response, "text", "") or "")
    head = " ".join(text[:800].split())
    name = str(called_name or "").strip()
    escaped = re.escape(name)
    unavailable_patterns = (
        ("no_such_tool", rf"no such tool(?: available| found| exists)?\s*[:=]?\s*['\"]?{escaped}"),
        ("tool_not_in_registry", rf"['\"]?{escaped}['\"]?\s+(?:is\s+)?not in (?:the )?(?:toolbelt|registry|available tools)"),
        ("tool_not_available", rf"(?:tool\s+)?['\"]?{escaped}['\"]?\s+(?:is\s+)?not (?:available|registered|enabled|found)"),
        ("unknown_tool", rf"unknown tool\s*[:=]?\s*['\"]?{escaped}"),
    )
    if name:
        for pattern_id, pattern in unavailable_patterns:
            if re.search(pattern, head, re.IGNORECASE):
                return ToolResultEvidence("unavailable", turn, pattern_id)

    outcome = getattr(response, "outcome", "unknown")
    if outcome == "success":
        return ToolResultEvidence("success", turn, "annotated_success")
    if outcome == "failure" or getattr(response, "is_error", False):
        return ToolResultEvidence("failure", turn, "annotated_failure")
    exit_code = getattr(response, "exit_code", None)
    if isinstance(exit_code, int):
        return ToolResultEvidence(
            "success" if exit_code == 0 else "failure",
            turn,
            "exit_code",
        )
    parsed = _json_body(text)
    if isinstance(parsed, dict) and isinstance(parsed.get("success"), bool):
        return ToolResultEvidence(
            "success" if parsed["success"] else "failure",
            turn,
            "json_success_field",
        )
    if re.match(
        r"^\s*(?:input\s+)?validation error\b|^\s*(?:internal\s+)?error\s*:|^\s*exception\s*:",
        text,
        re.IGNORECASE,
    ):
        return ToolResultEvidence("failure", turn, "error_prefix")
    return ToolResultEvidence("returned", turn, "paired_runtime_response")


def build_tool_registry_timeline(
    record: Dict[str, Any],
    messages: Sequence[Any],
) -> ToolRegistryTimeline:
    rec = record if isinstance(record, dict) else {}
    req = rec.get("request") if isinstance(rec.get("request"), dict) else {}

    definitions: List[ToolDefinitionEvidence] = []
    source_names: Dict[str, set[str]] = {}
    registry_sources: List[str] = []
    for source, value in (("request.tools", req.get("tools")), ("tools", rec.get("tools"))):
        if not isinstance(value, list):
            continue
        registry_sources.append(f"{source}[{len(value)}]")
        names: set[str] = set()
        for idx, row in enumerate(value):
            item = _definition(row, source=f"{source}[{idx}]")
            if item is None:
                continue
            definitions.append(item)
            names.add(item.name)
        source_names[source] = names

    initial_active = frozenset(
        item.name for item in definitions if item.effective_turn < 0
    )
    controllers = {
        controller.tool_name: controller
        for item in definitions
        if (controller := _controller_for(item)) is not None
    }
    dynamic = bool(controllers)
    distinct_sets = {frozenset(names) for names in source_names.values()}
    source_conflict = len(distinct_sets) > 1
    static_complete = bool(initial_active) and not dynamic and not source_conflict

    # Local import avoids a core.messages -> tool_registry -> core.messages
    # cycle.  The helper operates on the already-normalized typed messages.
    from .messages import pair_tool_calls_with_responses

    events: List[RegistryEvent] = []
    for call, response in pair_tool_calls_with_responses(messages):
        controller = controllers.get(call.raw_name or call.name)
        if controller is None:
            continue
        result = classify_tool_result(response, call.raw_name or call.name)
        target_value = (
            call.args.get(controller.target_field)
            if controller.target_field and isinstance(call.args, dict)
            else None
        )
        target = target_value.strip() if isinstance(target_value, str) else None
        response_turn = int(getattr(response, "turn_index", call.turn_index))

        if controller.operation in {"inspect", "list"} and result.state not in {
            "failure", "unavailable", "missing",
        }:
            response_definitions = _response_definitions(
                response,
                source=f"registry_{controller.operation}_response@turn={response_turn}",
            )
            if controller.operation == "inspect" and target:
                response_definitions = [
                    item for item in response_definitions if item.name == target
                ]
            definitions.extend(response_definitions)
        if controller.operation == "list" and result.state not in {
            "failure", "unavailable", "missing",
        }:
            names = _listed_names(_json_body(getattr(response, "text", "")))
            if names is not None:
                events.append(RegistryEvent(
                    effective_turn=response_turn,
                    operation="list",
                    source_tool=controller.tool_name,
                    listed_names=names,
                    list_scope=controller.list_scope,
                    list_is_complete=controller.list_is_complete,
                ))
        elif controller.operation in {"add", "remove"} and target and _mutation_succeeded(response):
            events.append(RegistryEvent(
                effective_turn=response_turn,
                operation=controller.operation,
                source_tool=controller.tool_name,
                target=target,
            ))

    if not registry_sources:
        incomplete_reason = "tool_registry_absent"
    elif not initial_active:
        incomplete_reason = "tool_registry_empty"
    elif source_conflict:
        incomplete_reason = "tool_registry_sources_conflict"
    elif dynamic:
        incomplete_reason = "dynamic_registry_without_complete_snapshot"
    else:
        incomplete_reason = ""

    return ToolRegistryTimeline(
        initial_active=initial_active,
        definitions=tuple(sorted(
            definitions,
            key=lambda item: (item.effective_turn, item.name, item.source),
        )),
        controllers=dict(controllers),
        events=tuple(sorted(
            events,
            key=lambda item: (item.effective_turn, item.operation, item.target or ""),
        )),
        registry_sources=tuple(registry_sources),
        static_complete=static_complete,
        dynamic=dynamic,
        incomplete_reason=incomplete_reason,
    )


__all__ = [
    "AvailabilityStatus",
    "RegistryController",
    "RegistryEvent",
    "ResultState",
    "ToolAvailability",
    "ToolDefinitionEvidence",
    "ToolRegistryTimeline",
    "ToolResultEvidence",
    "ToolSchemaResolution",
    "build_tool_registry_timeline",
    "classify_tool_result",
]
