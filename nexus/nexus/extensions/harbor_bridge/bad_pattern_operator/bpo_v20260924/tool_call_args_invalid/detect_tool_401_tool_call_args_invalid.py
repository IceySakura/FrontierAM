"""tool_call_args_invalid — validate tool-call arguments against authoritative schemas."""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List

from .._core.detector import Finding
from .._core.messages import TrajectoryView, pair_tool_calls_with_responses
from .._core.rule_detector import RuleDetector
from .._core.tool_registry import classify_tool_result

from .rules import (  # type: ignore[import-not-found]
    parse_arguments,
    validate_schema_instance,
)


def _snippet(text: str, max_len: int = 240) -> str:
    value = str(text or "").replace("\n", "\\n")
    return value[:max_len] + (" …" if len(value) > max_len else "")


# v1.2.0(102427 跨 benchmark 校验新增):HY3.0 工具调用是 md5 特殊 token 编码
# (<arg_key:6124c78e> / <arg_value:…> / <tool_call:…>)。当解码器没把它拆开、这些
# token 漏进 arguments 时,会出现参数边界崩坏(例如把相邻 key/value 拼进一个键值)。
# 只有污染位置能解释该调用的全部 schema 违规时才作不可判定证据；普通文档里引用
# 标签、或还有无关违规时保持原判。
_HY3_TAG_RE = re.compile(
    r"</?(?P<tag>arg_key|arg_value|tool_calls?|tool_sep):6124c78e>"
)
_HY3_PAIRED_ARGUMENT_RE = re.compile(
    r"<arg_key:6124c78e>\s*"
    r"(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*"
    r"</arg_key:6124c78e>\s*<arg_value:6124c78e>"
)


def _hy3_arg_tags_unbalanced(value: str) -> bool:
    balance: Counter[str] = Counter()
    for match in _HY3_TAG_RE.finditer(value):
        tag = match.group("tag")
        if tag not in {"arg_key", "arg_value"}:
            continue
        balance[tag] += -1 if match.group(0).startswith("</") else 1
    return any(balance[tag] != 0 for tag in ("arg_key", "arg_value"))


def _json_path(parts: List[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_-]*", part
        ):
            path += f".{part}"
        else:
            path += f"[{json.dumps(part, ensure_ascii=False)}]"
    return path


def _hy3_corruption_sites(
    instance: Any,
    parts: List[Any] | None = None,
) -> tuple[List[tuple[str, str, str]], Dict[str, set[str]]]:
    current = [] if parts is None else parts
    corrupt_keys: List[tuple[str, str, str]] = []
    paired_values: Dict[str, set[str]] = {}
    if isinstance(instance, dict):
        for key, value in instance.items():
            key_text = str(key)
            if _HY3_TAG_RE.search(key_text):
                corrupt_keys.append((
                    _json_path(current),
                    _json_path([*current, key_text]),
                    key_text,
                ))
            child_keys, child_values = _hy3_corruption_sites(
                value, [*current, key_text]
            )
            corrupt_keys.extend(child_keys)
            for path, keys in child_values.items():
                paired_values.setdefault(path, set()).update(keys)
    elif isinstance(instance, list):
        for index, value in enumerate(instance):
            child_keys, child_values = _hy3_corruption_sites(
                value, [*current, index]
            )
            corrupt_keys.extend(child_keys)
            for path, keys in child_values.items():
                paired_values.setdefault(path, set()).update(keys)
    elif isinstance(instance, str):
        embedded_keys = set()
        if _hy3_arg_tags_unbalanced(instance):
            embedded_keys = {
                match.group("key")
                for match in _HY3_PAIRED_ARGUMENT_RE.finditer(instance)
            }
        if embedded_keys:
            paired_values[_json_path(current)] = embedded_keys
    return corrupt_keys, paired_values


def _path_parent(path: str, field: str) -> str | None:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", field):
        suffix = f".{field}"
    else:
        suffix = f"[{json.dumps(field, ensure_ascii=False)}]"
    return path[:-len(suffix)] if path.endswith(suffix) else None


def _corrupt_key_explains_violation(
    violation: Dict[str, Any],
    corrupt_keys: List[tuple[str, str, str]],
) -> bool:
    kind = violation.get("kind")
    path = str(violation.get("path") or violation.get("field") or "$")
    field = str(violation.get("field") or "")
    for parent, key_path, key_text in corrupt_keys:
        if kind == "unknown_parameter" and path == key_path:
            return True
        if (
            kind == "missing_required"
            and _path_parent(path, field) == parent
            and re.search(
                rf"(?<![A-Za-z0-9_-]){re.escape(field)}"
                r"(?![A-Za-z0-9_-])",
                _HY3_TAG_RE.sub(" ", key_text),
            )
        ):
            return True
    return False


def _hy3_corruption_explains_violations(
    instance: Any,
    violations: List[Dict[str, Any]],
    schema: Dict[str, Any],
) -> bool:
    corrupt_keys, paired_values = _hy3_corruption_sites(instance)
    if not corrupt_keys and not paired_values:
        return False
    properties = schema.get("properties")
    top_level_properties = set(properties) if isinstance(properties, dict) else set()

    for violation in violations:
        path = str(violation.get("path") or violation.get("field") or "$")
        if _corrupt_key_explains_violation(violation, corrupt_keys):
            continue

        expected = violation.get("expected_type")
        expected_types = expected if isinstance(expected, list) else [expected]
        embedded_keys = paired_values.get(path, set())
        if (
            embedded_keys
            and violation.get("kind") == "type_mismatch"
            and violation.get("actual_json_type") == "string"
            and "string" not in expected_types
            and any(
                key in top_level_properties and key != violation.get("field")
                for key in embedded_keys
            )
        ):
            continue
        return False
    return True


class Tool401ToolCallArgsInvalidDetector(RuleDetector):
    detector_id = "tool_call_args_invalid"
    thresholds: Dict[str, Any] = {
        "checks": [
            "missing_required",
            "type_mismatch",
            "enum_violation",
            "unknown_parameter",
            "schema_constraint",
        ],
        "schema_features": [
            "nested_objects",
            "local_$ref",
            "anyOf_oneOf_allOf",
            "additionalProperties",
            "openapi_nullable",
        ],
        "ownership": {
            "arguments_json_parse_failure": "tool_call_args_parse_fail (tool_call_args_parse_fail)",
            "undefined_or_inactive_tool": "tool_call_hallucination (tool_call_hallucination)",
            "resolved_schema_violation": "tool_call_args_invalid (tool_call_args_invalid)",
        },
        "unknown_policy": (
            "Missing, conflicting, invalid, or externally-referenced schemas, "
            "and schema violations fully explained by structural HY3 argument "
            "corruption, emit non-issue judgement=unjudgeable evidence."
        ),
    }

    def scan(self, view: TrajectoryView) -> List[Finding]:
        findings: List[Finding] = []
        unjudgeable: List[Dict[str, Any]] = []
        timeline = view.tool_registry

        for call, response in pair_tool_calls_with_responses(view.messages):
            name = call.raw_name or call.name
            instance, parse_ok = parse_arguments(call.args_raw, call.args)
            if not parse_ok:
                continue  # tool_call_args_parse_fail owns malformed arguments JSON.

            availability = timeline.availability_for(
                name, call.turn_index, response
            )
            if availability.status in {"undefined", "unavailable"}:
                continue  # tool_call_hallucination owns call-time tool availability.

            result = classify_tool_result(response, name)
            schema_resolution = timeline.schema_for(name, call.turn_index)
            if schema_resolution.status != "resolved" or schema_resolution.schema is None:
                unjudgeable.append({
                    "turn_index": call.turn_index,
                    "tool_name": name,
                    "tool_call_id": call.tool_call_id,
                    "reason": schema_resolution.reason,
                    "schema_sources": list(schema_resolution.sources),
                    "availability": availability.status,
                    "availability_source": availability.source,
                    "result_state": result.state,
                    "result_turn_index": result.response_turn,
                })
                continue

            validation = validate_schema_instance(instance, schema_resolution.schema)
            if validation.is_unjudgeable:
                unjudgeable.append({
                    "turn_index": call.turn_index,
                    "tool_name": name,
                    "tool_call_id": call.tool_call_id,
                    "reason": validation.unjudgeable_reason,
                    "schema_error": validation.schema_error,
                    "schema_sources": list(schema_resolution.sources),
                    "availability": availability.status,
                    "availability_source": availability.source,
                    "result_state": result.state,
                    "result_turn_index": result.response_turn,
                })
                continue

            violations = list(validation.violations)
            if not violations:
                continue
            # v1.2.0:只在 HY3 标签污染能够解释该调用的全部 schema 违规时改为
            # 不可判定。普通文档正文即使展示成对标签,也不能吞掉无关 required/type
            # 违规；schema-valid 的 literal 示例保持 clean。
            if _hy3_corruption_explains_violations(
                instance,
                violations,
                schema_resolution.schema,
            ):
                unjudgeable.append({
                    "turn_index": call.turn_index,
                    "tool_name": name,
                    "tool_call_id": call.tool_call_id,
                    "reason": "hy3_tool_call_token_corruption",
                    "schema_sources": list(schema_resolution.sources),
                    "availability": availability.status,
                    "availability_source": availability.source,
                    "result_state": result.state,
                    "result_turn_index": result.response_turn,
                })
                continue
            kinds = sorted({str(v["kind"]) for v in violations})
            paths = [str(v.get("path") or v.get("field") or "$") for v in violations]
            has_stringified = any(v.get("stringified_json") for v in violations)
            detail = {
                "pattern_id": "tool_call_args_invalid",
                "judgement": "issue",
                "tool_name": name,
                "tool_call_id": call.tool_call_id,
                "tool_call_turn_index": call.turn_index,
                "schema_sources": list(schema_resolution.sources),
                "schema_source": schema_resolution.source_label,
                "availability": availability.status,
                "availability_source": availability.source,
                "violation_kinds": kinds,
                "violations": violations,
                "n_violations": len(violations),
                "violation_paths": paths,
                "stringified_json": has_stringified,
                "result_state": result.state,
                "result_pattern": result.pattern,
                "result_turn_index": result.response_turn,
                "field": "tool_calls",
            }
            quotes: List[Dict[str, Any]] = [{
                "kind": "tool_call_args_invalid",
                "turn_index": call.turn_index,
                "role": "assistant",
                "tool_name": name,
                "tool_call_id": call.tool_call_id,
                "schema_source": schema_resolution.source_label,
                "violations": violations,
                "args_snippet": _snippet(call.args_raw or str(instance)),
            }]
            if response is not None:
                quotes.append({
                    "kind": "tool_result_cross_evidence",
                    "turn_index": response.turn_index,
                    "role": "tool",
                    "tool_call_id": response.tool_call_id,
                    "result_state": result.state,
                    "content_snippet": _snippet(response.text),
                })
            findings.append(Finding(
                rule_id="tool_call_args_invalid",
                severity="medium",
                is_issue=True,
                turn_index=call.turn_index,
                description=(
                    f"assistant 第 {call.turn_index} 条消息的 tool_call "
                    f"{name!r} 参数违反权威 schema（{', '.join(kinds)}），"
                    f"路径：{paths}。"
                ),
                detail=detail,
                quotes=quotes,
            ))

        if unjudgeable:
            counts = Counter(str(item["reason"]) for item in unjudgeable)
            first_turn = min(int(item["turn_index"]) for item in unjudgeable)
            examples = unjudgeable[:20]
            findings.append(Finding(
                rule_id="tool_call_schema_unjudgeable",
                severity="candidate",
                is_issue=False,
                turn_index=first_turn,
                description=(
                    f"该轨迹有 {len(unjudgeable)} 次 tool_call 缺少足够证据判断"
                    "参数合法性；未计为 hit 或 clean。"
                ),
                detail={
                    "pattern_id": "tool_call_schema_unjudgeable",
                    "judgement": "unjudgeable",
                    "owner_detector": self.detector_id,
                    "n_calls": len(unjudgeable),
                    "reason_counts": dict(sorted(counts.items())),
                    "examples": examples,
                    "examples_truncated": len(unjudgeable) > len(examples),
                    "registry_sources": list(timeline.registry_sources),
                    "registry_dynamic": timeline.dynamic,
                },
                quotes=[{
                    "kind": "tool_call_schema_unjudgeable",
                    **item,
                } for item in examples[:5]],
            ))

        return findings


__all__ = ["Tool401ToolCallArgsInvalidDetector"]
