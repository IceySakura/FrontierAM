"""JSON-Schema and OpenAPI-schema validation helpers for tool_call_args_invalid.

Parsing ownership stays separate: malformed ``function.arguments`` belongs to
tool_call_args_parse_fail.  Once arguments are valid JSON and a schema is authoritative,
tool_call_args_invalid validates the complete instance (including nested objects, local
``$ref`` values, combinators, enum and additional-property policy). Tool menus
also commonly expose OpenAPI 3.0 Schema Objects, whose ``nullable: true``
keyword must be normalized before a standard JSON-Schema validator is used.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from jsonschema import validators


PARSE_FAIL_KEY = "_raw"


@dataclass(frozen=True, slots=True)
class SchemaValidationResult:
    violations: Tuple[Dict[str, Any], ...] = ()
    unjudgeable_reason: str = ""
    schema_error: str = ""

    @property
    def is_unjudgeable(self) -> bool:
        return bool(self.unjudgeable_reason)


@dataclass(frozen=True, slots=True)
class _PreparedSchema:
    validator_cls: Any = None
    schema: Optional[Dict[str, Any]] = None
    unjudgeable_reason: str = ""
    schema_error: str = ""


def is_parse_failure(args: Dict[str, Any]) -> bool:
    """Compatibility helper for the adapter's legacy parse-failure sentinel."""
    return isinstance(args, dict) and set(args.keys()) == {PARSE_FAIL_KEY}


def parse_arguments(args_raw: str, fallback: Any) -> Tuple[Any, bool]:
    """Return the original JSON value and whether parsing was decidable.

    ``ToolCall.args`` is dict-shaped for legacy consumers and therefore loses a
    valid top-level array/scalar.  ``args_raw`` retains the original JSON and is
    authoritative when present.  Empty arguments preserve the normalized
    fallback (normally ``{}``).
    """
    raw = str(args_raw or "").strip()
    if not raw:
        if is_parse_failure(fallback) and fallback.get(PARSE_FAIL_KEY) == "":
            return {}, True
        return fallback, not is_parse_failure(fallback) if isinstance(fallback, dict) else True
    try:
        return json.loads(raw), True
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, False


def tool_schema_index(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Backward-compatible static schema index used by older SDK callers."""
    out: Dict[str, Dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        schema = fn.get("parameters")
        out[name] = dict(schema) if isinstance(schema, dict) else {}
    return out


def _external_refs(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            out.append(ref)
        for child in value.values():
            out.extend(_external_refs(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(_external_refs(child))
    return out


def _normalize_openapi_nullable(value: Any) -> Any:
    """Translate OpenAPI 3.0 ``nullable`` into JSON-Schema type unions.

    OpenAPI 3.0 gives ``nullable: true`` meaning only when ``type`` is declared
    on the same Schema Object. Recursing through dictionaries and arrays also
    covers properties, local definitions, items, and combinator branches. The
    input is the freshly deserialized cache value, so no caller-owned schema is
    mutated.
    """
    if isinstance(value, list):
        return [_normalize_openapi_nullable(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _normalize_openapi_nullable(child)
        for key, child in value.items()
    }
    if normalized.get("nullable") is not True or "type" not in normalized:
        return normalized

    declared = normalized.get("type")
    if isinstance(declared, str):
        if declared != "null":
            normalized["type"] = [declared, "null"]
    elif isinstance(declared, list) and all(
        isinstance(item, str) for item in declared
    ):
        if "null" not in declared:
            normalized["type"] = [*declared, "null"]
    return normalized


@lru_cache(maxsize=512)
def _prepare_schema(serialized: str) -> _PreparedSchema:
    """Check and cache one canonical schema without sharing validator state."""
    schema = _normalize_openapi_nullable(json.loads(serialized))
    external = sorted(set(_external_refs(schema)))
    if external:
        return _PreparedSchema(
            unjudgeable_reason="external_ref_unsupported",
            schema_error=", ".join(external[:5]),
        )
    try:
        validator_cls = validators.validator_for(schema)
        validator_cls.check_schema(schema)
    except Exception as exc:  # invalid schemas are evidence gaps, not model errors
        return _PreparedSchema(
            unjudgeable_reason="schema_validation_unavailable",
            schema_error=f"{type(exc).__name__}: {str(exc)[:240]}",
        )
    return _PreparedSchema(validator_cls=validator_cls, schema=schema)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        elif isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", part):
            path += f".{part}"
        else:
            path += "[" + json.dumps(part, ensure_ascii=False) + "]"
    return path


def _safe_value(value: Any, limit: int = 160) -> Any:
    if isinstance(value, (type(None), bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[:limit] + "…"


def _looks_like_stringified(value: Any, expected: Any) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    if not isinstance(value, str) or not ({"object", "array"} & set(types)):
        return False
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return ("object" in types and isinstance(decoded, dict)) or (
        "array" in types and isinstance(decoded, list)
    )


def _unknown_properties(error: Any) -> List[str]:
    if error.validator != "additionalProperties" or error.validator_value is not False:
        return []
    if not isinstance(error.instance, dict) or not isinstance(error.schema, dict):
        return []
    declared = set((error.schema.get("properties") or {}).keys())
    patterns = []
    for pattern in (error.schema.get("patternProperties") or {}).keys():
        try:
            patterns.append(re.compile(pattern))
        except re.error:
            pass
    return sorted(
        str(key)
        for key in error.instance
        if key not in declared and not any(p.search(str(key)) for p in patterns)
    )


def _missing_required(error: Any) -> List[str]:
    if error.validator != "required" or not isinstance(error.instance, dict):
        return []
    required = error.validator_value
    if not isinstance(required, list):
        return []
    return [name for name in required if isinstance(name, str) and name not in error.instance]


def _violation_rows(error: Any) -> Iterable[Dict[str, Any]]:
    base_path = list(error.absolute_path)
    schema_path = _json_path(list(error.absolute_schema_path))
    message = str(error.message)[:300]

    missing = _missing_required(error)
    if missing:
        properties = error.schema.get("properties") if isinstance(error.schema, dict) else {}
        for name in missing:
            prop = properties.get(name) if isinstance(properties, dict) else None
            yield {
                "kind": "missing_required",
                "field": name,
                "path": _json_path([*base_path, name]),
                "expected": "required property",
                "expected_type": prop.get("type") if isinstance(prop, dict) else None,
                "actual": "missing",
                "schema_path": schema_path,
                # jsonschema emits one Required error per missing name. This
                # field-specific stable message lets the outer deduper collapse
                # the repeated full-required-list expansion deterministically.
                "message": f"{name!r} is a required property",
            }
        return

    unknown = _unknown_properties(error)
    if unknown:
        for name in unknown:
            yield {
                "kind": "unknown_parameter",
                "field": name,
                "path": _json_path([*base_path, name]),
                "expected": "no additional property",
                "actual": "present",
                "schema_path": schema_path,
                "message": message,
            }
        return

    path = _json_path(base_path)
    field = str(base_path[-1]) if base_path else "$"
    if error.validator == "type":
        expected = error.validator_value
        yield {
            "kind": "type_mismatch",
            "field": field,
            "path": path,
            "expected": expected,
            "expected_type": expected,
            "actual": _safe_value(error.instance),
            "actual_type": type(error.instance).__name__,
            "actual_json_type": _json_type(error.instance),
            "stringified_json": _looks_like_stringified(error.instance, expected),
            "schema_path": schema_path,
            "message": message,
        }
        return
    if error.validator == "enum":
        yield {
            "kind": "enum_violation",
            "field": field,
            "path": path,
            "expected": [_safe_value(v) for v in error.validator_value],
            "expected_values": [_safe_value(v) for v in error.validator_value],
            "actual": _safe_value(error.instance),
            "actual_type": type(error.instance).__name__,
            "actual_json_type": _json_type(error.instance),
            "schema_path": schema_path,
            "message": message,
        }
        return

    # Combinators and the remaining standard JSON-Schema constraints are still
    # genuine contract violations.  Keep their exact validator and paths
    # rather than guessing which alternative the author intended.
    yield {
        "kind": "schema_constraint",
        "constraint": str(error.validator),
        "field": field,
        "path": path,
        "expected": _safe_value(error.validator_value),
        "actual": _safe_value(error.instance),
        "actual_type": type(error.instance).__name__,
        "actual_json_type": _json_type(error.instance),
        "schema_path": schema_path,
        "message": message,
    }


def validate_schema_instance(instance: Any, schema: Dict[str, Any]) -> SchemaValidationResult:
    if not isinstance(schema, dict):
        return SchemaValidationResult(unjudgeable_reason="schema_missing")
    try:
        serialized = json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        prepared = _prepare_schema(serialized)
    except (TypeError, ValueError) as exc:
        return SchemaValidationResult(
            unjudgeable_reason="schema_validation_unavailable",
            schema_error=f"{type(exc).__name__}: {str(exc)[:240]}",
        )
    if prepared.unjudgeable_reason:
        return SchemaValidationResult(
            unjudgeable_reason=prepared.unjudgeable_reason,
            schema_error=prepared.schema_error,
        )
    try:
        validator = prepared.validator_cls(prepared.schema)
        errors = sorted(
            validator.iter_errors(instance),
            key=lambda error: (
                tuple(str(p) for p in error.absolute_path),
                tuple(str(p) for p in error.absolute_schema_path),
                str(error.message),
            ),
        )
    except Exception as exc:  # invalid/unresolvable schemas are not model errors
        return SchemaValidationResult(
            unjudgeable_reason="schema_validation_unavailable",
            schema_error=f"{type(exc).__name__}: {str(exc)[:240]}",
        )

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for error in errors:
        for row in _violation_rows(error):
            key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda row: (
        str(row.get("path") or ""),
        str(row.get("kind") or ""),
        str(row.get("field") or ""),
    ))
    return SchemaValidationResult(violations=tuple(rows))


def validate_args(args: Any, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compatibility wrapper returning only decidable violation rows."""
    return list(validate_schema_instance(args, schema).violations)


def type_matches(value: Any, prop_schema: Dict[str, Any]) -> bool:
    """Compatibility helper backed by the same full validator."""
    result = validate_schema_instance(value, prop_schema)
    return not result.is_unjudgeable and not result.violations


__all__ = [
    "PARSE_FAIL_KEY",
    "SchemaValidationResult",
    "is_parse_failure",
    "parse_arguments",
    "tool_schema_index",
    "type_matches",
    "validate_args",
    "validate_schema_instance",
]
