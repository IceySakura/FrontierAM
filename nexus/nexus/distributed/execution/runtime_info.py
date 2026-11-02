from __future__ import annotations

import platform

from typing import Any


RUNTIMES_KEY = 'runtimes'
INPUT_KEY = 'input'
ARTIFACTS_KEY = '_artifacts'
RUN_TRACE_KEY = 'run_trace'
LIVE_ARTIFACTS_KEY = 'live'
_RESERVED_KEYS = {RUNTIMES_KEY, INPUT_KEY, ARTIFACTS_KEY, RUN_TRACE_KEY}
_INPUT_IDENTITY_KEYS = ('turn_id',)


def manager_capability_snapshot(
    *,
    supported_modes: list[str],
    supported_spec_types: list[str],
    supported_instance_modes: list[str],
    supported_spec_versions: dict[str, list[int]],
    bootstrap_version: int | None,
    artifact_ref_version: int | None,
) -> dict[str, Any]:
    return {
        'schema_version': 1,
        'python_version': platform.python_version(),
        'supported_modes': list(supported_modes),
        'supported_spec_types': list(supported_spec_types),
        'supported_instance_modes': list(supported_instance_modes),
        'supported_spec_versions': {key: list(value) for key, value in supported_spec_versions.items()},
        'bootstrap_version': bootstrap_version,
        'artifact_ref_version': artifact_ref_version,
    }


def normalize_runtime_info(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {}

    runtimes: dict[str, dict[str, Any]] = {}
    structured_runtimes = value.get(RUNTIMES_KEY)
    if isinstance(structured_runtimes, dict):
        runtimes.update(_runtime_map(structured_runtimes))

    legacy_runtimes = {
        str(name): raw_runtime
        for name, raw_runtime in value.items()
        if name not in _RESERVED_KEYS and isinstance(raw_runtime, dict)
    }
    runtimes.update(_runtime_map(legacy_runtimes))

    raw_input_payload = value.get(INPUT_KEY)
    input_payload = None
    if isinstance(raw_input_payload, dict):
        input_payload = compact_input_payload(raw_input_payload) or None
    artifacts = value.get(ARTIFACTS_KEY)
    run_trace = value.get(RUN_TRACE_KEY)
    return _runtime_info(runtimes=runtimes, input_payload=input_payload, artifacts=artifacts, run_trace=run_trace)


def merge_runtime_info(existing: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    current = normalize_runtime_info(existing)
    update = normalize_runtime_info(incoming)

    runtimes = {
        **runtime_entries(current),
        **runtime_entries(update),
    }
    input_payload = update.get(INPUT_KEY, current.get(INPUT_KEY))
    current_artifacts = current.get(ARTIFACTS_KEY)
    update_artifacts = update.get(ARTIFACTS_KEY)
    artifacts = _merge_dicts(current_artifacts, update_artifacts)
    current_run_trace = current.get(RUN_TRACE_KEY)
    update_run_trace = update.get(RUN_TRACE_KEY)
    run_trace = _merge_dicts(current_run_trace, update_run_trace)
    return _runtime_info(runtimes=runtimes, input_payload=input_payload, artifacts=artifacts, run_trace=run_trace)


def compact_input_payload(payload: dict[str, Any]) -> dict[str, str]:
    return {key: value for key in _INPUT_IDENTITY_KEYS if isinstance((value := payload.get(key)), str) and value}


def runtime_entries(value: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    normalized = normalize_runtime_info(value)
    runtimes = normalized.get(RUNTIMES_KEY)
    return dict(runtimes) if isinstance(runtimes, dict) else {}


def _runtime_map(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(name): dict(raw_runtime) for name, raw_runtime in value.items() if isinstance(raw_runtime, dict)}


def _merge_dicts(existing: Any, incoming: Any) -> dict[str, Any] | None:
    if isinstance(existing, dict) and isinstance(incoming, dict):
        return {**existing, **incoming}
    if isinstance(incoming, dict):
        return dict(incoming)
    if isinstance(existing, dict):
        return dict(existing)
    return None


def _runtime_info(
    *,
    runtimes: dict[str, dict[str, Any]],
    input_payload: Any = None,
    artifacts: Any = None,
    run_trace: Any = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if runtimes:
        result[RUNTIMES_KEY] = runtimes
    if input_payload is not None:
        result[INPUT_KEY] = input_payload
    if isinstance(artifacts, dict):
        result[ARTIFACTS_KEY] = artifacts
    if isinstance(run_trace, dict):
        result[RUN_TRACE_KEY] = run_trace
    return result
