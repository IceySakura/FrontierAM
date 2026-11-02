"""Self-contained bad-pattern detectors exported from Bad-Pattern-Operator.

    from badpattern_detectors import detect
    detect(record)                              # all detectors
    detect(record, ["tool_call_args_invalid"])  # a subset

Each result is ``{"is_issue": bool, "severity": str | None, "findings": [...]}``.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Optional

SOURCE_COMMIT = "e96ed0b82e62c940b9cb5e3d7dee97577a0bf2a1"

_REGISTRY = {
    "tool_call_args_invalid": ("tool_call_args_invalid.detect_tool_401_tool_call_args_invalid", "Tool401ToolCallArgsInvalidDetector"),
    "tool_call_protocol_leak_hy": ("tool_call_protocol_leak_hy.detect_tool_call_protocol_leak_hy", "ToolCallProtocolLeakHyDetector"),
    "tool_call_protocol_leak_all_family": ("tool_call_protocol_leak_all_family.detect_tool_call_protocol_leak_all_family", "ToolCallProtocolLeakDetector"),
    "think_protocol_leak_hy": ("think_protocol_leak_hy.detect_think_protocol_leak_hy", "ThinkProtocolLeakHyDetector"),
    "think_protocol_leak_all_family": ("think_protocol_leak_all_family.detect_think_protocol_leak_all_family", "ThinkProtocolLeakDetector"),
    "reasoning_selftalk_flood": ("reasoning_selftalk_flood.detect_reasoning_selftalk_flood", "ReasoningSelftalkFloodDetector"),
    "zh_ask_foreign_reply": ("zh_ask_foreign_reply.detect_zh_ask_foreign_reply", "ZhAskForeignReplyDetector"),
}

BEHAVIOR_VERSIONS = {
    "tool_call_args_invalid": "1.2.0",
    "tool_call_protocol_leak_hy": "1.1.0",
    "tool_call_protocol_leak_all_family": "1.1.0",
    "think_protocol_leak_hy": "1.0.0",
    "think_protocol_leak_all_family": "1.1.0",
    "reasoning_selftalk_flood": "0.3.0",
    "zh_ask_foreign_reply": "1.0.0",
}

DETECTOR_NAMES = tuple(_REGISTRY)

_instances: Dict[str, Any] = {}


def get_detector(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; known: {', '.join(DETECTOR_NAMES)}")
    if name not in _instances:
        module, cls = _REGISTRY[name]
        _instances[name] = getattr(importlib.import_module(f"{__name__}.{module}"), cls)()
    return _instances[name]


def detect(
    record: Dict[str, Any],
    detectors: Optional[Iterable[str]] = None,
    sample_id: str = "",
) -> Dict[str, Dict[str, Any]]:
    from ._core.rule_detector import trajectory_view

    view = trajectory_view(record, sample_id)
    return {name: get_detector(name).detect_view(view) for name in (detectors or DETECTOR_NAMES)}


__all__ = ["detect", "get_detector", "DETECTOR_NAMES", "BEHAVIOR_VERSIONS", "SOURCE_COMMIT"]
