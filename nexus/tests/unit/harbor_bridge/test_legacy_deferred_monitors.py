from __future__ import annotations

from nexus.extensions.harbor_bridge.bad_pattern_operator.legacy import (
    detect_str_replace_old_eq_new,
    legacy_deferred_metrics,
    run_legacy_deferred_monitors,
)
from nexus.extensions.projects.code_mt.task.reward_hacking._vendored_swe.monitors import run_swe_monitors


def test_legacy_detector_outputs_match_existing_swe_contract() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "中文说明 " * 20 + " the result is ready and it is available for you " * 5,
            "reasoning_content": "<tool_call>leaked</tool_call>",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "str_replace",
                        "arguments": {
                            "old_string": "same",
                            "new_string": "same",
                        },
                    },
                }
            ],
        }
    ]
    expected = run_swe_monitors(messages, {"messages": messages})
    actual = run_legacy_deferred_monitors(
        messages,
        enabled_families={"codeswitch", "str_replace_old_eq_new", "output_format_bad_pattern"},
        extra_fields={"messages": messages},
    )

    assert actual["codeswitch"] == expected["codeswitch"]
    assert actual["output_format_bad_pattern"] == expected["output_format_bad_pattern"]
    assert actual["str_replace_old_eq_new"]["str_replace_count"] == 1
    assert actual["str_replace_old_eq_new"]["old_eq_new_count"] == 1
    assert actual["str_replace_old_eq_new"]["noop_edit_count"] == 1
    assert actual["str_replace_old_eq_new"]["edit_call_failure_count"] == 0
    assert actual["str_replace_old_eq_new"]["edit_call_never_recovered_count"] == 0


def test_legacy_str_replace_family_accepts_hycode_edit_noop() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "Edit",
                        "arguments": {
                            "file_path": "/tmp/example.py",
                            "old_string": "same",
                            "new_string": "same",
                        },
                    },
                }
            ],
        }
    ]

    detection = detect_str_replace_old_eq_new({"messages": messages})
    deferred = run_legacy_deferred_monitors(
        messages,
        enabled_families={"str_replace_old_eq_new"},
        extra_fields={"messages": messages},
    )

    assert detection == {"str_replace_count": 1, "old_eq_new_count": 1}
    assert deferred["str_replace_old_eq_new"] == {
        "str_replace_count": 1,
        "old_eq_new_count": 1,
        "noop_edit_count": 1,
        "edit_call_failure_count": 0,
        "edit_call_never_recovered_count": 0,
    }


def test_legacy_str_replace_family_accepts_serialized_hycode_edit_noop() -> None:
    serialized = (
        "<tool_calls:6124c78e><tool_call:6124c78e>Edit"
        "<arg_key:6124c78e>file_path</arg_key:6124c78e>"
        "<arg_value:6124c78e>/tmp/example.py</arg_value:6124c78e>"
        "<arg_key:6124c78e>old_string</arg_key:6124c78e>"
        "<arg_value:6124c78e>same</arg_value:6124c78e>"
        "<arg_key:6124c78e>new_string</arg_key:6124c78e>"
        "<arg_value:6124c78e>same</arg_value:6124c78e>"
        "</tool_call:6124c78e></tool_calls:6124c78e>"
    )

    assert detect_str_replace_old_eq_new({"agent_timeline": serialized}) == {
        "str_replace_count": 1,
        "old_eq_new_count": 1,
    }


def test_legacy_str_replace_family_ignores_serialized_calls_without_tool_name() -> None:
    serialized_values = (
        "edit <tool_call></tool_call>",
        (
            "edit <tool_call:6124c78e>"
            "<arg_key:6124c78e>old_string</arg_key:6124c78e>"
            "<arg_value:6124c78e>same</arg_value:6124c78e>"
            "</tool_call:6124c78e>"
        ),
    )

    for serialized in serialized_values:
        assert detect_str_replace_old_eq_new({"agent_timeline": serialized}) == {
            "str_replace_count": 0,
            "old_eq_new_count": 0,
        }


def test_legacy_str_replace_family_folds_hycode_edit_failures_into_deferred_hit() -> None:
    messages = []
    for index, old_string in enumerate(("first", "second"), start=1):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": "Edit",
                                "arguments": {
                                    "file_path": "/tmp/example.py",
                                    "old_string": old_string,
                                    "new_string": "replacement",
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{index}",
                    "content": "String to replace not found in file.",
                    "is_error": True,
                },
            ]
        )

    result = run_legacy_deferred_monitors(
        messages,
        enabled_families={"str_replace_old_eq_new"},
        extra_fields={"messages": messages},
    )["str_replace_old_eq_new"]

    assert result["noop_edit_count"] == 0
    assert result["edit_call_failure_count"] == 1
    assert result["edit_call_never_recovered_count"] == 1
    assert result["old_eq_new_count"] == 2


def test_legacy_str_replace_family_folds_tool_use_error_protocol_without_metadata() -> None:
    messages = []
    for index in range(2):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": "Edit",
                                "arguments": {
                                    "file_path": "/tmp/example.py",
                                    "old_string": "before",
                                    "new_string": "after",
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{index}",
                    "content": (
                        "<tool_use_error>"
                        "String to replace not found in file."
                        "</tool_use_error>"
                    ),
                },
            ]
        )

    result = run_legacy_deferred_monitors(
        messages,
        enabled_families={"str_replace_old_eq_new"},
        extra_fields={"messages": messages},
    )["str_replace_old_eq_new"]

    assert result["noop_edit_count"] == 0
    assert result["edit_call_failure_count"] == 1
    assert result["edit_call_never_recovered_count"] == 1
    assert result["old_eq_new_count"] == 2


def test_legacy_str_replace_family_recognizes_hycode_unread_file_failure() -> None:
    messages = []
    for index in range(2):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": "Edit",
                                "arguments": {
                                    "file_path": "/tmp/example.py",
                                    "old_string": "before",
                                    "new_string": "after",
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{index}",
                    "content": "File has not been read yet. Read it first.",
                    "is_error": True,
                },
            ]
        )

    result = run_legacy_deferred_monitors(
        messages,
        enabled_families={"str_replace_old_eq_new"},
        extra_fields={"messages": messages},
    )["str_replace_old_eq_new"]

    assert result["edit_call_failure_count"] == 1
    assert result["edit_call_never_recovered_count"] == 1
    assert result["old_eq_new_count"] == 2


def test_legacy_metrics_use_unified_namespace_without_changing_detector_fields() -> None:
    results = {
        "codeswitch": {"is_codeswitch": True},
        "str_replace_old_eq_new": {"str_replace_count": 2, "old_eq_new_count": 1},
        "output_format_bad_pattern": {
            "is_bad_pattern": True,
            "F6_7_count": 1,
            "F7_2_count": 2,
            "F7_3_count": 3,
        },
    }

    metrics = legacy_deferred_metrics(results)

    assert metrics["tool_call_bad_pattern/legacy_codeswitch/hit"] == 1.0
    assert metrics["tool_call_bad_pattern/legacy_str_replace_old_eq_new/count"] == 1.0
    assert metrics["tool_call_bad_pattern/legacy_str_replace_old_eq_new/noop_edit/count"] == 0.0
    assert metrics["tool_call_bad_pattern/legacy_str_replace_old_eq_new/edit_call_failure/count"] == 0.0
    assert metrics["tool_call_bad_pattern/legacy_str_replace_old_eq_new/edit_call_never_recovered/count"] == 0.0
    assert metrics["tool_call_bad_pattern/legacy_output_format_bad_pattern/count"] == 6.0
    assert metrics[
        "tool_call_bad_pattern/legacy_output_format_bad_pattern/F7_3_tool_call_leak_in_content/count"
    ] == 3.0
