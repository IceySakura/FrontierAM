#!/usr/bin/env python3
"""Protocol-test adapter for running Codex with the baseline SWE generator.

The adapter supplies an explicit Codex RunInput for the real-SWE comparison.
It defaults to the direct API client and selects the Codex Gateway policy with
``SWE_USE_GATEWAY``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe import generator as baseline_generator


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value if value else None


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_openai_clients(*, use_gateway: bool) -> dict[str, dict[str, Any]]:
    direct_client = {
        "name": os.getenv("OPENAI_CLIENT_NAME", "openai"),
        "config": {
            "client_args": {
                "api_key": '${oc.env:OPENAI_API_KEY,""}',
                "base_url": '${oc.env:OPENAI_BASE_URL,""}',
            },
            "request_args": {"model": '${oc.env:OPENAI_MODEL,""}'},
        },
    }
    if not use_gateway:
        return {"main": direct_client}
    return {
        "main": {
            "name": "gateway",
            "config": {
                "client_args": {
                    "scaffold_policy": "codex",
                    "openai_client": direct_client,
                }
            },
        }
    }


def _build_codex_agent_config() -> dict:
    """Build only the version-neutral options unless a lane opts in explicitly."""
    config = {
        "working_dir": os.getenv("CODEX_WORKING_DIR", "/testbed"),
        "timeout": float(os.getenv("CODEX_TIMEOUT", "3600")),
        "version": os.getenv("CODEX_VERSION", "0.114.0"),
        "model": os.getenv("CODEX_MODEL") or os.getenv("OPENAI_MODEL"),
        "use_azure_provider": False,
        "developer_instructions": _optional_env("CODEX_DEVELOPER_INSTRUCTIONS"),
        "extra_cli_args": json.loads(os.getenv("CODEX_EXTRA_CLI_ARGS_JSON", "[]")),
        "retain_json_events_stdout": True,
    }
    optional_integer_fields = {
        "auto_compact_token_limit": "CODEX_AUTO_COMPACT_TOKEN_LIMIT",
        "model_context_window": "CODEX_MODEL_CONTEXT_WINDOW",
    }
    for field_name, environment_name in optional_integer_fields.items():
        value = _optional_env(environment_name)
        if value is not None:
            config[field_name] = int(value)
    optional_string_fields = {
        "reasoning_effort": "CODEX_REASONING_EFFORT",
        "auto_compact_token_limit_scope": "CODEX_AUTO_COMPACT_TOKEN_LIMIT_SCOPE",
        "compact_prompt": "CODEX_COMPACT_PROMPT",
    }
    for field_name, environment_name in optional_string_fields.items():
        value = _optional_env(environment_name)
        if value is not None:
            config[field_name] = value
    return config


def _apply_optional_runtime_settings(config: Any) -> None:
    purpose = _optional_env("GONGFENG_PURPOSE")
    if purpose is not None:
        config.runtime_provider_config["purpose"] = purpose


def _enable_current_nexus_on_main_runtime(config: Any) -> None:
    config.runtime_configs.setdefault("main", {})["worker_venv"] = True


def generate_configs() -> Iterator[BatchItem]:
    for item in baseline_generator.generate_configs():
        config = item.config
        if "nexus.extensions.agents.codex" not in config.imports:
            config.imports.append("nexus.extensions.agents.codex")

        config.agent_name = "codex"
        config.agent_config = _build_codex_agent_config()
        use_gateway = _env_bool("SWE_USE_GATEWAY")
        config.openai_clients = _build_openai_clients(use_gateway=use_gateway)
        if use_gateway:
            config.runner.use_gateway = True
        config.runtime_provider_config["token"] = '${oc.env:GONGFENG_RUNTIME_TOKEN,""}'
        _enable_current_nexus_on_main_runtime(config)
        _apply_optional_runtime_settings(config)
        yield item


def get_total_count() -> int:
    return baseline_generator.get_total_count()


def get_batch_metadata() -> dict:
    metadata = baseline_generator.get_batch_metadata()
    metadata.update(
        agent="codex",
        protocol_test_adapter="gateway" if _env_bool("SWE_USE_GATEWAY") else "legacy",
    )
    return metadata
