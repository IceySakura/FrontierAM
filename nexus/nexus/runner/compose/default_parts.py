"""Built-in run parts owned by the compose package."""

# /// script
# dependencies = []
#
# [project]
# name = "compose_default_parts"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# run_parts = { worker_venv = "nexus.runner.compose.default_parts" }
# ///

from __future__ import annotations

from typing import Any

from nexus.runner.compose import run_part


@run_part('worker_venv')
def worker_venv_run_part(params: dict[str, Any]) -> dict[str, Any]:
    if params:
        raise ValueError(f'Run part worker_venv does not accept parameters: {sorted(params)}')
    return {'runtime_configs': {'main': {'worker_venv': True}}}
