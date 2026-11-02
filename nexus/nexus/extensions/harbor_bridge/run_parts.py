"""Reusable run-compose part for Harbor Bridge tasks."""

from __future__ import annotations

from typing import Any

from nexus.runner.compose import RunPart, run_part


@run_part('harbor_bridge')
def harbor_bridge_run_part(params: dict[str, Any]) -> RunPart:
    """Select the Harbor Bridge task and its execution mode."""
    unexpected = sorted(set(params) - {'mode'})
    if unexpected:
        raise ValueError(f'harbor_bridge does not accept parameters: {unexpected}')

    mode = str(params.get('mode') or 'agent').strip().lower()
    if mode not in {'agent', 'oracle', 'nop'}:
        raise ValueError('harbor_bridge mode must be agent, oracle, or nop')

    source = f'harbor_bridge:mode={mode}'

    return RunPart(
        source=source,
        body={
            'imports': ['nexus.extensions.harbor_bridge.task'],
            'task_name': 'harbor_bridge_task',
            'task_config': {
                'mode': mode,
            },
        },
        merge={'imports': 'append_unique'},
    )


__all__ = ['harbor_bridge_run_part']
