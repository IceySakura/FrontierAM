"""Task harness models: a TaskResult's execution fields must agree with each other."""

from __future__ import annotations

import pytest

from pydantic import ValidationError

from nexus.extensions.projects.hy_live.coordination.models import TaskResult


@pytest.mark.parametrize(
    'payload',
    [
        {
            'execution_status': 'success',
            'failure_kind': 'execution',
            'error': {'code': 'UNEXPECTED', 'message': 'success cannot carry failure'},
        },
        {
            'execution_status': 'failed',
            'failure_kind': 'execution',
        },
        {
            'execution_status': 'timeout',
            'failure_kind': 'execution',
            'error': {'code': 'WORK_AGENT_TIMEOUT', 'message': 'timed out'},
        },
        {
            'execution_status': 'cancelled',
            'failure_kind': 'cancel',
        },
    ],
)
def test_task_result_rejects_inconsistent_execution_fields(payload: dict) -> None:
    with pytest.raises(ValidationError):
        TaskResult.model_validate(payload)
