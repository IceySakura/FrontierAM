"""Opt-in runner-test acceptance; this test must execute inside the target Job."""

from __future__ import annotations

import argparse
import json
import os

from pathlib import Path

import pytest

from tests.e2e.distributed.run_single_trace_api import Acceptance, run_cancellable, validate_target_job


pytestmark = pytest.mark.e2e


async def test_runner_test_single_trace_read_api():
    path = os.environ.get('NEXUS_TRACE_API_E2E_CONFIG')
    if not path:
        pytest.skip('set NEXUS_TRACE_API_E2E_CONFIG inside the authorized runner-test Job')
    validate_target_job()
    values = json.loads(Path(path).read_text())
    preview = values.pop('preview', False)
    assert type(preview) is bool
    args = argparse.Namespace(**{name: Path(value) for name, value in values.items()}, preview=preview)
    await run_cancellable(Acceptance(args))
