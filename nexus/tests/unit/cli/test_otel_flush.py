from __future__ import annotations

import argparse

import pytest

from nexus.cli.commands import run as run_command
from nexus.runner import compose as compose_module
from nexus.runner import single as single_runner


def test_single_cli_relies_on_sdk_atexit_when_run_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(_run_input):  # noqa: ANN001, ANN202
        raise RuntimeError('task failed')

    monkeypatch.setattr(
        compose_module,
        'compile_single_file',
        lambda _path, item_id=None, split=None: single_runner.RunInput(task_name='unit'),
    )
    monkeypatch.setattr(single_runner, 'run', fail)
    monkeypatch.setattr(run_command, 'setup_logging', lambda **_kwargs: None)
    monkeypatch.setattr(
        run_command,
        'force_flush_best_effort',
        lambda: (_ for _ in ()).throw(AssertionError('run-single shutdown must rely on the SDK atexit handler')),
        raising=False,
    )
    args = argparse.Namespace(
        config='unit.yaml',
        data=None,
        id=None,
        hold=None,
        output=None,
        set_json_values=None,
        set_values=None,
        split=None,
        use_parts=None,
        verbose=False,
    )

    with pytest.raises(RuntimeError, match='task failed'):
        run_command.run_single(args)
