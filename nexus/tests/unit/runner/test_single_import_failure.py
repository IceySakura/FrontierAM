import pytest

from nexus.runner.single import RunInput, run


@pytest.mark.asyncio
async def test_run_reports_missing_import_module_as_an_initialization_failure() -> None:
    """A module listed in ``imports`` that cannot be loaded must not escape ``run()``.

    Batch callers collect results, not exceptions: anything raised out of ``run()`` leaves
    the run recorded as still executing, with no error and no traceback.
    """
    result = await run(
        RunInput(
            task_name='unused_task_name',
            imports=['nexus_missing_module_for_tests'],
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.type == 'ImportError'
    assert 'nexus_missing_module_for_tests' in result.error.message
    assert 'Traceback' in result.error.traceback


@pytest.mark.asyncio
async def test_run_reports_import_time_exceptions_as_initialization_failures(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract for a module that imports fine but raises while executing."""
    module_name = 'nexus_exploding_module_for_tests'
    (tmp_path / f'{module_name}.py').write_text("raise RuntimeError('boom at import time')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    result = await run(RunInput(task_name='unused_task_name', imports=[module_name]))

    assert result.success is False
    assert result.error is not None
    assert result.error.type == 'RuntimeError'
    assert 'boom at import time' in result.error.message
