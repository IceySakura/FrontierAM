"""Running an evaluator in a sandbox of its own.

An evaluator in ``evaluators/`` takes a prepared runtime and reports a result. This
module is what prepares one: it acquires the evaluation runtime, puts the repo at
the base commit, lets the dataset handler install whatever the official script
needs, and releases the runtime whether or not the evaluator finished.

The evaluation runtime is separate from the one the worker used, because the
worker's sandbox holds its edits and the official tests have to run against a clean
checkout plus the extracted patch.
"""

import asyncio
import contextlib
import logging

from typing import Any

from nexus.core import require_context
from nexus.extensions.tasks.hitl.eval.evaluators.unit_test import run_swe_unit_test_eval
from nexus.extensions.tasks.hitl.eval.result import EvaluationResult
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)


def eval_script_of(dataset_handler: Any, instance: dict[str, Any]) -> str:
    """The instance's official evaluation script.

    Handlers that build the script themselves implement ``get_eval_script``; the
    rest carry it in ``spec_dict``.
    """
    script = dataset_handler.get_eval_script(instance)
    if script:
        return str(script)
    spec_dict = instance.get('spec_dict') or {}
    if isinstance(spec_dict, dict):
        return str(spec_dict.get('run_script.sh') or spec_dict.get('test_cmd') or '')
    return ''


async def run_unit_eval_in_sandbox(
    *,
    instance: dict[str, Any],
    instance_id: str,
    patch_text: str,
    dataset_handler: Any,
    configure_params: dict[str, Any],
    runtime_name: str,
    working_dir: str,
    base_commit: str,
    init_timeout_seconds: int,
    eval_timeout_seconds: int,
) -> EvaluationResult:
    """Run the official unit tests against ``patch_text`` in a fresh sandbox.

    Reports a failed :class:`EvaluationResult` rather than raising, so a broken
    evaluation scores zero instead of discarding the run that produced the patch.
    """
    ctx = require_context()
    provider = ctx.runtime_provider
    assert provider is not None, 'runtime provider is not configured'
    provider.configure(runtime_name, **configure_params)
    eval_runtime = await ctx.get_runtime(runtime_name)
    try:
        if base_commit:
            await eval_runtime.call_executor(
                'run_command',
                {'command': f'git checkout {base_commit}', 'cwd': working_dir},
            )
        async with async_timeout(init_timeout_seconds):
            await dataset_handler.eval_runtime_init(instance, eval_runtime)
        async with async_timeout(eval_timeout_seconds):
            return await run_swe_unit_test_eval(
                eval_runtime,
                patch_text=patch_text,
                instance=instance,
                instance_id=instance_id,
                timeout_seconds=eval_timeout_seconds,
            )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        return EvaluationResult(name='unit_test', completed=False, score=0.0, error=f'Timed out: {exc}')
    except Exception as exc:
        logger.exception('[hitl/eval] unit_test evaluator failed')
        return EvaluationResult(
            name='unit_test',
            completed=False,
            score=0.0,
            error=f'{type(exc).__name__}: {exc}',
        )
    finally:
        with contextlib.suppress(Exception):
            await ctx.runtime_provider.release(name=runtime_name)


__all__ = ['eval_script_of', 'run_unit_eval_in_sandbox']
