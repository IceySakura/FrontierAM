import asyncio
import shlex
import sys

from pathlib import Path

import pytest

from nexus.extensions.tasks.stem_mas.solvers.official import (
    PutnamBenchOfficialPipeline,
    resolve_pipeline,
    solve_with_official_pipeline,
)


def test_resolve_pipeline_uses_benchmark_mapping() -> None:
    pipeline = resolve_pipeline(None, 'putnam-2025')

    assert isinstance(pipeline, PutnamBenchOfficialPipeline)


def test_official_solver_command_protocol(tmp_path: Path) -> None:
    script = tmp_path / 'solver.py'
    script.write_text(
        '\n'.join(
            [
                'import json, sys',
                'request = json.load(sys.stdin)',
                "print(json.dumps({'candidate': request['task_id']}))",
            ]
        )
    )

    result = asyncio.run(
        solve_with_official_pipeline(
            pipeline='putnambench',
            benchmark='putnam672',
            task_id='putnam672.putnam_2025_a1',
            question='theorem putnam_2025_a1 : True := by sorry',
            reference='',
            command=shlex.join([sys.executable, str(script)]),
            timeout_seconds=10,
        )
    )

    assert result.prediction == 'putnam672.putnam_2025_a1'
    assert result.raw == {'candidate': 'putnam672.putnam_2025_a1'}


def test_official_solver_rejects_ambiguous_stdout() -> None:
    pipeline = PutnamBenchOfficialPipeline()

    with pytest.raises(ValueError, match='exactly'):
        pipeline.parse_output('{"candidate":"proof","output":"legacy"}')
