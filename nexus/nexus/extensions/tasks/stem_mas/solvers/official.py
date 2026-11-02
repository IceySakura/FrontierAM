"""Official solver pipeline adapters.

The benchmark package owns dataset normalization and verification. Official
solver implementations are benchmark-specific algorithms that may live in
external repos, so this module defines a small JSON command protocol instead of
embedding a particular prover implementation in the task runtime.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import signal

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class OfficialSolverRequest:
    pipeline: str
    benchmark: str
    task_id: str
    question: str
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            'pipeline': self.pipeline,
            'benchmark': self.benchmark,
            'task_id': self.task_id,
            'question': self.question,
            'reference': self.reference,
            'metadata': self.metadata,
            'extra': self.extra,
        }


@dataclass(frozen=True)
class OfficialSolverResult:
    prediction: str
    raw: dict[str, Any]
    stderr: str = ''


class OfficialSolverPipeline:
    name: ClassVar[str] = ''
    benchmark_names: ClassVar[tuple[str, ...]] = ()
    command_env: ClassVar[str] = ''
    registry: ClassVar[dict[str, type[OfficialSolverPipeline]]] = {}
    benchmark_registry: ClassVar[dict[str, type[OfficialSolverPipeline]]] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if cls.name:
            if cls.name in OfficialSolverPipeline.registry:
                raise ValueError(f'Duplicate official solver pipeline: {cls.name}')
            OfficialSolverPipeline.registry[cls.name] = cls
        for benchmark_name in cls.benchmark_names:
            normalized_benchmark = benchmark_name.strip().lower().replace('-', '_')
            if normalized_benchmark in OfficialSolverPipeline.benchmark_registry:
                registered = OfficialSolverPipeline.benchmark_registry[normalized_benchmark]
                raise ValueError(
                    f'Duplicate official solver benchmark {benchmark_name!r}: '
                    f'{registered.__module__}.{registered.__name__} and {cls.__module__}.{cls.__name__}'
                )
            OfficialSolverPipeline.benchmark_registry[normalized_benchmark] = cls

    async def solve(
        self, request: OfficialSolverRequest, command: str | None, timeout_seconds: int
    ) -> OfficialSolverResult:
        resolved_command = self.resolve_command(command)
        if not resolved_command:
            raise ValueError(
                f'official solver pipeline {self.name!r} requires either task_config.official_solver_command, '
                f'or {self.command_env}.'
            )

        stdout, stderr = await self.run_command(resolved_command, request.payload(), timeout_seconds)
        raw = self.parse_output(stdout)
        return OfficialSolverResult(prediction=raw['candidate'], raw=raw, stderr=stderr)

    def resolve_command(self, command: str | None) -> str:
        return command or os.getenv(self.command_env, '')

    @staticmethod
    async def run_command(command: str, payload: dict[str, Any], timeout_seconds: int) -> tuple[str, str]:
        argv = shlex.split(command)
        if not argv:
            raise ValueError('official solver command is empty.')
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        input_bytes = json.dumps(payload, ensure_ascii=False).encode()
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(input=input_bytes), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            raise TimeoutError(f'official solver command timed out after {timeout_seconds}s: {command}') from exc

        stdout_text = stdout.decode(errors='replace')
        stderr_text = stderr.decode(errors='replace')
        if process.returncode != 0:
            raise RuntimeError(
                f'official solver command failed with exit code {process.returncode}: {command}\n{stderr_text}'
            )
        return stdout_text, stderr_text

    @staticmethod
    def parse_output(stdout: str) -> dict[str, Any]:
        text = stdout.strip()
        if not text:
            raise ValueError('official solver command produced empty stdout.')
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError('official solver stdout must be one JSON object.') from exc
        if not isinstance(parsed, dict) or set(parsed) != {'candidate'}:
            raise ValueError('official solver stdout must be exactly {"candidate": "<string>"}.')
        candidate = parsed['candidate']
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError('official solver candidate must be a non-empty string.')
        return {'candidate': candidate.strip()}


class MiniF2FOfficialPipeline(OfficialSolverPipeline):
    name = 'minif2f'
    benchmark_names = ('minif2f',)
    command_env = 'STEM_MAS_MINIF2F_OFFICIAL_SOLVER_COMMAND'


class PutnamBenchOfficialPipeline(OfficialSolverPipeline):
    name = 'putnambench'
    benchmark_names = ('putnam672', 'putnam-2025', 'putnambench')
    command_env = 'STEM_MAS_PUTNAMBENCH_OFFICIAL_SOLVER_COMMAND'


class FateEvalOfficialPipeline(OfficialSolverPipeline):
    name = 'fate_eval'
    benchmark_names = ('fatem', 'fateh', 'fatex', 'fate-eval')
    command_env = 'STEM_MAS_FATE_EVAL_OFFICIAL_SOLVER_COMMAND'


def normalize_pipeline_name(value: str) -> str:
    return value.strip().lower().replace('-', '_')


def resolve_pipeline(pipeline: str | None, benchmark: str) -> OfficialSolverPipeline:
    if pipeline:
        normalized = normalize_pipeline_name(pipeline)
        if normalized in OfficialSolverPipeline.registry:
            return OfficialSolverPipeline.registry[normalized]()
        known = ', '.join(sorted(OfficialSolverPipeline.registry))
        raise ValueError(f'Unknown official solver pipeline {pipeline!r}. Known pipelines: {known}.')

    normalized_benchmark = normalize_pipeline_name(benchmark)
    if normalized_benchmark in OfficialSolverPipeline.benchmark_registry:
        return OfficialSolverPipeline.benchmark_registry[normalized_benchmark]()
    known = ', '.join(sorted(OfficialSolverPipeline.benchmark_registry))
    raise ValueError(f'Cannot infer official solver pipeline from benchmark {benchmark!r}. Known benchmarks: {known}.')


async def solve_with_official_pipeline(
    *,
    pipeline: str | None,
    benchmark: str,
    task_id: str,
    question: str,
    reference: str,
    metadata: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    command: str | None = None,
    timeout_seconds: int = 1800,
) -> OfficialSolverResult:
    solver = resolve_pipeline(pipeline, benchmark)
    request = OfficialSolverRequest(
        pipeline=solver.name,
        benchmark=benchmark,
        task_id=task_id,
        question=question,
        reference=reference,
        metadata=metadata or {},
        extra=extra or {},
    )
    return await solver.solve(request, command=command, timeout_seconds=timeout_seconds)


__all__ = [
    'FateEvalOfficialPipeline',
    'MiniF2FOfficialPipeline',
    'OfficialSolverPipeline',
    'OfficialSolverRequest',
    'OfficialSolverResult',
    'PutnamBenchOfficialPipeline',
    'resolve_pipeline',
    'solve_with_official_pipeline',
]
