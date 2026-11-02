"""Registered run-compose profiles for STEM-MAS benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.stem_mas.benchmark.eval.profile import BenchmarkProfile
from nexus.runner.compose import RunPart, run_part


CONFIG_ROOT = Path(__file__).resolve().parent / 'benchmark' / 'eval' / 'config'


@run_part('stem_mas_benchmark')
def stem_mas_benchmark_run_part(params: dict[str, Any]) -> RunPart:
    """Load a checked-in benchmark profile as a reusable partial RunInput."""

    unexpected = sorted(set(params) - {'official_pipeline', 'profile'})
    if unexpected:
        raise ValueError(f'stem_mas_benchmark does not accept parameters: {unexpected}')
    profile = str(params.get('profile') or '').strip()
    if not profile:
        raise ValueError('stem_mas_benchmark requires profile=<path-under-eval/config>')

    path = _profile_path(profile)
    benchmark_profile = BenchmarkProfile.load(path)
    official_pipeline = str(params.get('official_pipeline') or '').strip() or None
    dataset_name = str(benchmark_profile.dataset_config.get('name') or '').strip() or None
    config = benchmark_profile.build_run_config(
        dataset_name=dataset_name,
        official_pipeline=official_pipeline,
        configure_cache_metrics=True,
    )

    return RunPart(
        body=config,
        merge={'imports': 'append_unique'},
        source=f'stem_mas_benchmark:{profile}',
    )


def _profile_path(profile: str) -> Path:
    relative = Path(profile)
    if relative.is_absolute():
        raise ValueError('stem_mas_benchmark profile must be relative to the benchmark config directory')
    if relative.suffix == '':
        relative = relative.with_suffix('.yaml')
    if relative.suffix not in {'.yaml', '.yml'}:
        raise ValueError('stem_mas_benchmark profile must be a YAML file')

    root = CONFIG_ROOT.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError('stem_mas_benchmark profile must stay under the benchmark config directory')
    if not path.is_file():
        raise FileNotFoundError(f'STEM-MAS benchmark profile not found: {path}')
    return path


__all__ = ['stem_mas_benchmark_run_part']
