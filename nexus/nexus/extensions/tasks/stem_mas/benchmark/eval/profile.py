"""Benchmark profile loading for Nexus Dataset and RunPart execution."""

from __future__ import annotations

import copy

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from nexus.extensions.tasks.stem_mas.benchmark.eval.config_models import CacheMetricsConfig
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets import BenchmarkDatasetAdapter, build_dataset_adapter
from nexus.extensions.tasks.stem_mas.benchmark.eval.providers import BenchmarkModelProvider
from nexus.extensions.tasks.stem_mas.benchmark.eval.runtime_config import BenchmarkRuntimeConfig
from nexus.extensions.tasks.stem_mas.solvers import SolverStrategy, resolve_solver_strategy
from nexus.extensions.tasks.stem_mas.solvers.official import resolve_pipeline


@dataclass(frozen=True)
class BenchmarkProfile:
    """A validated profile split into dataset selection and static run configuration."""

    path: Path
    dataset_config: dict[str, Any]
    run_config: dict[str, Any]
    solver_strategy: SolverStrategy | None
    model_provider: BenchmarkModelProvider | None
    cache_metrics_config: CacheMetricsConfig

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkProfile:
        resolved_path = Path(path)
        raw = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(f'Benchmark profile must contain a mapping: {resolved_path}')
        if not all(isinstance(key, str) for key in raw):
            raise TypeError(f'Benchmark profile keys must be strings: {resolved_path}')
        config = copy.deepcopy(cast(dict[str, Any], raw))

        raw_dataset = config.pop('dataset', {})
        if not isinstance(raw_dataset, dict):
            raise TypeError(f'dataset must be a mapping in {resolved_path}')
        dataset_config = dict(raw_dataset)

        model_provider = BenchmarkRuntimeConfig.model_provider_from_config(config)
        if model_provider is not None:
            BenchmarkRuntimeConfig.apply_model_provider(config, model_provider)
            config.pop(BenchmarkRuntimeConfig.MODEL_PROVIDER_KEY, None)

        solver_strategy = resolve_solver_strategy(config) if 'solver_strategy' in config else None
        config.pop('solver_strategy', None)
        cache_metrics_config = BenchmarkRuntimeConfig.cache_metrics_config(config)

        return cls(
            path=resolved_path,
            dataset_config=dataset_config,
            run_config=config,
            solver_strategy=solver_strategy,
            model_provider=model_provider,
            cache_metrics_config=cache_metrics_config,
        )

    def build_run_config(
        self,
        *,
        dataset_name: str | None = None,
        official_pipeline: str | None = None,
        configure_cache_metrics: bool,
    ) -> dict[str, Any]:
        config = copy.deepcopy(self.run_config)
        self.apply_solver(config, dataset_name=dataset_name, official_pipeline=official_pipeline)
        if configure_cache_metrics and BenchmarkRuntimeConfig.CACHE_METRICS_KEY in config:
            client_metadata = BenchmarkRuntimeConfig.openai_client_metadata(
                config,
                self.cache_metrics_config,
                main_model_provider=self.model_provider,
            )
            BenchmarkRuntimeConfig.ensure_cache_metrics_processor(
                config,
                cache_metrics_config=self.cache_metrics_config,
                client_metadata=client_metadata,
            )
            config.pop(BenchmarkRuntimeConfig.CACHE_METRICS_KEY, None)
        return config

    def build_dataset_adapter(self, input_path: str | Path | None = None) -> BenchmarkDatasetAdapter:
        """Resolve this profile's upstream source adapter without adding execution settings."""

        dataset_config = copy.deepcopy(self.dataset_config)
        configured_name = dataset_config.pop('name', None)
        if configured_name is None or not str(configured_name).strip():
            raise ValueError(f'dataset.name is required in {self.path}')
        dataset_name = str(configured_name).strip()

        configured_path = dataset_config.pop('path', None)
        source_path = input_path if input_path is not None else configured_path
        path = '' if source_path is None else str(source_path).strip()
        return build_dataset_adapter(dataset_name, path, **dataset_config)

    def apply_solver(
        self,
        config: dict[str, Any],
        *,
        dataset_name: str | None,
        official_pipeline: str | None,
    ) -> None:
        strategy = self.solver_strategy
        if strategy is None:
            if official_pipeline:
                raise ValueError('official_pipeline is only valid for an official solver profile')
            return

        task_config = config.setdefault('task_config', {})
        if not isinstance(task_config, dict):
            raise TypeError(f'task_config must be a mapping in {self.path}')
        metadata = task_config.setdefault('metadata', {})
        if not isinstance(metadata, dict):
            raise TypeError(f'task_config.metadata must be a mapping in {self.path}')
        metadata.update(strategy.metadata())

        if not strategy.official_method:
            if official_pipeline:
                raise ValueError('official_pipeline is only valid for an official solver profile')
            return

        pipeline = official_pipeline
        if not pipeline and dataset_name:
            pipeline = resolve_pipeline(None, dataset_name).name
        if not pipeline:
            raise ValueError(
                'Official solver profile requires a dataset or official_pipeline=<minif2f|putnambench|fate_eval>'
            )
        task_config.setdefault('official_solver_pipeline', pipeline)


__all__ = ['BenchmarkProfile']
