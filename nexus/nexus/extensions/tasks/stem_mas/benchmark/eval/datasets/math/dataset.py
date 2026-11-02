"""Formal-math benchmark adapters for STEM-MAS."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from nexus.dataset import DatasetItem
from nexus.extensions.tasks.stem_mas.benchmark.eval.dataset_items import build_stem_mas_item
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets.base import BenchmarkDatasetAdapter
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets.math.profiles import build_math_prompt_profile
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets.math.sources import MathSourceLoader
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets.math.specs import MATH_BENCHMARK_SPECS, MathBenchmarkSpec


DEFAULT_LEAN_HEADER = '\n'.join([
    'import Mathlib',
    'import Aesop',
    'set_option maxHeartbeats 0',
    'open BigOperators Real Nat Topology Rat',
])


class MathDataset(BenchmarkDatasetAdapter):
    """Convert an official formal-math source into portable Nexus DatasetItems."""

    dataset_names: ClassVar[tuple[str, ...]] = tuple(MATH_BENCHMARK_SPECS)

    def iter_items(self) -> Iterator[DatasetItem]:
        for index, row in enumerate(self.load_records()):
            yield self.to_dataset_item(row, index)

    def load_records(self) -> list[dict[str, Any]]:
        if str(self.path).strip():
            return MathSourceLoader.load_source_records(Path(self.path))

        spec = self.benchmark_spec()
        if spec.hf_id and spec.hf_file:
            rows = MathSourceLoader.load_hf_file_records(spec.hf_id, spec.hf_file, spec.hf_revision)
        elif spec.source_url:
            rows = MathSourceLoader.load_url_records(spec.source_url)
        else:
            raise FileNotFoundError(f'{self.name} has no official source configured.')
        if not rows:
            raise ValueError(f'Official source produced no rows for {self.name!r}.')
        if spec.included_task_ids is not None:
            rows = [row for row in rows if row.get(spec.task_id_field) in spec.included_task_ids]
        return rows

    def benchmark_spec(self) -> MathBenchmarkSpec:
        return MATH_BENCHMARK_SPECS[self.name]

    def to_dataset_item(self, row: dict[str, Any], index: int) -> DatasetItem:
        spec = self.benchmark_spec()
        self.validate_source_row(row, index, spec)

        task_id = self.required_text(row, spec.task_id_field, index)
        statement = self.required_text(row, 'formal_statement', index)
        prompt_profile = build_math_prompt_profile(spec.prompt_profile, spec.name)
        question = prompt_profile.build_prompt(
            lean_version=spec.lean_version,
            header=self.lean_header(row, spec),
            statement=statement,
        )
        prompt = [
            {'role': 'system', 'content': prompt_profile.system_prompt()},
            {'role': 'user', 'content': question},
        ]
        verifier_task_id = self.verifier_task_id(spec, task_id)
        task_fields = {
            'evaluator_type': 'lean4_verifier',
            'lean4_verifier_benchmark': spec.lean4_verifier_benchmark,
            'lean4_verifier_manifest': spec.lean4_verifier_manifest,
            'lean4_verifier_candidate_kind': 'proof_body',
            'lean4_verifier_candidate_policy': spec.candidate_policy,
            'lean4_verifier_task_id': verifier_task_id,
        }
        metadata = {
            **self.source_metadata(row, spec),
            **self.remote_source_metadata(spec),
        }

        return build_stem_mas_item(
            item_id=task_id,
            dataset_name=self.name,
            source_index=index,
            question=question,
            prompt=prompt,
            answer=statement,
            category='mathematics',
            task_fields=task_fields,
            metadata=metadata,
        )

    def validate_source_row(self, row: dict[str, Any], index: int, spec: MathBenchmarkSpec) -> None:
        unsupported = set(row) - set(spec.source_fields)
        if unsupported:
            fields = ', '.join(sorted(unsupported))
            raise ValueError(f'{self.name} row {index} contains unsupported source fields: {fields}.')
        self.required_text(row, spec.task_id_field, index)
        self.required_text(row, 'formal_statement', index)

    def required_text(self, row: dict[str, Any], field: str, index: int) -> str:
        value = row.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f'{self.name} row {index} is missing required source field {field!r}.')
        return str(value).strip()

    @staticmethod
    def source_metadata(row: dict[str, Any], spec: MathBenchmarkSpec) -> dict[str, Any]:
        return {field: row[field] for field in spec.metadata_fields if field in row}

    def remote_source_metadata(self, spec: MathBenchmarkSpec) -> dict[str, Any]:
        if str(self.path).strip():
            return {}
        if spec.hf_id:
            return {'hf_id': spec.hf_id, 'hf_file': spec.hf_file, 'hf_revision': spec.hf_revision}
        return {'source_url': spec.source_url}

    @staticmethod
    def verifier_task_id(spec: MathBenchmarkSpec, task_id: str) -> str:
        benchmark = spec.lean4_verifier_benchmark
        if task_id.startswith(f'{benchmark}.'):
            return task_id
        if spec.lean4_verifier_task_id_zero_pad_width and task_id.isdigit():
            width = spec.lean4_verifier_task_id_zero_pad_width
            return f'{benchmark}.{int(task_id):0{width}d}'
        return f'{benchmark}.{task_id}'

    def lean_header(self, row: dict[str, Any], spec: MathBenchmarkSpec) -> str:
        if spec.header_field:
            value = row.get(spec.header_field)
            if value is not None and (not isinstance(value, str) or value.strip()):
                return str(value).strip()
        return DEFAULT_LEAN_HEADER


__all__ = ['MathDataset']
