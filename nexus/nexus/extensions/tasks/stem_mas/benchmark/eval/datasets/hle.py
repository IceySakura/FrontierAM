"""HLE source dataset adapter for stem_mas benchmark evaluation."""

from __future__ import annotations

import importlib
import json
import math

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from nexus.dataset import DatasetItem
from nexus.extensions.tasks.stem_mas.benchmark.eval.dataset_items import build_stem_mas_item
from nexus.extensions.tasks.stem_mas.benchmark.eval.datasets.base import BenchmarkDatasetAdapter


def _is_blank(value: Any) -> bool:
    return value is None or value == '' or (isinstance(value, float) and math.isnan(value))


class HLEDataset(BenchmarkDatasetAdapter):
    """Convert HLE source rows into portable Nexus DatasetItems."""

    dataset_names: ClassVar[tuple[str, ...]] = ('hle', 'hle_200')

    data_source: str = 'hle_200'
    default_ability: str = 'stem'

    def iter_items(self) -> Iterator[DatasetItem]:
        records = self.load_source_records(Path(self.path))
        for index, record in enumerate(records):
            yield self.to_dataset_item(record, index)

    def to_dataset_item(self, row: dict[str, Any], index: int) -> DatasetItem:
        question = self.required_text(row, 'origin_question', index)
        answer = self.required_text(row, 'ref_answer', index)
        task_id = row.get('id')
        if _is_blank(task_id):
            task_id = index
        ability = self.optional_text(row.get('subject')) or self.default_ability

        item_id = str(task_id)
        return build_stem_mas_item(
            item_id=item_id,
            dataset_name=self.name or self.data_source,
            source_index=index,
            question=question,
            prompt=[{'role': 'user', 'content': question}],
            answer=answer,
            category=ability,
            metadata={
                'subject': row.get('subject'),
                'source_format': 'hle',
            },
        )

    @classmethod
    def load_source_records(cls, path: Path) -> list[dict[str, Any]]:
        if cls.looks_like_parquet(path):
            return cls.load_parquet_records(path)
        return cls.load_jsonl_records(path)

    @staticmethod
    def looks_like_parquet(path: Path) -> bool:
        if path.is_dir():
            data_dir = path / 'data'
            return any(data_dir.glob('*.parquet')) if data_dir.exists() else any(path.glob('*.parquet'))
        return path.suffix == '.parquet'

    @staticmethod
    def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
        paths = sorted(path.glob('*.jsonl')) if path.is_dir() else [path]
        if not paths:
            raise FileNotFoundError(f'No HLE JSONL files found in {path}.')

        records: list[dict[str, Any]] = []
        for item_path in paths:
            with item_path.open('r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'Invalid HLE JSONL line {line_number} in {item_path}: {exc}') from exc
                    if not isinstance(data, dict):
                        raise ValueError(f'Invalid HLE JSONL line {line_number} in {item_path}: expected object.')
                    records.append(data)
        return records

    @staticmethod
    def load_parquet_records(path: Path) -> list[dict[str, Any]]:
        pd = importlib.import_module('pandas')

        if path.is_dir():
            data_dir = path / 'data'
            paths = sorted(data_dir.glob('*.parquet')) if data_dir.exists() else []
            if not paths:
                paths = sorted(path.glob('*.parquet'))
        else:
            paths = [path]

        if not paths:
            raise FileNotFoundError(f'No HLE parquet files found in {path}.')

        frames = [pd.read_parquet(item_path) for item_path in paths]
        df = pd.concat(frames, ignore_index=True)
        return df.to_dict(orient='records')

    @classmethod
    def required_text(cls, row: dict[str, Any], key: str, index: int) -> str:
        value = row.get(key)
        if _is_blank(value):
            raise ValueError(f'HLE row {index} missing required field {key!r}.')
        return cls.optional_text(value) or ''

    @staticmethod
    def optional_text(value: Any) -> str | None:
        if _is_blank(value):
            return None
        return value if isinstance(value, str) else str(value)


__all__ = ['HLEDataset']
