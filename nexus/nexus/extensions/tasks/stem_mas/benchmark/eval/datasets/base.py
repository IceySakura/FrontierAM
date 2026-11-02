"""Dataset adapter base classes for stem_mas benchmark evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, model_validator

from nexus.dataset import DatasetItem


_DATASET_ADAPTERS: dict[str, type[BenchmarkDatasetAdapter]] = {}


class BenchmarkDatasetAdapter(BaseModel):
    """Convert an upstream benchmark source into portable Nexus DatasetItems."""

    dataset_names: ClassVar[tuple[str, ...]] = ()

    path: str | Path
    name: str = 'stem_mas'

    model_config = ConfigDict(extra='forbid')

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for dataset_name in cls.dataset_names:
            if not dataset_name:
                continue
            if dataset_name in _DATASET_ADAPTERS:
                registered = _DATASET_ADAPTERS[dataset_name]
                raise ValueError(
                    f'Duplicate dataset adapter {dataset_name!r}: '
                    f'{registered.__module__}.{registered.__name__} and {cls.__module__}.{cls.__name__}'
                )
            _DATASET_ADAPTERS[dataset_name] = cls

    @model_validator(mode='before')
    @classmethod
    def from_path(cls, data: Any) -> Any:
        if isinstance(data, (str, Path)):
            return {'path': data}
        return data

    def iter_items(self) -> Iterator[DatasetItem]:
        raise NotImplementedError

    def load_items(self, *, limit: int | None = None) -> list[DatasetItem]:
        items = self.iter_items()
        return list(islice(items, limit)) if limit is not None else list(items)


def build_dataset_adapter(name: str, path: str | Path, **options: Any) -> BenchmarkDatasetAdapter:
    dataset_cls = _DATASET_ADAPTERS.get(name)
    if dataset_cls is None:
        known = ', '.join(sorted(_DATASET_ADAPTERS))
        raise ValueError(f'Unknown dataset {name!r}. Known datasets: {known}.')
    dataset = dataset_cls(path=path, name=name, **options)
    if not isinstance(dataset, BenchmarkDatasetAdapter):
        raise TypeError(f'{dataset_cls.__module__}.{dataset_cls.__name__} must inherit BenchmarkDatasetAdapter.')
    return dataset


__all__ = [
    'BenchmarkDatasetAdapter',
    'build_dataset_adapter',
]
