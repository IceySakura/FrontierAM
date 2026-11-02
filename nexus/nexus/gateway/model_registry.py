"""Keep registered model dependencies independent of Gateway routing policy."""

from __future__ import annotations

import copy

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from nexus.gateway.processor import Processor


@dataclass(frozen=True)
class ModelBinding:
    """Keep model dependencies fixed while requests copy their own parameters."""

    model_id: str
    processor: Processor
    request_args: Mapping[str, Any]
    max_model_len: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'request_args', MappingProxyType(copy.deepcopy(dict(self.request_args))))


class ModelRegistry:
    """Keep model dependencies and the default ID; look up only explicit IDs."""

    def __init__(self, models: Mapping[str, ModelBinding], *, default_model_id: str) -> None:
        if not isinstance(default_model_id, str) or default_model_id not in models:
            raise ValueError(f'default_model_id must explicitly name one of {sorted(models)}')
        self.models: Mapping[str, ModelBinding] = MappingProxyType(dict(models))
        self.default_model_id = default_model_id

    def get(self, model_id: str) -> ModelBinding:
        if not isinstance(model_id, str) or model_id not in self.models:
            raise ValueError(f'Unknown model ID {model_id!r}; choose one of {sorted(self.models)}')
        return self.models[model_id]
