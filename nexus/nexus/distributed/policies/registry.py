from __future__ import annotations

import importlib

from typing import Any, Protocol

from pydantic import BaseModel

from nexus.core.registry import Registry
from nexus.distributed.policies.models import PolicyInput, PolicyResult, SignalObservation


class Policy(Protocol):
    kind: str
    decision_model: type[BaseModel]

    def evaluate(self, inputs: PolicyInput, previous_state: dict[str, Any]) -> PolicyResult: ...


class SignalSource(Protocol):
    async def read(self, config: dict[str, Any], *, max_age_seconds: float) -> SignalObservation: ...


policy_registry = Registry[type[Policy]]('distributed_policy')
signal_source_registry = Registry[type[SignalSource]]('distributed_signal_source')


def load_plugins(imports: list[str]) -> None:
    importlib.import_module('nexus.distributed.policies.collections.infer_proxy')
    importlib.import_module('nexus.distributed.policies.collections.infer_proxy_request_rate')
    for module in imports:
        importlib.import_module(module)
