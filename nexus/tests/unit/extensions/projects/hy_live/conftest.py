from __future__ import annotations

import importlib.util

from pathlib import Path
from typing import Any

import pytest


if importlib.util.find_spec('hylive') is None:
    # hy_live runs on the hylive package, installed on demand by `nexus modules install hy-live`.
    collect_ignore_glob = ['test_*.py']


class FakeRuntime:
    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.emitted: list[tuple[str, dict[str, Any]]] = []

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        self.emitted.append((event, data))


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()
