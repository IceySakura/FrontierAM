from __future__ import annotations

from enum import Enum


class BenchmarkServerMode(str, Enum):
    API = 'api'
    MOCK = 'mock'
