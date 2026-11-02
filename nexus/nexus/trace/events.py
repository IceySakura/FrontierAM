from __future__ import annotations

import time

from typing import Literal

from pydantic import BaseModel, Field

from nexus.trace.types import Event, Status
from nexus.utils.common import random_urlsafe_id


class SpanEvent(BaseModel):
    type: Literal['START', 'END', 'UPDATE']
    span_id: str = Field(
        default_factory=lambda: random_urlsafe_id(8), description='span id, 8 bytes = 11 base64 encoded chars'
    )
    time_unix_nano: int = Field(default_factory=lambda: int(time.time() * 1e9))


class SpanStartEvent(SpanEvent):
    type: Literal['START'] = 'START'
    parent_span_id: str
    trace_id: str = Field(
        default_factory=lambda: random_urlsafe_id(16), description='trace id, 16 bytes = 22 base64 encoded chars'
    )
    name: str
    attributes: dict = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    status: Status = Field(default_factory=Status)


class SpanUpdateEvent(SpanEvent):
    type: Literal['UPDATE'] = 'UPDATE'
    attributes: dict | None = None  # set keys
    events: list[Event] | None = None  # append events
    status: Status | None = None  # set object


class SpanEndEvent(SpanEvent):
    type: Literal['END'] = 'END'
