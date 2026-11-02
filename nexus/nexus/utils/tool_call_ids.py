"""Trajectory-local tool IDs with an opt-in deterministic parity mode."""

import os
import uuid


class ToolCallIdGenerator:
    def __init__(self) -> None:
        self._deterministic = os.environ.get('NEXUS_DETERMINISTIC_TOOL_CALL_IDS', '') == '1'
        self._next_sequence = 0

    def next_id(self) -> str:
        if not self._deterministic:
            return f'call_{uuid.uuid4().hex[:24]}'
        sequence = self._next_sequence
        self._next_sequence += 1
        return f'call_{sequence:024x}'
