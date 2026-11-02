"""Frozen minimal trajectory model — the single structure the adapter produces and
the rule consumes. This is the decoupling seam: the adapter is source specific; the
rule depends only on this model, never on a source shape.

Trimmed to the fields the online_answer_fetch path uses (ToolCall / Turn / Trajectory
with the bash command + observation + outcome fields). Stdlib only. Self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolCallStatus(Enum):
    SUCCESS = 'success'
    UNKNOWN = 'unknown'


@dataclass
class ToolCall:
    tool_use_id: str
    tool_name: str  # normalized: Bash / Read / Edit / Write / Submit / ...
    input: dict[str, Any]  # {"command": ..., "file_path": ..., "old_str": ..., "new_str": ...}

    result: str | None = None  # tool observation / output
    status: ToolCallStatus = ToolCallStatus.UNKNOWN


@dataclass
class Turn:
    turn_number: int
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Trajectory:
    id: str
    task_id: str
    scaffold: str = ''  # provenance only; rules never branch on it
    turns: list[Turn] = field(default_factory=list)

    # outcome + final artifact
    repo: str | None = None  # e.g. "django/django" — package under fix
    base_commit: str | None = None  # the commit the work tree starts at; anchors history checks
    # Measured in the runtime before the agent starts: True when no ref reaches outside
    # base_commit's ancestry. None when it could not be settled, which is what every
    # caller that does not measure it gets.
    history_truncated: bool | None = None
    resolved: bool | None = None  # did the task pass grading
    score: float | None = None
    final_patch: str = ''  # submitted unified diff (may be "")

    metadata: dict[str, Any] = field(default_factory=dict)
