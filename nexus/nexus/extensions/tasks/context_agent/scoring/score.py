"""Internal scoring protocol shared by context-agent Tasks.

Public Task output fields stay unchanged. Scorers may return this mapping
(or a superset); Tasks copy the keys they already expose.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class ScoreResult(TypedDict, total=False):
    score: float | None
    rationale: str | None
    requirement_status: list[Any]
    judge_unavailable: bool
    error: str | None
    ok: bool
    final_score: float | None
    final_score_ori: Any
    binary: int
    error_msg: str
    reward_error: bool
    raw: Any
    request_id: str
    request_ids: list[str]


class Scorer(Protocol):
    async def __call__(self, *args: Any, **kwargs: Any) -> ScoreResult: ...
