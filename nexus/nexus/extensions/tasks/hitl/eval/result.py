"""Recipe-neutral evaluation result types."""

from typing import Any

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """A single evaluator result.

    Recipes decide how to interpret and aggregate these fields. The base type
    intentionally avoids universal meanings like resolved/passed.
    """

    name: str
    completed: bool = False
    score: float | None = None
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
