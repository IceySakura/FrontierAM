"""Rubric data models and YAML parsing/validation.

Provides Pydantic models for rubric structure, score computation,
and YAML serialization/deserialization.
"""

import json
import re

from typing import Any

import yaml

from pydantic import BaseModel, Field, model_validator


# Pattern to match rubric item keys (e.g., FC1, SA2, R3, I4)
RUBRIC_KEY_PATTERN = re.compile(r'^[A-Z]{1,3}\d+$')

# Known axis names in baseline rubrics (for convenience properties)
AXIS_FILE_CHANGE = 'file_change_rubrics'
AXIS_SPEC_ALIGNMENT = 'spec_alignment_rubrics'
AXIS_INTEGRITY = 'integrity_rubrics'
AXIS_RUNTIME = 'runtime_rubrics'


class RubricItem(BaseModel):
    """A single rubric evaluation item."""

    id: str = Field(description='Rubric item ID, e.g. FC1, SA2, I3, R4')
    description: str = Field(description='What this rubric item evaluates')
    weight: int = Field(default=1, ge=0, description='Importance weight for scoring')


class RubricAxes(BaseModel):
    """Rubric categories. Uses dict for extensibility - new categories can be added without schema changes."""

    categories: dict[str, list[RubricItem]] = Field(default_factory=dict)

    @property
    def file_change_rubrics(self) -> list[RubricItem]:
        return self.categories.get(AXIS_FILE_CHANGE, [])

    @property
    def spec_alignment_rubrics(self) -> list[RubricItem]:
        return self.categories.get(AXIS_SPEC_ALIGNMENT, [])

    @property
    def integrity_rubrics(self) -> list[RubricItem]:
        return self.categories.get(AXIS_INTEGRITY, [])

    @property
    def runtime_rubrics(self) -> list[RubricItem]:
        return self.categories.get(AXIS_RUNTIME, [])

    def get_all_items(self) -> list[RubricItem]:
        """Flatten all categories into a single list."""
        items = []
        for category_items in self.categories.values():
            items.extend(category_items)
        return items

    @model_validator(mode='before')
    @classmethod
    def _parse_axes_dict(cls, data: Any) -> Any:
        """Accept either {categories: {...}} or direct {file_change_rubrics: [...], ...} format."""
        if isinstance(data, dict) and 'categories' not in data:
            # Direct format: treat all keys as category names
            categories = {}
            for key, value in data.items():
                if isinstance(value, list):
                    categories[key] = value
            return {'categories': categories}
        return data


class RubricMetadata(BaseModel):
    """Metadata about the rubric."""

    task_summary: str = ''
    underlying_bug: str = ''


class RubricSchema(BaseModel):
    """Top-level rubric definition with metadata and evaluation axes."""

    metadata: RubricMetadata = Field(default_factory=RubricMetadata)
    axes: RubricAxes = Field(default_factory=RubricAxes)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'RubricSchema':
        """Parse a YAML string into a RubricSchema.

        Raises:
            yaml.YAMLError: If YAML is malformed
            ValueError: If parsed data is not a valid rubric structure
        """
        if not yaml_str or not yaml_str.strip():
            raise ValueError('Empty rubrics YAML')

        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError(f'Expected dict from YAML, got {type(data).__name__}')

        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        data = self.model_dump()
        # Convert categories back to flat axes format for compatibility
        if 'axes' in data and 'categories' in data['axes']:
            data['axes'] = data['axes']['categories']
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)

    def get_all_items(self) -> list[RubricItem]:
        """Get all rubric items across all axes."""
        return self.axes.get_all_items()

    def get_weights(self) -> dict[str, int]:
        """Get mapping of rubric_id -> weight for all items."""
        return {item.id: item.weight for item in self.get_all_items()}

    def get_max_possible_score(self) -> float:
        """Sum of all weights (score when every item gets 1.0)."""
        return float(sum(item.weight for item in self.get_all_items()))


class RubricScores(BaseModel):
    """Evaluation result: per-item scores and aggregated metrics."""

    item_scores: dict[str, float] = Field(
        default_factory=dict,
        description='Raw binary scores per rubric item, e.g. {"FC1": 1.0, "SA1": 0.0}',
    )
    weighted_sum: float = Field(default=0.0, description='sum(score_i * weight_i)')
    max_possible_score: float = Field(default=0.0, description='sum(weight_i) when all items score 1.0')
    score: float = Field(default=0.0, description='Normalized score in [0, 1] = weighted_sum / max_possible_score')
    raw_response: str = Field(default='', description='Raw LLM response for debugging')

    @classmethod
    def compute(cls, item_scores: dict[str, float], weights: dict[str, int]) -> 'RubricScores':
        """Compute scores from raw item results and rubric weights.

        Args:
            item_scores: {rubric_id: binary_score} from LLM judge
            weights: {rubric_id: weight} from RubricSchema
        """
        weighted_sum = 0.0
        max_possible_score = 0.0

        for rubric_id, score_val in item_scores.items():
            weight = weights.get(rubric_id, 1)
            weighted_sum += score_val * weight

        # max_possible_score uses all weights (not just scored items)
        for weight in weights.values():
            max_possible_score += float(weight)

        # Fallback: if no weights found, use number of items
        if max_possible_score == 0.0 and item_scores:
            max_possible_score = float(len(item_scores))
            weighted_sum = sum(item_scores.values())

        normalized = weighted_sum / max_possible_score if max_possible_score > 0 else 0.0

        return cls(
            item_scores=item_scores,
            weighted_sum=weighted_sum,
            max_possible_score=max_possible_score,
            score=min(max(normalized, 0.0), 1.0),  # Clamp to [0, 1]
        )


def parse_llm_scores_response(response_text: str) -> dict[str, float]:
    """Parse LLM judge response to extract per-item scores.

    Handles multiple formats:
    - Direct JSON: {"FC1": 1, "SA1": 0, ...}
    - Markdown code fence: ```json\\n{...}\\n```
    - JSON embedded in extra text (picks the last {...} block)
    - Nested format: {"file_change_scores": {"FC1": {"score": 1, "weight": 3}}}

    Returns:
        {rubric_id: score} dict
    """
    if not response_text:
        return {}

    # Try direct JSON parse
    parsed = _try_parse_json(response_text)
    if parsed is not None:
        return _extract_scores_from_dict(parsed)

    # Strip markdown code fence: ```json\n...\n``` or ```\n...\n```
    code_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n\s*```', response_text, re.DOTALL)
    if code_block_match:
        parsed = _try_parse_json(code_block_match.group(1).strip())
        if parsed is not None:
            return _extract_scores_from_dict(parsed)

    # Find the last balanced {...} block in the text
    parsed = _extract_last_json_object(response_text)
    if parsed is not None:
        return _extract_scores_from_dict(parsed)

    return {}


def _try_parse_json(text: str) -> dict | None:
    """Try to parse text as JSON dict. Returns None on failure."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _extract_last_json_object(text: str) -> dict | None:
    """Find the last balanced {...} in text and parse it as JSON.

    Scans from the end to find the last '}', then walks backwards
    to find the matching '{' with balanced braces.
    """
    # Find all '}' positions from the end
    close_pos = len(text)
    while True:
        close_pos = text.rfind('}', 0, close_pos)
        if close_pos == -1:
            return None

        # Walk backwards to find the matching '{'
        depth = 0
        for i in range(close_pos, -1, -1):
            if text[i] == '}':
                depth += 1
            elif text[i] == '{':
                depth -= 1
            if depth == 0:
                candidate = text[i : close_pos + 1]
                parsed = _try_parse_json(candidate)
                if parsed is not None:
                    return parsed
                break

        # Try the next '}' further left
        close_pos = close_pos - 1
        if close_pos < 0:
            return None


def _extract_scores_from_dict(data: dict) -> dict[str, float]:
    """Recursively extract rubric scores from a dict."""
    scores: dict[str, float] = {}

    for key, value in data.items():
        if RUBRIC_KEY_PATTERN.match(key):
            scores[key] = _extract_score_value(value)
        elif isinstance(value, dict) and key not in ('overall',):
            scores.update(_extract_scores_from_dict(value))

    return scores


def _extract_score_value(value: Any) -> float:
    """Extract numeric score from value (handles flat and nested formats)."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and 'score' in value:
        score = value['score']
        if isinstance(score, (int, float)):
            return float(score)
    return 0.0
