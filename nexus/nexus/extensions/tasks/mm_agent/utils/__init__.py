"""Helper utilities for the mm_agent task generator.

This module centralizes small, stateless helpers used by
``nexus.extensions.tasks.mm_agent.generator`` so the generator file stays
focused on orchestration. The implementations here mirror the equivalent
helpers in ``tasks/hy3_search/utils.py`` to keep behavior consistent across
task families.

It also stages installed-harness input artifacts in the sandbox (``upload_artifacts``).
"""

from __future__ import annotations

import ast
import asyncio
import base64
import logging

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nexus.core import require_context
from nexus.runtime.executors import FileItem


logger = logging.getLogger(__name__)


# ============================================================================
# Model Utilities
# ============================================================================


def _is_aws_third_anthropic_model(model: str) -> bool:
    """Return True if model is served by api_aws_third_anthropic provider."""
    normalized = model.strip().lower()
    return 'api_aws_third_anthropic' in normalized


def _is_litellm_azure_openai_responses_model(model: str) -> bool:
    """Return True for LiteLLM Azure OpenAI Responses routes."""
    normalized = model.strip().lower()
    # LiteLLM Azure OpenAI Responses route examples:
    # - openai/responses/api_azure_openai_gpt-5
    # - responses/api_azure_openai_gpt-5
    # Require both:
    # 1) `responses` as a '/'-separated route segment
    # 2) `api_azure_openai` provider marker in model route
    route_segments = [segment for segment in normalized.split('/') if segment]
    has_responses_segment = 'responses' in route_segments
    has_azure_openai_provider = any(segment.startswith('api_azure_openai') for segment in route_segments)
    return has_responses_segment and has_azure_openai_provider


def _is_dashscope_ali_qwen_model(model: str) -> bool:
    """Return True if model is served by DashScope ali_qwen provider."""
    normalized = model.strip().lower()
    return 'api_ali_qwen' in normalized


def _resolve_openai_responses_reasoning_effort(model: str, value: str | None) -> dict[str, Any] | None:
    """Parse and normalize reasoning_effort for OpenAI Responses models."""
    if not _is_litellm_azure_openai_responses_model(model):
        return None
    if not value:
        return None

    stripped = value.strip()
    if not stripped:
        return None

    parsed_value: str | dict[str, Any] = stripped
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, dict):
                parsed_value = parsed
        except (ValueError, SyntaxError):
            logger.warning('SEARCH_REASONING_EFFORT looks like JSON/dict but parsing failed, using raw string')

    if isinstance(parsed_value, str):
        return {'effort': parsed_value, 'summary': 'auto'}

    effort = parsed_value.get('effort')
    summary = parsed_value.get('summary', 'auto')
    if effort:
        return {'effort': effort, 'summary': summary}

    logger.warning('Responses reasoning_effort dict missing "effort", skipping reasoning_effort')
    return None


# ============================================================================
# Batch Generation Utilities
# ============================================================================


def _safe_parse_imports(imports_value: Any) -> list[str]:
    """Safely parse imports field from parquet data.

    Handles multiple input formats commonly found in Parquet files:
    - numpy.ndarray (most common when reading from parquet)
    - Serialized string representation of list
    - Python list or other iterables
    - None/empty values

    Args:
        imports_value: The imports field value from parquet

    Returns:
        List of import strings, empty list if None or invalid
    """
    if imports_value is None:
        return []

    # Handle numpy array (most common from parquet)
    if hasattr(imports_value, 'tolist'):
        result = imports_value.tolist()
        return [str(item) for item in result] if result else []

    # Handle string (possibly serialized list like "['a', 'b']")
    if isinstance(imports_value, str):
        try:
            parsed = ast.literal_eval(imports_value)
            return parsed if isinstance(parsed, list) else [parsed]
        except (ValueError, SyntaxError):
            return [imports_value]

    # Handle list or other iterables
    if hasattr(imports_value, '__iter__') and not isinstance(imports_value, (str, bytes)):
        return [str(item) for item in imports_value]

    return []


def _build_metadata(instance_id: str, task_name: str, agent_name: str | None, task_config: dict) -> dict[str, Any]:
    """Build metadata dictionary for a batch item.

    Extracts relevant information from task config to store as searchable metadata.

    Args:
        instance_id: Unique instance identifier
        task_name: Name of the task
        agent_name: Name of the agent (or None)
        task_config: Task configuration dictionary

    Returns:
        Metadata dictionary with extracted fields
    """
    metadata: dict[str, Any] = {
        'instance_id': instance_id,
        'task_name': task_name,
        'agent_name': agent_name or 'none',
    }

    # Extract question for analysis and debugging
    if 'question' in task_config:
        question = str(task_config['question'])
        metadata['question'] = question
        metadata['question_length'] = len(question)

    # Extract ground truth answer for evaluation
    if 'answer' in task_config:
        metadata['answer'] = task_config['answer']

    return metadata


# ============================================================================
# Artifact Staging
# ============================================================================


# Formats `codex exec -i` can attach; image artifacts are also sent to the reward judge.
_IMAGE_SUFFIXES = frozenset({'.png', '.jpg', '.jpeg', '.gif', '.webp'})


class TaskArtifact(BaseModel):
    """A per-sample input file: ``source`` on the rollout worker, absolute ``target`` in the sandbox."""

    source: str
    target: str

    @property
    def is_image(self) -> bool:
        return Path(self.target).suffix.lower() in _IMAGE_SUFFIXES


async def upload_artifacts(artifacts: list[TaskArtifact]) -> None:
    """Stage every artifact at its declared sandbox path before the harness starts.

    Two targets colliding would silently overwrite, and an unacknowledged upload would leave the
    harness unable to find the file, so both fail here instead of downstream.
    """
    targets = sorted(artifact.target for artifact in artifacts)
    if len(set(targets)) != len(targets):
        raise ValueError(f'mm_agent artifacts declare duplicate targets: {targets}')

    runtime = await require_context().get_runtime('main')
    # One request per artifact bounds peak base64 memory for long PDFs.
    for artifact in artifacts:
        content = await asyncio.to_thread(_encode_base64, artifact.source)
        item = FileItem(path=artifact.target, content=content, encoding='base64')
        response = await runtime.upload_file([item], overwrite=True)
        results = response.results
        if len(results) != 1 or not results[0].success or results[0].path != artifact.target:
            raise RuntimeError(
                f'Failed to upload mm_agent artifact {artifact.source!r} to {artifact.target!r}: {results}'
            )
    logger.debug(f'Staged {len(artifacts)} artifact(s) in the main runtime')


def _encode_base64(source: str) -> str:
    return base64.b64encode(Path(source).read_bytes()).decode('ascii')
