"""PinchBench grading engine for Nexus framework.

This module implements the grading logic for PinchBench tasks:
- Automated grading: Execute embedded Python grading code
- LLM Judge: Use an LLM to evaluate based on rubric
- Hybrid: Combine both approaches with configurable weights
"""

import json
import logging
import re

from dataclasses import dataclass
from typing import Any

from nexus.core import require_context
from nexus.runtime.executors.command import CommandRunStatus
from nexus.runtime.executors.file import FileItem


logger = logging.getLogger(__name__)


@dataclass
class GradeResult:
    """Result of grading a PinchBench task."""

    task_id: str
    score: float
    max_score: float
    grading_type: str
    breakdown: dict[str, float]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'task_id': self.task_id,
            'score': self.score,
            'max_score': self.max_score,
            'grading_type': self.grading_type,
            'breakdown': self.breakdown,
            'notes': self.notes,
        }


async def grade_automated(
    *,
    task_id: str,
    grading_code: str,
    transcript: list[dict],
    workspace_path: str,
) -> GradeResult:
    """Execute automated grading code from the task in runtime container.

    Args:
        task_id: Unique task identifier
        grading_code: Python code containing grade() function
        transcript: Agent's conversation transcript
        workspace_path: Path to the workspace directory in runtime

    Returns:
        GradeResult with scores from the automated checks
    """
    if not grading_code:
        return GradeResult(
            task_id=task_id,
            score=0.0,
            max_score=1.0,
            grading_type='automated',
            breakdown={},
            notes='No automated grading code found',
        )

    # Extract Python code from markdown code block if present
    code_match = re.search(r'```python\s*(.*?)\s*```', grading_code, re.DOTALL)
    if code_match:
        grading_code = code_match.group(1)

    # Get runtime context to execute grading in container
    ctx = require_context()
    runtime = await ctx.get_runtime()

    # Prepare grading script that will run in container
    # The script writes results to a JSON file
    # Serialize transcript and workspace_path as JSON strings so that JSON
    # ``null`` values don't cause ``NameError`` when the script is executed
    # as Python code (Python has ``None``, not ``null``).
    transcript_json = json.dumps(json.dumps(transcript))  # double-encode
    workspace_json = json.dumps(workspace_path)

    grading_script = f"""
import json
import sys
import traceback

# Inject the grading code
{grading_code}

# Execute grading
try:
    transcript = json.loads({transcript_json})
    workspace_path = {workspace_json}

    if not callable(grade):
        result = {{"error": "grade() function not found"}}
    else:
        scores = grade(transcript, workspace_path)
        if not isinstance(scores, dict):
            result = {{"scores": {{}}, "error": "grade() must return a dict"}}
        else:
            result = {{"scores": scores}}
except Exception as e:
    result = {{"error": str(e), "traceback": traceback.format_exc()}}

# Write result to file
with open("/tmp/grading_result.json", "w") as f:
    json.dump(result, f)
"""

    # Write grading script to runtime (use utf8 for text files)
    await runtime.upload_file([
        FileItem(
            path='/tmp/grade_script.py',
            content=grading_script,
            encoding='utf8',
        )
    ])

    # Execute grading script in runtime
    try:
        exec_result = await runtime.run_command(
            'python3 /tmp/grade_script.py',
            timeout=60,
        )

        # Check command execution status (not just return code)
        if exec_result.status != CommandRunStatus.Finished or exec_result.return_code != 0:
            logger.error(f'Grading script failed: status={exec_result.status}, code={exec_result.return_code}')
            logger.error(f'stderr: {exec_result.stderr}')
            return GradeResult(
                task_id=task_id,
                score=0.0,
                max_score=1.0,
                grading_type='automated',
                breakdown={},
                notes=f'Grading script execution failed: {exec_result.stderr}',
            )

        # Read result from runtime and parse JSON directly
        download_response = await runtime.download_file([FileItem(path='/tmp/grading_result.json')])
        if not download_response.files:
            error_msg = download_response.errors[0].message if download_response.errors else 'Unknown error'
            return GradeResult(
                task_id=task_id,
                score=0.0,
                max_score=1.0,
                grading_type='automated',
                breakdown={},
                notes=f'Failed to read grading result: {error_msg}',
            )

        # Parse JSON result (content is plain text string for utf8 files)
        result = json.loads(download_response.files[0].content or '{}')

        # Check for errors
        if 'error' in result:
            error_msg = result['error']
            if 'traceback' in result:
                logger.error(f'Grading traceback:\n{result["traceback"]}')
                error_msg += f'\n{result["traceback"]}'
            return GradeResult(
                task_id=task_id,
                score=0.0,
                max_score=1.0,
                grading_type='automated',
                breakdown={},
                notes=f'Grading error: {error_msg}',
            )

        scores = result.get('scores', {})
        total = _average_scores(scores)
        return GradeResult(
            task_id=task_id,
            score=total,
            max_score=1.0,
            grading_type='automated',
            breakdown=_normalize_score_dict(scores),
            notes='',
        )

    except Exception as e:
        logger.error(f'Failed to execute grading in runtime: {e}', exc_info=True)
        return GradeResult(
            task_id=task_id,
            score=0.0,
            max_score=1.0,
            grading_type='automated',
            breakdown={},
            notes=f'Runtime grading failed: {e}',
        )


async def grade_llm_judge(
    *,
    task_id: str,
    prompt: str,
    expected_behavior: str,
    grading_criteria: list[str],
    llm_judge_rubric: str | None,
    agent_output: Any,
) -> GradeResult:
    """Use LLM to evaluate the agent's response.

    Args:
        task_id: Unique task identifier
        prompt: The original task prompt
        expected_behavior: Expected agent behavior description
        grading_criteria: List of grading criteria
        llm_judge_rubric: Custom rubric for LLM judge (optional)
        agent_output: Agent's output (should have main_trajectory)

    Returns:
        GradeResult with scores from the LLM judge
    """
    # Summarize transcript (async now to support runtime file reading)
    transcript_summary = _summarize_transcript(agent_output.transcript)

    # Build rubric
    rubric = llm_judge_rubric or _format_grading_criteria(grading_criteria)

    # Build judge prompt
    judge_prompt = _build_judge_prompt(prompt, expected_behavior, transcript_summary, rubric)

    # Get LLM judge client from context
    ctx = require_context()
    try:
        client = ctx.get_openai_client('judge')
    except Exception as e:
        logger.error(f'Failed to get judge client: {e}')
        return GradeResult(
            task_id=task_id,
            score=0.0,
            max_score=1.0,
            grading_type='llm_judge',
            breakdown={},
            notes=f'Failed to initialize judge client: {e}',
        )

    # Call LLM judge
    try:
        messages = [
            {
                'role': 'system',
                'content': 'You are a grading function that outputs only JSON objects.',
            },
            {'role': 'user', 'content': judge_prompt},
        ]

        response = await client.create_completion(
            messages=messages,
            temperature=0.0,
        )

        result_text = response.choices[0].message.content
        if not result_text:
            return GradeResult(
                task_id=task_id,
                score=0.0,
                max_score=1.0,
                grading_type='llm_judge',
                breakdown={},
                notes='Empty response from judge',
            )

        # Parse judge response
        parsed = _parse_judge_text(result_text)

        breakdown = parsed.get('scores', {})
        total = parsed.get('total')
        notes = parsed.get('notes', '')

        return GradeResult(
            task_id=task_id,
            score=float(total) if total is not None else 0.0,
            max_score=1.0,
            grading_type='llm_judge',
            breakdown=_normalize_score_dict(breakdown),
            notes=str(notes) if notes else '',
        )

    except Exception as e:
        logger.error(f'LLM judge failed: {e}', exc_info=True)
        return GradeResult(
            task_id=task_id,
            score=0.0,
            max_score=1.0,
            grading_type='llm_judge',
            breakdown={},
            notes=f'Judge evaluation failed: {e}',
        )


def grade_hybrid(
    *,
    task_id: str,
    auto_result: GradeResult,
    llm_result: GradeResult,
    weights: dict[str, float] | None,
) -> GradeResult:
    """Combine automated and LLM judge scores with configurable weights.

    Args:
        task_id: Unique task identifier
        auto_result: Result from automated grading
        llm_result: Result from LLM judge
        weights: Optional weights for combining scores

    Returns:
        Combined GradeResult
    """
    weights = weights or {'automated': 0.5, 'llm_judge': 0.5}
    auto_weight = float(weights.get('automated', 0.5))
    llm_weight = float(weights.get('llm_judge', 0.5))

    total_weight = auto_weight + llm_weight
    if total_weight <= 0:
        auto_weight = llm_weight = 0.5
        total_weight = 1.0

    combined_score = (auto_result.score * auto_weight + llm_result.score * llm_weight) / total_weight

    breakdown = {
        **{f'automated.{k}': v for k, v in auto_result.breakdown.items()},
        **{f'llm_judge.{k}': v for k, v in llm_result.breakdown.items()},
    }

    notes = ' | '.join(filter(None, [auto_result.notes, llm_result.notes]))

    return GradeResult(
        task_id=task_id,
        score=combined_score,
        max_score=1.0,
        grading_type='hybrid',
        breakdown=breakdown,
        notes=notes,
    )


def _average_scores(scores: dict[str, Any]) -> float:
    """Calculate average of score values."""
    values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_score_dict(scores: dict[str, Any]) -> dict[str, float]:
    """Normalize score dict to ensure all values are floats."""
    normalized: dict[str, float] = {}
    for key, value in scores.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _format_grading_criteria(criteria: list[str]) -> str:
    """Format grading criteria as a rubric string."""
    if not criteria:
        return ''
    return '\n'.join(f'- {criterion}' for criterion in criteria)


def _summarize_transcript(transcript: list[dict[str, Any]]) -> str:
    summary_parts: list[str] = []
    for event in transcript:
        if event.get('type') != 'message':
            continue
        msg = event.get('message', {})
        role = msg.get('role')
        if role == 'assistant':
            for item in msg.get('content', []):
                if item.get('type') == 'toolCall':
                    summary_parts.append(f'Tool: {item.get("name")}({json.dumps(item.get("arguments", {}))})')
        elif role == 'toolResult':
            content = msg.get('content', [])
            if content:
                result_preview = str(content[0])[:200]
                summary_parts.append(f'Result: {result_preview}')
        elif role == 'user':
            content = msg.get('content', [])
            if content:
                summary_parts.append(f'User: {content[0]}')
    return '\n'.join(summary_parts)


def _build_judge_prompt(
    prompt: str,
    expected_behavior: str,
    transcript_summary: str,
    rubric: str,
) -> str:
    """Build the prompt for the LLM judge."""
    return (
        'You are a grading function. Your ONLY job is to output a single JSON object.\n\n'
        'CRITICAL RULES:\n'
        '- Do NOT use any tools (no Read, Write, exec, or any other tool calls)\n'
        '- Do NOT create files or run commands\n'
        '- Do NOT write any prose, explanation, or commentary outside the JSON\n'
        '- Respond with ONLY a JSON object — nothing else\n\n'
        'Be a strict evaluator. Reserve 1.0 for genuinely excellent performance. '
        'An average acceptable completion should score around 0.6-0.7. '
        'Deduct points for unnecessary steps, verbose output, and inefficient tool usage.\n\n'
        f'## Task\n{prompt}\n\n'
        f'## Expected Behavior\n{expected_behavior}\n\n'
        f'## Agent Transcript (summarized)\n{transcript_summary}\n\n'
        f'## Grading Rubric\n{rubric}\n\n'
        'Score each criterion from 0.0 to 1.0.\n\n'
        'Respond with ONLY this JSON structure (no markdown, no code fences, no extra text):\n'
        '{"scores": {"criterion_name": 0.0}, "total": 0.0, "notes": "brief justification"}'
    )


def _parse_judge_text(raw_text: str) -> dict[str, Any]:
    """Parse the LLM judge's text response into structured format.

    Args:
        raw_text: Raw text response from LLM judge

    Returns:
        Dictionary with 'scores', 'total', and 'notes' keys
    """
    raw_text = raw_text.strip()
    if not raw_text:
        return {}

    # Try to extract JSON from code blocks
    code_block_match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1))
            if isinstance(parsed, dict):
                return _normalize_judge_response(parsed)
        except json.JSONDecodeError:
            pass

    # Remove markdown code block markers if present
    if raw_text.startswith('```JSON') or raw_text.startswith('```json'):
        raw_text = raw_text[7:]
    elif raw_text.startswith('```'):
        raw_text = raw_text[3:]
    if raw_text.endswith('```'):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return _normalize_judge_response(parsed)
    except json.JSONDecodeError:
        pass

    # Find JSON objects by balanced braces
    json_candidates: list[str] = []
    brace_depth = 0
    current_json: list[str] = []

    for char in raw_text:
        if char == '{':
            if brace_depth == 0:
                current_json = []
            brace_depth += 1
        if brace_depth > 0:
            current_json.append(char)
        if char == '}':
            brace_depth -= 1
            if brace_depth == 0 and current_json:
                json_candidates.append(''.join(current_json))

    # Try parsing from last JSON object
    for candidate in reversed(json_candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return _normalize_judge_response(parsed)
        except json.JSONDecodeError:
            continue

    # Fallback: extract numeric score from prose
    score_pattern = re.search(
        r'(?:total|overall|final)\s*(?:score)?[:\s]*(0\.\d+|1\.0+)',
        raw_text,
        re.IGNORECASE,
    )
    if score_pattern:
        try:
            total = float(score_pattern.group(1))
            if 0.0 <= total <= 1.0:
                return {'scores': {}, 'total': total, 'notes': 'Score extracted from prose'}
        except ValueError:
            pass

    return {}


def _normalize_judge_response(parsed: dict[str, Any]) -> dict[str, Any]:
    """Normalize judge response to expected format."""
    result: dict[str, Any] = {'scores': {}, 'total': None, 'notes': ''}

    # Extract scores from various keys
    if 'scores' in parsed:
        scores_data = parsed['scores']
        if isinstance(scores_data, dict):
            for key, value in scores_data.items():
                if isinstance(value, dict) and 'score' in value:
                    result['scores'][key] = (
                        float(value['score']) if isinstance(value['score'], (int, float, str)) else 0.0
                    )
                elif isinstance(value, (int, float)):
                    result['scores'][key] = value
    elif 'criteria_scores' in parsed:
        criteria = parsed['criteria_scores']
        if isinstance(criteria, dict):
            for key, value in criteria.items():
                if isinstance(value, dict) and 'score' in value:
                    result['scores'][key] = value['score']
                elif isinstance(value, (int, float)):
                    result['scores'][key] = value

    # Extract total score
    if 'total' in parsed and parsed['total'] is not None:
        result['total'] = float(parsed['total']) if isinstance(parsed['total'], (int, float)) else None
    elif 'score' in parsed and isinstance(parsed['score'], (int, float)):
        result['total'] = float(parsed['score'])
    elif 'overall_score' in parsed and isinstance(parsed['overall_score'], (int, float)):
        result['total'] = float(parsed['overall_score'])
    elif result['scores']:
        values = [v for v in result['scores'].values() if isinstance(v, (int, float))]
        if values:
            result['total'] = sum(values) / len(values)

    # Extract notes
    if 'notes' in parsed:
        result['notes'] = str(parsed['notes'])
    elif 'justification' in parsed:
        result['notes'] = str(parsed['justification'])
    elif 'reasoning' in parsed:
        result['notes'] = str(parsed['reasoning'])

    return result


__all__ = [
    'GradeResult',
    'grade_automated',
    'grade_llm_judge',
    'grade_hybrid',
]
