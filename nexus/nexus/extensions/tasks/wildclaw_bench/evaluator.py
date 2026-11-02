"""WildClawBench grading engine for Nexus framework.

Faithfully replicates the original ``utils/grading.py::run_grading`` logic:
1. Build a Python grading script that imports the task's ``grade()`` function
2. Upload the script to the runtime container
3. Execute it with ``python3`` inside the container
4. Parse the JSON output from stdout (with fallback to last-line JSON extraction)
5. Return structured GradeResult

The grading code runs **inside the container** where the agent has already
executed, so it can access agent-produced files, mock services, and workspace
data — exactly matching the original WildClawBench behavior.

Critical fix: the ``runtime`` parameter is passed explicitly from the task
function, ensuring grading runs in the **same container** where the agent ran.
"""

import json
import logging

from dataclasses import dataclass, field
from typing import Any

from nexus.runtime.executors.command import CommandRunStatus
from nexus.runtime.executors.file import FileItem


logger = logging.getLogger(__name__)


# Default workspace path inside the container
DEFAULT_TMP_WORKSPACE = '/tmp_workspace'


@dataclass
class GradeResult:
    """Result of grading a WildClawBench task."""

    task_id: str
    score: float
    scores: dict[str, Any] = field(default_factory=dict)
    notes: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'task_id': self.task_id,
            'score': self.score,
            'scores': self.scores,
            'notes': self.notes,
        }


async def grade_automated(
    *,
    task_id: str,
    automated_checks: str,
    workspace_path: str = DEFAULT_TMP_WORKSPACE,
    runtime: Any = None,
    transcript: list[dict[str, Any]] | None = None,
) -> GradeResult:
    """Execute automated grading code inside the runtime container.

    Faithfully replicates ``utils/grading.py::run_grading``:

    1. Build a runner script that concatenates:
       - ``import json, sys, os``
       - ``os.chdir("<workspace_path>")``  (match original: cwd=/tmp_workspace)
       - The task's automated_checks code (containing ``def grade(**kwargs)``)
       - A call to ``grade(transcript=<data>, workspace_path="<path>")``
       - ``print(json.dumps(result))``

    2. Upload the script to ``/tmp/_grade_runner.py`` in the container

    3. Execute ``cd <workspace_path> && python3 /tmp/_grade_runner.py`` with 120s timeout

    4. Parse the JSON output:
       - Try parsing the full stdout
       - Fallback: scan lines in reverse for the first valid JSON object

    5. Extract ``overall_score`` as the primary score, with full scores dict

    Args:
        task_id: Unique task identifier
        automated_checks: Python code containing the grade() function
        workspace_path: Path to the workspace directory in runtime
        runtime: Runtime instance to execute grading in. If None, falls back
                 to require_context() (but this is discouraged).
        transcript: Agent transcript data (list of message dicts from the
                    OpenClaw session JSONL). When provided, this is serialized
                    into the grading script so that ``grade()`` functions which
                    need transcript data can access it via the ``transcript``
                    kwarg — even when the container's filesystem does not
                    contain the original ``chat.jsonl`` at the expected path.

    Returns:
        GradeResult with scores from the automated checks
    """
    if not automated_checks:
        return GradeResult(
            task_id=task_id,
            score=0.0,
            scores={},
            notes='No automated grading code found',
        )

    # Get runtime — prefer explicitly passed runtime
    if runtime is None:
        from nexus.core import require_context

        ctx = require_context()
        runtime = await ctx.get_runtime()
        logger.warning(
            '[%s] grade_automated: runtime not passed explicitly, using default. '
            'This may grade in a different container than where the agent ran.',
            task_id,
        )

    # Build the grading runner script — matching the original grading.py
    # Key: set cwd to workspace_path to match original behavior where
    # grading script runs with cwd=/tmp_workspace
    #
    # Transcript injection: In the original WildClawBench pipeline, OpenClaw
    # runs with HOME=/root and OPENCLAW_STATE_DIR=/root/.openclaw, so the
    # session transcript naturally lands at
    #   /root/.openclaw/agents/main/sessions/chat.jsonl
    # and grade() functions that need transcript data read it from that path.
    #
    # In the Nexus runtime, OpenClaw runs with a temporary HOME and state
    # directory (e.g. /tmp/openclaw-run-XXXXXXXX), so the chat.jsonl file
    # does NOT exist at /root/.openclaw/... when grading runs.  The grade()
    # functions fall back to kwargs["transcript"], which was previously [].
    #
    # Fix: serialize the actual transcript (read by _read_transcript from the
    # temp state dir) into a file in the container, then have the grading
    # script read it.  This avoids embedding potentially large JSON directly
    # in the Python source (which could cause issues with very long transcripts).
    transcript_data = transcript or []
    transcript_json = json.dumps(transcript_data, ensure_ascii=False)

    # Upload transcript data as a separate file
    await runtime.upload_file([
        FileItem(
            path='/tmp/_grade_transcript.json',
            content=transcript_json,
            encoding='utf8',
        )
    ])

    runner_code = (
        '\n'.join([
            'import json, sys, os',
            f'os.chdir("{workspace_path}")',
            automated_checks,
            '',
            '# Load transcript from file (injected by Nexus evaluator)',
            'with open("/tmp/_grade_transcript.json", "r", encoding="utf-8") as _f:',
            '    _transcript_data = json.load(_f)',
            f'result = grade(transcript=_transcript_data, workspace_path="{workspace_path}")',
            'print(json.dumps(result))',
        ])
        + '\n'
    )

    # Upload grading script to container
    await runtime.upload_file([
        FileItem(
            path='/tmp/_grade_runner.py',
            content=runner_code,
            encoding='utf8',
        )
    ])

    # Execute grading script in runtime.
    # Original used 120s, but tasks with LLM Judge grading (e.g.
    # video_notes) need more time — the grade() function calls an
    # external LLM for evaluation which can take 2-3 minutes.
    try:
        exec_result = await runtime.run_command(
            f'cd {workspace_path} && python3 /tmp/_grade_runner.py',
            timeout=300,
        )

        if exec_result.status != CommandRunStatus.Finished or exec_result.return_code != 0:
            logger.error(
                '[%s] Grading script failed: status=%s, code=%s',
                task_id,
                exec_result.status,
                exec_result.return_code,
            )
            logger.error('[%s] stderr: %s', task_id, exec_result.stderr)
            logger.error('[%s] stdout: %s', task_id, (exec_result.stdout or '')[:500])
            return GradeResult(
                task_id=task_id,
                score=0.0,
                scores={'error': f'grade script failed: {exec_result.stderr}'},
                notes=f'Grading script execution failed: {exec_result.stderr}',
            )

        # Parse JSON output — matching original logic with fallback
        stdout = (exec_result.stdout or '').strip()
        scores = _parse_grading_output(stdout, task_id)

        if scores is None:
            return GradeResult(
                task_id=task_id,
                score=0.0,
                scores={'error': 'json parse failed: no valid JSON in stdout'},
                notes=f'Failed to parse grading output. stdout: {stdout[:500]}',
            )

        # Check for errors in scores
        if 'error' in scores:
            return GradeResult(
                task_id=task_id,
                score=0.0,
                scores=scores,
                notes=f'Grading error: {scores["error"]}',
            )

        # Extract overall_score (matching original behavior)
        overall_score = _extract_overall_score(scores)

        return GradeResult(
            task_id=task_id,
            score=overall_score,
            scores=scores,
            notes='',
        )

    except Exception as e:
        logger.error('[%s] Failed to execute grading: %s', task_id, e, exc_info=True)
        return GradeResult(
            task_id=task_id,
            score=0.0,
            scores={'error': str(e)},
            notes=f'Runtime grading failed: {e}',
        )


def _parse_grading_output(stdout: str, task_id: str) -> dict | None:
    """Parse grading JSON output from stdout.

    Matches the original grading.py parsing logic:
    1. Try parsing the full stdout as JSON
    2. Fallback: scan lines in reverse for the first valid JSON object

    Args:
        stdout: Raw stdout from the grading script
        task_id: Task ID for logging

    Returns:
        Parsed scores dict, or None if parsing fails
    """
    if not stdout:
        return None

    # Try parsing entire stdout
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass

    # Fallback: scan lines in reverse for valid JSON
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    logger.error(
        '[%s] Failed to parse grading result, no valid JSON found in stdout\nstdout: %s',
        task_id,
        stdout[:500],
    )
    return None


def _extract_overall_score(scores: dict[str, Any]) -> float:
    """Extract the overall score from the grading result.

    Priority:
    1. ``overall_score`` key (used by most WildClawBench tasks)
    2. Average of all numeric values

    Args:
        scores: The raw scores dict from the grading function

    Returns:
        Overall score as a float (0.0 to 1.0)
    """
    if 'overall_score' in scores:
        try:
            return float(scores['overall_score'])
        except (TypeError, ValueError):
            pass

    # Fallback: average of all numeric values
    numeric_values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    if numeric_values:
        return sum(numeric_values) / len(numeric_values)

    return 0.0


def extract_usage_from_transcript(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract token usage information from an OpenClaw transcript.

    Replicates ``utils/grading.py::extract_usage_from_jsonl`` but operates on
    the in-memory transcript list instead of reading a JSONL file.

    Args:
        transcript: List of transcript events (from chat.jsonl)

    Returns:
        Usage statistics dict with token counts and cost
    """
    totals: dict[str, Any] = {
        'input_tokens': 0,
        'output_tokens': 0,
        'cache_read_tokens': 0,
        'cache_write_tokens': 0,
        'total_tokens': 0,
        'cost_usd': 0.0,
        'request_count': 0,
    }

    for entry in transcript:
        if entry.get('type') != 'message':
            continue
        msg = entry.get('message', {})
        if msg.get('role') != 'assistant':
            continue

        totals['request_count'] += 1
        usage = msg.get('usage', {})
        totals['input_tokens'] += usage.get('input', 0)
        totals['output_tokens'] += usage.get('output', 0)
        totals['cache_read_tokens'] += usage.get('cacheRead', 0)
        totals['cache_write_tokens'] += usage.get('cacheWrite', 0)
        totals['total_tokens'] += usage.get('totalTokens', 0)
        cost = usage.get('cost', {})
        totals['cost_usd'] += cost.get('total', 0.0)

    totals['cost_usd'] = round(totals['cost_usd'], 6)
    return totals


def format_scores(task_id: str, scores: dict) -> str:
    """Format scores for display (matching original grading.py format).

    Args:
        task_id: Task identifier
        scores: Scores dictionary

    Returns:
        Formatted string for console output
    """
    if 'error' in scores:
        return f'[{task_id}] Grading error: {scores["error"]}'

    lines = [f'\n{"=" * 60}', f'  {task_id}', f'{"=" * 60}']

    for k, v in scores.items():
        if isinstance(v, (int, float)):
            bar = '█' * int(v * 10) + '░' * (10 - int(v * 10))
            lines.append(f'  {bar} {v:.2f}  {k}')

    lines.append('=' * 60)
    return '\n'.join(lines)


__all__ = [
    'GradeResult',
    'grade_automated',
    'extract_usage_from_transcript',
    'format_scores',
]
