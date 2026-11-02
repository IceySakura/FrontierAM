import ast
import logging
import os
import re

from io import StringIO
from typing import Any

import httpx
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================================
# Core Search Utilities
# ============================================================================


_SEARCH_ID_RE = re.compile(r'_search_(\d+)$')


def extract_instance_id(r: dict[str, Any]) -> str:
    """Extract stable dataset instance_id, fallback to run_id."""
    config = r.get('config', {}) or {}
    task_config = config.get('task_config', {}) or {}
    metadata = r.get('metadata', {}) or {}
    return task_config.get('instance_id') or metadata.get('instance_id') or r.get('run_id') or ''


def normalize_instance_id_for_match(instance_id: str) -> str:
    """Normalize IDs like 'xxx_search_00001' to 'search_00001' for matching."""
    if not instance_id:
        return ''
    m = _SEARCH_ID_RE.search(instance_id)
    return f'search_{m.group(1)}' if m else instance_id


def filter_excluded_results(
    results: list[dict[str, Any]],
    excluded_instance_ids: set[str] | None,
) -> list[dict[str, Any]]:
    """Filter out results whose instance IDs (raw or normalized) are in the exclusion set."""
    if not excluded_instance_ids:
        return results
    normalized_excluded = {normalize_instance_id_for_match(iid) for iid in excluded_instance_ids}

    def _is_excluded(instance_id: str) -> bool:
        return (
            instance_id in excluded_instance_ids or normalize_instance_id_for_match(instance_id) in normalized_excluded
        )

    return [r for r in results if not _is_excluded(extract_instance_id(r))]


def norm_column(col: str) -> str:
    return col.strip().lower().replace(' ', '')


def preprocess_norm_str(content: Any) -> str:
    """Normalize string for comparison: strip, lower, remove spaces and '*'"""
    return str(content or '').lower().strip().replace(' ', '').replace('*', '')


def preprocess_wide_search_dataframes(pred_df: pd.DataFrame, exp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess ground truth and prediction DataFrames for wide_search evaluation (align with widesearch.py).
    - Normalize column names (strip, lower, remove whitespace).
    - Normalize cell values (strip, lower, remove spaces and '*') for robust comparison.
    """
    pred_df = pred_df.copy()
    exp_df = exp_df.copy()

    pred_df.columns = [norm_column(str(col)) for col in pred_df.columns]
    exp_df.columns = [norm_column(str(col)) for col in exp_df.columns]

    for col in pred_df.columns:
        pred_df[col] = pred_df[col].apply(preprocess_norm_str)
    for col in exp_df.columns:
        exp_df[col] = exp_df[col].apply(preprocess_norm_str)

    return pred_df, exp_df


def calc_f1(p: float, r: float) -> float:
    """Compute F1 from precision and recall. Returns 0.0 when p + r is zero."""
    return (2 * p * r) / (p + r) if (p + r) > 1e-9 else 0.0


def get_judge_use_tool_calling_default() -> bool:
    """Get default value for use_tool_calling from environment variable."""
    env_value = os.environ.get('JUDGE_USE_TOOL_CALLING', 'true').lower()
    return env_value in ('true', '1', 'yes')


def parse_router_id(router_id: str) -> tuple[str, str, str]:
    """Parse router_id into (class1, class2, router_version) tuple."""
    parts = router_id.split('-', maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"Invalid router_id format: '{router_id}'. Expected format: 'class1-class2-router_version'")
    return parts[0], parts[1], parts[2]


def flat_string_to_dataframe(flat_str: str, row_sep=' || ', kv_sep=' | ') -> pd.DataFrame:
    """
    Parse a flattened string back into a DataFrame

    Args:
        flat_str: Flattened string with each row as key=value, separated by row_sep
        row_sep: Separator between different rows, default " || "
        kv_sep: Separator for key=value pairs in each row, default " | "

    Returns:
        pd.DataFrame
    """

    rows = flat_str.split(row_sep)
    data = []

    for row in rows:
        kv_pairs = row.split(kv_sep)
        row_dict = {}
        for kv in kv_pairs:
            if '=' in kv:
                key, value = kv.split('=', 1)
                row_dict[key.strip()] = value.strip()
        data.append(row_dict)

    # Build DataFrame
    return pd.DataFrame(data)


def _is_flat_string(data: str) -> bool:
    """
    Detect whether a string is in flat_string format (key=value | key=value || ...).

    A flat_string has no markdown table structure (no header/separator rows) and
    contains key=value pairs separated by ' | ', optionally with ' || ' as row separator.
    We distinguish it from a markdown pipe table by checking that the first non-empty
    line contains '=' signs (key=value pattern) rather than plain column headers.
    """
    stripped = data.strip()
    if not stripped:
        return False
    # Must contain '=' to be key=value format
    if '=' not in stripped:
        return False
    # Must contain ' | ' as kv separator
    if ' | ' not in stripped:
        return False
    # Markdown tables start with '|' at the beginning of a line (pipe table)
    # or contain a separator row like '|---|---|'. Flat strings start with 'key=value'.
    first_line = stripped.split('\n')[0].strip()
    # If the first line starts with '|', it's a markdown pipe table, not a flat string
    if first_line.startswith('|'):
        return False
    # If the first line contains '=' it's likely a flat string
    return '=' in first_line


def unified_to_dataframe(data: dict | str) -> pd.DataFrame:
    """
    Convert nested dictionaries or Markdown/plain table text into a unified DataFrame.

    Args:
        data (dict or str):
            - dict: {1: {column_name: value, ...}, 2: {...}}
            - str: Text containing Markdown tables, plain tables, or flat_string format
                   (key=value | key=value || key=value | key=value)

    Returns:
        pd.DataFrame: Unified format table
    """

    if isinstance(data, dict):
        if not data:
            return pd.DataFrame()
        all_columns = []
        for item in data.values():
            for col in item:
                if col not in all_columns:
                    all_columns.append(col)
        rows = []
        for key in sorted(data.keys(), key=lambda x: int(x)):
            item = data[key]
            row = [item.get(col, pd.NA) for col in all_columns]
            rows.append(row)
        return pd.DataFrame(rows, columns=all_columns)

    if isinstance(data, str):
        # Detect flat_string format (key=value | key=value || ...) and delegate directly
        if _is_flat_string(data):
            return flat_string_to_dataframe(data)

        markdown_strs = re.findall(r'```markdown(.*?)```', data, re.DOTALL)
        table_text = None

        if markdown_strs:
            table_text = markdown_strs[0].strip()
        else:
            pipe_positions = [m.start() for m in re.finditer(r'\|', data)]
            if len(pipe_positions) >= 4:
                first_pipe = pipe_positions[0]
                last_pipe = pipe_positions[-1]
                start = data.rfind('\n', 0, first_pipe)
                start = 0 if start == -1 else start
                end = data.find('\n', last_pipe)
                end = len(data) if end == -1 else end
                table_text = data[start:end].strip()

        if not table_text:
            raise ValueError('No table found in text')

        lines = table_text.split('\n')
        new_lines = []
        for line in lines:
            if set(line.strip()).issubset(set('|- :')) or '|' not in line:
                continue
            new_lines.append('|'.join([cell.strip() for cell in line.split('|')]))
        cleaned_table = '\n'.join(new_lines)

        df = pd.read_csv(StringIO(cleaned_table), sep='|')
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        df.columns = [col.strip() for col in df.columns]
        return df.replace(r'^\s*$', pd.NA, regex=True)

    raise TypeError('Type should be dict or str')


def extract_reward_score(result: dict[str, Any]) -> float:
    """Extract and validate reward score from API response.

    Raises:
        RuntimeError: If response is missing required fields or contains errors.
    """
    data = result.get('data', {})
    request_id = data.get('request_id')
    unique_request_id = data.get('unique_request_id')
    ctx = f'request_id={request_id}, unique_request_id={unique_request_id}'

    route_info = data.get('route_info', {})
    reward_data = data.get('reward')
    if not reward_data:
        raise RuntimeError(f"Invalid reward service response: missing 'data.reward'. {ctx}")

    if reward_data.get('error_code', 0) != 0:
        error_msg = reward_data.get('error_msg', 'unknown')
        raise RuntimeError(f'Reward service error: {error_msg}. {ctx}')

    final_score = reward_data.get('final_score')
    if final_score is None:
        raise RuntimeError(f"Missing 'final_score' in reward response. {ctx}")

    logger.info(
        'Reward service result: score=%s, %s, cost_time=%s, router=%s-%s-%s',
        final_score,
        ctx,
        data.get('cost_time'),
        route_info.get('class1'),
        route_info.get('class2'),
        route_info.get('version'),
    )
    return float(final_score)


def extract_error_context(response: httpx.Response | None) -> str:
    """Extract request/unique IDs from error response for debugging."""
    if response is None:
        return ''
    try:
        data = response.json().get('data', {})
        parts = []
        if rid := data.get('request_id'):
            parts.append(f'request_id={rid}')
        if urid := data.get('unique_request_id'):
            parts.append(f'unique_request_id={urid}')
        return f' ({", ".join(parts)})' if parts else ''
    except Exception:
        return ''


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
