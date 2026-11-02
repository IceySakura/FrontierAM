"""Nexus-owned RewardKit/OpenCode configuration for EnterpriseBench Native."""

from __future__ import annotations

import json

from typing import Any


JUDGE_PROVIDER = 'nexus-responses'
JUDGE_ROUTE_MODEL = 'judge'
JUDGE_ROUTE = f'{JUDGE_PROVIDER}/{JUDGE_ROUTE_MODEL}'
JUDGE_PROXY_API_KEY = 'nexus-proxy-key'
JUDGE_TRACE_DIR = '/logs/verifier/opencode_judge'
OPENCODE_BIN = '/root/.opencode/bin'
DEFAULT_PROMPT_TEMPLATE = 'prompt_template.md'
DEFAULT_JUDGE_MODE = 'batched'
DEFAULT_SCORING_AGGREGATION = 'weighted_mean'
DEFAULT_SCORING_THRESHOLD = 0.5
_AGENT_DELIVERABLE_GUIDANCE = (
    '\nNEXUS AGENT DELIVERABLES:\n\n'
    '- The agent final answer is at `/tmp/agent_deliverable/answer.txt`.\n'
    '- The full agent transcript is at `/tmp/agent_deliverable/trajectory.txt`.\n'
    '- For text-deliverable criteria, read `answer.txt` directly rather than '
    'looking for `/logs/agent/trajectory.json`.\n'
)

VERIFIER_ENV = {
    'PATH': f'{OPENCODE_BIN}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
    'NEXUS_JUDGE_PROXY_API_KEY': JUDGE_PROXY_API_KEY,
    'OPENCODE_DISABLE_MODELS_FETCH': '1',
    'REWARDKIT_MODEL': JUDGE_ROUTE,
    'REWARDKIT_OPENCODE_TRACE_DIR': JUDGE_TRACE_DIR,
}


def normalize_judge_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the RewardKit subset stored in a Nexus Dataset item."""
    judge = dict(raw.get('judge') or {})
    criteria = raw.get('criteria', raw.get('criterion'))
    scoring = dict(raw.get('scoring') or {})
    if not isinstance(criteria, list) or not criteria:
        raise ValueError('judge_config.criteria must be a non-empty list')

    harness = str(judge.get('judge') or 'opencode')
    mode = str(judge.get('mode') or DEFAULT_JUDGE_MODE)
    if harness != 'opencode':
        raise ValueError(f'EnterpriseBench Native requires the OpenCode judge harness, got {harness!r}')
    if mode != 'batched':
        raise ValueError(f'EnterpriseBench Native requires batched Judge mode, got {mode!r}')

    normalized_criteria: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_criterion in enumerate(criteria, 1):
        if not isinstance(raw_criterion, dict):
            raise TypeError(f'judge_config.criteria[{index - 1}] must be a mapping')
        criterion = dict(raw_criterion)
        name = str(criterion.get('name') or criterion.get('id') or f'c{index:03d}')
        description = str(criterion.get('description') or '').strip()
        criterion_type = str(criterion.get('type') or 'binary')
        if not description:
            raise ValueError(f'Judge criterion {name!r} has no description')
        if name in names:
            raise ValueError(f'Duplicate Judge criterion name: {name!r}')
        if criterion_type not in {'binary', 'likert', 'numeric'}:
            raise ValueError(f'Unsupported Judge criterion type for {name!r}: {criterion_type!r}')
        names.add(name)
        normalized = {
            key: value
            for key, value in criterion.items()
            if key
            in {
                'name',
                'id',
                'description',
                'type',
                'weight',
                'optional',
                'negate',
                'files',
                'points',
                'min',
                'max',
                'annotations',
            }
        }
        normalized['name'] = name
        normalized['description'] = description
        normalized['type'] = criterion_type
        normalized_criteria.append(normalized)

    prompt_template = str(judge.get('prompt_template') or DEFAULT_PROMPT_TEMPLATE)
    prompt_template = prompt_template.replace('\\', '/')
    if not prompt_template or prompt_template.startswith('/') or '..' in prompt_template.split('/'):
        raise ValueError('judge_config.judge.prompt_template must be a relative path inside tests/')

    aggregation = str(scoring.get('aggregation') or DEFAULT_SCORING_AGGREGATION)
    if aggregation not in {'weighted_mean', 'all_pass', 'any_pass', 'threshold', 'required_pass'}:
        raise ValueError(f'Unsupported RewardKit scoring aggregation: {aggregation!r}')

    normalized_judge = {
        key: value
        for key, value in judge.items()
        if key in {'judge', 'prompt_template', 'mode', 'isolated', 'cwd', 'atif-trajectory'}
    }
    normalized_judge.update({
        'judge': harness,
        'prompt_template': prompt_template,
        'mode': mode,
    })
    return {
        'judge': normalized_judge,
        'criteria': normalized_criteria,
        'scoring': {
            'aggregation': aggregation,
            'threshold': float(scoring.get('threshold', DEFAULT_SCORING_THRESHOLD)),
        },
    }


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_toml_value(item) for item in value) + ']'
    if isinstance(value, dict):
        return '{ ' + ', '.join(f'{key} = {_toml_value(item)}' for key, item in value.items()) + ' }'
    raise TypeError(f'Unsupported TOML value: {type(value).__name__}')


def render_judge_toml(config: dict[str, Any], *, timeout: int) -> str:
    """Render normalized Nexus Judge config into RewardKit's judge.toml format."""
    if timeout <= 0:
        raise ValueError('judge_timeout must be positive')
    normalized = normalize_judge_config(config)
    lines = ['# Generated by Nexus EnterpriseBench Native. Do not edit in the Runtime.', '', '[judge]']
    lines.append(f'model = {_toml_value(JUDGE_ROUTE)}')
    for key, value in normalized['judge'].items():
        lines.append(f'{key} = {_toml_value(value)}')
    lines.append(f'timeout = {_toml_value(timeout)}')
    scoring = normalized['scoring']
    lines.extend(['', '[scoring]'])
    for key, value in scoring.items():
        lines.append(f'{key} = {_toml_value(value)}')
    for criterion in normalized['criteria']:
        lines.extend(['', '[[criterion]]'])
        for key, value in criterion.items():
            lines.append(f'{key} = {_toml_value(value)}')
    return '\n'.join(lines).rstrip() + '\n'


def render_opencode_config(
    *,
    mcp_url: str = 'http://localhost:8900/mcp',
    output_limit: int = 32768,
) -> str:
    """Render the OpenCode configuration used by the Runtime-local Judge."""
    if output_limit <= 0:
        raise ValueError('OpenCode Judge output limit must be positive')
    config = {
        '$schema': 'https://opencode.ai/config.json',
        'compaction': {'auto': False, 'prune': False},
        'tools': {'task': False},
        'agent': {'general': {'disable': True}, 'explore': {'disable': True}},
        'mcp': {
            'world': {
                'type': 'remote',
                'url': mcp_url,
                'enabled': True,
                'timeout': 30000,
            }
        },
        'provider': {
            JUDGE_PROVIDER: {
                'npm': '@ai-sdk/openai',
                'name': 'Nexus Responses proxy',
                'env': [],
                'options': {
                    'baseURL': '{env:NEXUS_JUDGE_PROXY_URL}',
                    'apiKey': '{env:NEXUS_JUDGE_PROXY_API_KEY}',
                },
                'models': {
                    JUDGE_ROUTE_MODEL: {
                        'id': JUDGE_ROUTE_MODEL,
                        'name': JUDGE_ROUTE_MODEL,
                        'tool_call': True,
                        'reasoning': True,
                        'limit': {'context': 200000, 'output': output_limit},
                    }
                },
            }
        },
    }
    return json.dumps(config, indent=2, ensure_ascii=False) + '\n'


def render_prompt_template(source: str) -> str:
    """Add Nexus's persisted final-answer paths without changing rubric wording."""
    if 'NEXUS AGENT DELIVERABLES:' in source:
        return source
    marker = '\nYour job is to grade the agent'
    if marker not in source:
        raise ValueError('EnterpriseBench prompt template has no grading instruction marker')
    return source.replace(marker, _AGENT_DELIVERABLE_GUIDANCE + marker, 1)


__all__ = [
    'JUDGE_PROVIDER',
    'JUDGE_ROUTE',
    'JUDGE_TRACE_DIR',
    'VERIFIER_ENV',
    'normalize_judge_config',
    'render_judge_toml',
    'render_opencode_config',
    'render_prompt_template',
]
