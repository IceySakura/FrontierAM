"""Build the minimal hy-live row with a scripted A-then-B user; the example harness overlays its cases on it."""

from __future__ import annotations

import json

from typing import Any, Literal


FIRST_USER_TURN = '请先让后台在 /workspace 创建 A.txt，文件内容只写 A。先告诉我任务已受理，不要等待后台完成。'
SECOND_USER_TURN = '看到你的受理回复了。现在再让后台在 /workspace 创建 B.txt，文件内容只写 B，也先告诉我任务已受理。'


def build_scripted_user_row(
    *,
    agent_loop: Literal['nexus_agent', 'session_agent'] = 'session_agent',
    backend: Literal['codebuddy', 'workbuddy'] = 'workbuddy',
) -> dict[str, Any]:
    """Build an A-then-B training row with the selected working-agent config."""
    work_agent_config: dict[str, Any] = {'working_dir': '/workspace', 'llm_client_name': 'external'}
    if backend == 'codebuddy':
        work_agent_config['version'] = '2.97.2'
    nexus_config = {
        'imports': [
            'nexus.extensions.projects.hy_live',
            'nexus.extensions.openai_clients.openai',
        ],
        'task_name': 'hy-live',
        'agent_name': 'hy-live',
        'agent_config': {},
        'task_config': {
            'user_simulator': {
                'type': 'scripted',
                'turns': [FIRST_USER_TURN, SECOND_USER_TURN],
            },
            'policy_client_name': 'main',
            'external_client_name': 'external',
            'backend_adapter': backend,
            'work_agent_configs': {backend: work_agent_config},
        },
        'openai_clients': {
            'external': {
                'name': 'openai',
                'config': {
                    'client_args': {
                        'api_key__env': 'LIVE_AGENT_EXTERNAL_API_KEY',
                        'base_url__env': 'LIVE_AGENT_EXTERNAL_BASE_URL',
                        'timeout': 600,
                    },
                    'request_args': {
                        'model__env': 'LIVE_AGENT_EXTERNAL_MODEL',
                        'temperature': 0,
                        'max_tokens': 8192,
                    },
                    'enforced_request_args': {
                        'model__env': 'LIVE_AGENT_EXTERNAL_MODEL',
                        'stream': False,
                    },
                },
            }
        },
        'runner': {'timeout': 1800},
    }
    return {
        'prompt': [{'role': 'user', 'content': FIRST_USER_TURN}],
        'agent_name': agent_loop,
        'nexus': json.dumps(nexus_config, ensure_ascii=False),
        'data_source': 'hy_live/scripted_a_then_b',
        'ability': 'multi_turn_background_delegation',
        'ground_truth': 'A.txt contains A and B.txt contains B',
        'reward_model': {
            'style': 'rule',
            'ground_truth': 'A.txt contains A and B.txt contains B',
        },
        'extra_info': {
            'scenario': 'scripted_a_then_b',
            'expected_user_turns': 2,
        },
    }
