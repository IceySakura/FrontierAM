"""One-sample VLM-STEM Agent smoke runner.

All credentials come from environment variables. The script prints only a
sanitized result summary and can run with LocalRuntime or Gongfeng.
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import uuid

from io import BytesIO

from PIL import Image

from nexus.runner.single import RunInput, run


DEFAULT_RESPONSES_BASE = 'http://trpc-gpt-eval.production.polaris:8080/compatible-mode/v1'
DEFAULT_CHAT_BASE = 'http://llm-api.model-eval.woa.com'
# Only this image/interpreter pair has passed jupyter_with_image on the platform;
# nexus-runtime:dongdu_debug02 + /usr/local/bin/python always ended in tool_execution_error.
DEFAULT_RUNTIME_IMAGE = 'mirrors.tencent.com/asi/kumohu_dev/nexus-runtime-with-image:sweb_with_image1'
DEFAULT_RUNTIME_PYTHON = '/opt/miniconda3/envs/py312-dev/bin/python'


def _tiny_image_data_uri() -> str:
    buffer = BytesIO()
    # AIHub vision models reject either side <= 10 pixels.
    Image.new('RGB', (32, 32), color='white').save(
        buffer,
        format='PNG',
    )
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f'Required environment variable is missing: {name}')
    return value


def _auth_with_route(
    app_id: str,
    app_key: str,
    provider: str,
    model: str,
    timeout: int,
) -> str:
    return (
        f'{app_id}:{app_key}?provider={provider}&model={model}&timeout={timeout}&max_retry=0&request_id={uuid.uuid4()}'
    )


def _client_config(args) -> dict:
    app_id = _required_env('APP_ID')
    app_key = _required_env('APP_KEY')
    if args.client == 'responses':
        api_key = _auth_with_route(
            app_id,
            app_key,
            args.provider,
            args.model,
            args.model_timeout,
        )
        return {
            'name': 'litellm',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'api_base': args.api_base or DEFAULT_RESPONSES_BASE,
                    'max_retries': 1,
                },
                'request_args': {
                    'model': (f'{args.litellm_provider}/responses/{args.model}'),
                    'reasoning_effort': args.reasoning_effort,
                    'allowed_openai_params': ['reasoning_effort'],
                },
            },
        }

    return {
        'name': 'litellm',
        'config': {
            'client_args': {
                'api_key': f'{app_id}:{app_key}',
                'api_base': args.api_base or DEFAULT_CHAT_BASE,
                'max_retries': 1,
            },
            'request_args': {
                'model': args.chat_model,
                'temperature': 0.0,
            },
        },
    }


def _runtime_config(args) -> tuple[str, dict, dict]:
    if args.runtime == 'local':
        return (
            'local',
            {'resource_spec': {'cpu': 2, 'memory': 4}},
            {'main': {}},
        )
    return (
        'gongfeng',
        {
            'api_base_url': _required_env('GONGFENG_API_BASE_URL'),
            'token': _required_env('GONGFENG_RUNTIME_TOKEN'),
            'runtime_image': 'current',
            'resource_spec': {'cpu': 2, 'memory': 10},
            'request_timeout': args.timeout,
            'purpose': 'eval',
        },
        {'main': {'image': args.runtime_image, 'worker_venv': True}},
    )


def _question(force_tool: str) -> tuple[str, str]:
    if force_tool == 'python':
        return (
            'You must call python exactly once to calculate 12345 * 6789. Then answer with the integer result.',
            '83810205',
        )
    if force_tool == 'search':
        return (
            'Call search exactly once for the capital of France, then answer with the city name.',
            'Paris',
        )
    if force_tool == 'visit':
        return (
            'Call visit exactly once on https://example.com with the goal of '
            'identifying the page title, then answer with that title.',
            'Example Domain',
        )
    return ('What is 1 + 1? Answer with the integer only.', '2')


def _strip_multimodal_content(value: str) -> bool | list[str] | None:
    """Parse the CLI form of hytuner's ``strip_multimodal_content``."""
    if value == 'none':
        return None
    if value == 'all':
        return True
    return [router_id for router_id in value.split(',') if router_id]


def build_run_input(args) -> RunInput:
    question, answer = _question(args.force_tool)
    runtime_name, runtime_provider, runtime_configs = _runtime_config(args)
    system_prompt = 'You are a multimodal tool-using assistant. Follow the user instruction exactly.'
    if args.force_tool != 'none':
        system_prompt += f' You must call {args.force_tool} exactly once and then stop calling tools.'

    tool_app_id = os.environ.get('DIST_PLAT_APP_ID', '')
    tool_app_key = os.environ.get('DIST_PLAT_APP_KEY', '')
    runtime_module = 'nexus.runtime.providers.local' if args.runtime == 'local' else 'nexus.runtime.providers.gongfeng'
    task_config = {
        'question': question,
        'images': [args.image or _tiny_image_data_uri()],
        'answer': answer,
        'ground_truth': answer,
        'image_download_timeout': 60,
        'max_image_bytes': 20 * 1024 * 1024,
        'evaluator_type': args.evaluator_type,
    }
    if args.evaluator_type == 'reward_system':
        task_config.update({
            'reward_system_router_id': args.reward_router_id,
            'reward_system_exp_id': args.reward_exp_id,
            'reward_system_strip_multimodal_content': _strip_multimodal_content(args.reward_strip_multimodal),
        })
    return RunInput.model_validate({
        'imports': [
            runtime_module,
            'nexus.extensions.openai_clients.litellm',
            'nexus.extensions.tasks.vlm_stem_agent.task',
            'nexus.extensions.agents.main.agent',
            'nexus.extensions.tools.search_stateful.search',
            'nexus.extensions.tools.search_stateful.visit',
            ('nexus.extensions.tools.code_interpreter.jupyter_with_image'),
        ],
        'run_id': f'vlm-stem-smoke-{uuid.uuid4().hex[:12]}',
        'task_name': 'vlm_stem_agent',
        'task_config': task_config,
        'runtime_provider_name': runtime_name,
        'runtime_provider_config': runtime_provider,
        'runtime_configs': runtime_configs,
        'agent_name': 'main_agent',
        'agent_config': {
            'system_prompt': system_prompt,
            'max_iterations': 5,
            'tools': [
                'search_stateful/search',
                'search_stateful/visit',
                'jupyter_with_image',
            ],
            'parallel_tool_calls': False,
            'tool_call_error_retries': 1,
            'inference_time_timeout': args.timeout,
            'observation_time_timeout': args.timeout,
            'tool_overrides': [
                {
                    'tool_name': 'search_stateful/search',
                    'name': 'search',
                },
                {'tool_name': 'search_stateful/visit', 'name': 'visit'},
                {
                    'tool_name': 'jupyter_with_image',
                    'name': 'python',
                },
            ],
        },
        'tool_configs': {
            'search_stateful': {
                'serper_app_id': tool_app_id,
                'serper_app_key': tool_app_key,
                'jina_app_id': tool_app_id,
                'jina_app_key': tool_app_key,
                'jina_provider': 'aws_jina',
                'jina_model': 'api_naci_default_deepseek-v3.2',
                'jina_model_app_id': os.environ.get(
                    'JINA_MODEL_APP_ID',
                    '',
                ),
                'jina_model_app_key': os.environ.get(
                    'JINA_MODEL_APP_KEY',
                    '',
                ),
                # The default pager view hides URLs behind 【id†title†domain】 markers for
                # open/find, leaving visit nothing to fetch.
                'search_result_format': 'links',
                'use_cache': False,
            },
            'code_interpreter': {
                'python_path': (sys.executable if args.runtime == 'local' else DEFAULT_RUNTIME_PYTHON),
                'work_dir': '/tmp/jupyter_work',
            },
        },
        'openai_clients': {'main': _client_config(args)},
    })


def _trajectory_summary(trajectory) -> list[dict]:
    summary = []
    for message in trajectory or []:
        item = {'role': message.get('role')}
        calls = message.get('tool_calls') or []
        if calls:
            item['tools'] = [(call.get('function') or {}).get('name') for call in calls]
        if message.get('role') == 'tool':
            item['tool_output'] = str(message.get('content') or '')[:300]
        elif isinstance(message.get('content'), str):
            item['content'] = message['content'][:500]
        elif isinstance(message.get('content'), list):
            item['content_types'] = [part.get('type') for part in message['content'] if isinstance(part, dict)]
        summary.append(item)
    return summary


async def _run(args) -> None:
    if args.dry_run:
        config = build_run_input(args).model_dump(mode='json')
        config['openai_clients']['main']['config']['client_args']['api_key'] = '***'
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return

    result = await asyncio.wait_for(
        run(build_run_input(args)),
        timeout=args.timeout + 60,
    )
    task_output = (
        result.task_output.model_dump() if hasattr(result.task_output, 'model_dump') else (result.task_output or {})
    )
    agent_output = (
        result.agent_output.model_dump() if hasattr(result.agent_output, 'model_dump') else (result.agent_output or {})
    )
    print(
        json.dumps(
            {
                'success': result.success,
                'error': result.error.model_dump() if result.error else None,
                'task_output': {
                    'score': task_output.get('score'),
                    'prediction': task_output.get('prediction'),
                    'exit_status': task_output.get('exit_status'),
                    'tool_call_counts': task_output.get('tool_call_counts'),
                },
                'agent_output': {
                    'exit_status': agent_output.get('exit_status'),
                    'iterations': agent_output.get('iterations'),
                    'final_answer': agent_output.get('final_answer'),
                },
                'trajectory': _trajectory_summary(agent_output.get('main_trajectory')),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--client',
        choices=('responses', 'chat'),
        default='responses',
    )
    parser.add_argument(
        '--runtime',
        choices=('local', 'gongfeng'),
        default='local',
    )
    parser.add_argument(
        '--force-tool',
        choices=('none', 'python', 'search', 'visit'),
        default='none',
    )
    parser.add_argument('--provider', default='ali')
    parser.add_argument('--litellm-provider', default='openai')
    parser.add_argument('--model', default='qwen3.8-max')
    parser.add_argument(
        '--chat-model',
        default='openai/api_taiji_hy3',
    )
    parser.add_argument('--api-base')
    parser.add_argument('--reasoning-effort', default='low')
    parser.add_argument('--model-timeout', type=int, default=300)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument(
        '--runtime-image',
        default=DEFAULT_RUNTIME_IMAGE,
    )
    parser.add_argument('--image')
    parser.add_argument(
        '--evaluator-type',
        choices=('none', 'hle', 'general', 'olympiad', 'research', 'oss', 'reward_system'),
        default='none',
    )
    parser.add_argument('--reward-router-id', default='STEMVerifierA20B--2')
    parser.add_argument('--reward-exp-id', default='nexus-vlm-stem-smoke')
    parser.add_argument(
        '--reward-strip-multimodal',
        default='none',
        help="'none' to send images, 'all' to strip them, or a comma-separated router_id list",
    )
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    asyncio.run(_run(parse_args()))
