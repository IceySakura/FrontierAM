"""V2-owned config policy binding the existing CUA/CC/Codex agents and clients."""

import hashlib
import json
import os
import re

from pathlib import Path
from types import SimpleNamespace


SUPPORTED = frozenset({
    ('gpt', 'bare'),
    ('claude', 'bare'),
    ('gpt', 'codex'),
    ('claude', 'cc'),
    ('hunyuan', 'codex'),
    ('hunyuan', 'cc'),
})


def env_int(name, default):
    return int(os.environ[name]) if os.getenv(name, '').strip() else default


def env_float(name, default):
    return float(os.environ[name]) if os.getenv(name, '').strip() else default


def env_bool(name, default):
    return os.environ[name].lower() not in {'0', 'false', 'no', 'off'} if name in os.environ else default


def required(name):
    value = os.getenv(name, '').strip()
    if not value:
        raise ValueError(f'{name} is required')
    return value


def validate_tool_id(value):
    if not re.fullmatch(r'sdt-[A-Za-z0-9]+', value):
        raise ValueError(
            'AGS_OSWORLD_TOOL_ID must be an AGS tool ID (sdt-...), not a hostname. '
            'Set WEBSITE_HOST_SUFFIX=localhost separately, then reload the private V2 .env.'
        )
    return value


def selection(*, model_family=None, harness=None):
    family = model_family or os.getenv('OSWORLD_MODEL_FAMILY', '')
    harness = harness or os.getenv('OSWORLD_HARNESS', '')
    if (family, harness) not in SUPPORTED:
        raise ValueError(f'Unsupported V2 model/harness pair: {family}/{harness}')
    return SimpleNamespace(model_family=family, harness=harness)


def _ags_provision_mode():
    return os.getenv('OSWORLD_AGS_PROVISION_MODE', 'control_plane')


def _runtime_configs():
    return {
        'main': {
            'runtime_kind': 'osworld',
            'tool_type': 'osworld',
            'tool_id': '${oc.env:AGS_OSWORLD_TOOL_ID}',
            'skip_tool_lookup': env_bool('OSWORLD_AGS_SKIP_TOOL_LOOKUP', True),
            'tunnel_port': env_int('OSWORLD_TUNNEL_PORT', 32000),
        }
    }


def _runtime_provider_config():
    return {
        'secret_id': '${oc.env:AGS_SECRET_ID}',
        'secret_key': '${oc.env:AGS_SECRET_KEY}',
        'region': os.getenv('AGS_REGION', 'ap-guangzhou'),
        'domain': os.getenv('AGS_DOMAIN', 'ap-guangzhou.tencentags.com'),
        'interface_mode': 'nexus',
        'request_timeout': env_float('AGS_REQUEST_TIMEOUT', 1800),
        'instance_ready_timeout': env_float('OSWORLD_AGS_INSTANCE_READY_TIMEOUT', 600),
    }


def model_client(selected, choice):
    if choice.model_family == 'hunyuan':
        profile = os.getenv('HUNYUAN_SERVING_PROFILE', 'claude_passthrough')
        if profile not in {'claude_passthrough', 'native_hunyuan'}:
            raise ValueError('Invalid HUNYUAN_SERVING_PROFILE')
        args = {
            'model': required('HUNYUAN_MODEL'),
            'reasoning_effort': os.getenv('HUNYUAN_REASONING_EFFORT', 'high'),
            'stream': False,
            'extra_headers': {
                'StrategyType': 'ConsistentHash',
                'SessionID': '${oc.env:HUNYUAN_SESSION_ID,osworld-v2-' + selected + '}',
            },
        }
        fixed = ['extra_headers']
        if profile == 'native_hunyuan':
            args.update(
                temperature=env_float('HUNYUAN_TEMPERATURE', 0),
                top_p=env_float('HUNYUAN_TOP_P', 0.95),
                max_tokens=env_int('HUNYUAN_MAX_TOKENS', 1024),
            )
            fixed += ['temperature', 'top_p', 'max_tokens']
        base = required('HUNYUAN_BASE_URL').rstrip('/').removesuffix('/chat/completions')
        return {
            'name': 'infer_proxy',
            'config': {
                'client_args': {
                    'base_url': base,
                    'api_key': '${oc.env:HUNYUAN_API_KEY,unused}',
                    'timeout': env_float('HUNYUAN_REQUEST_TIMEOUT', 600),
                    'max_retries': env_int('HUNYUAN_MAX_REQUEST_RETRIES', 5),
                },
                'request_args': args,
                'enforced_request_args': {key: args[key] for key in fixed},
            },
        }
    required('NEXUS_API_KEY')
    client = {'base_url': required('NEXUS_BASE_URL'), 'api_key': '${oc.env:NEXUS_API_KEY}'}
    request = {'model': required('NEXUS_MODEL')}
    if choice.model_family == 'claude':
        name = 'anthropic_messages'
        client.update(
            api_style=os.getenv('NEXUS_ANTHROPIC_API_STYLE', 'messages'),
            timeout=env_float('NEXUS_LLM_TIMEOUT', 900),
            max_retries=env_int('NEXUS_LLM_MAX_RETRIES', 3),
            cache_task_id=hashlib.md5(('v2/' + selected).encode()).hexdigest(),
            cache_control_enabled=True,
            cache_control_strategy='claude_code',
        )
        if os.getenv('NEXUS_LLM_PROVIDER'):
            client['provider'] = os.environ['NEXUS_LLM_PROVIDER']
    else:
        name = 'openai_responses'
        client.update(max_retries=env_int('NEXUS_LLM_MAX_RETRIES', 5))
        if choice.harness == 'bare':
            client.update(ensure_v1_path=True, capture_raw_io=True)
        else:
            client.update(
                timeout=env_float('NEXUS_LLM_TIMEOUT', 3600),
                prompt_cache_task_id=os.getenv('OSWORLD_SESSION_NAMESPACE', '') + '/v2/' + selected,
            )
            request.update(
                reasoning={'effort': os.getenv('CODEX_REASONING_EFFORT', 'none')},
                max_output_tokens=env_int('CODEX_MAX_OUTPUT_TOKENS', 64000),
            )
    return {'name': name, 'config': {'client_args': client, 'request_args': request}}


def agent_config(choice):
    if choice.harness == 'bare':
        common = {
            'max_iterations': env_int('OSWORLD_MAX_ITERATIONS', 100),
            'client_password': os.getenv('OSWORLD_CLIENT_PASSWORD', 'osworld-public-evaluation'),
            'sleep_after_execution': env_float('OSWORLD_SLEEP_AFTER_EXECUTION', 3),
            'step_pause': env_float('OSWORLD_STEP_PAUSE', 1),
            'pyautogui_execution_mode': 'run_command',
            'pyautogui_timeout': env_float('OSWORLD_PYAUTOGUI_TIMEOUT', 30),
        }
        if choice.model_family == 'gpt':
            return 'cua_gpt_agent', {**common, 'reasoning_effort': os.getenv('OSWORLD_REASONING_EFFORT', 'xhigh')}
        return 'cua_anthropic_agent', {
            **common,
            'transport': 'client',
            'client_name': 'main',
            'max_tokens': env_int('OSWORLD_LLM_MAX_TOKENS', 4096),
            'screen_width': 1920,
            'screen_height': 1080,
            'model_display_width': 1280,
            'model_display_height': 720,
            'only_n_most_recent_images': env_int('OSWORLD_MAX_IMAGE_HISTORY', 8),
            'effort': os.getenv('ANTHROPIC_EFFORT', 'max'),
            'no_thinking': env_bool('ANTHROPIC_NO_THINKING', False),
        }
    if choice.harness == 'cc':
        options = {
            'runtime_name': 'main',
            'guide_version': os.getenv('CC_GUIDE_VERSION', 'v1'),
            'cc_model': os.getenv('CC_MODEL', 'claude-opus-4-8'),
            'expected_claude_code_version': os.getenv('CC_VERSION', '2.1.205'),
            'thinking_preset': os.getenv('CC_THINKING_PRESET', 'auto'),
            'max_turns': env_int('CC_MAX_TURNS', 100),
            'disallowed_tools': os.getenv('CC_DISALLOWED_TOOLS', ''),
            'process_guard': env_bool('OSWORLD_CC_PROCESS_GUARD', False),
            'system_grep': True,
            'extra_env': {
                'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
                'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
                **(
                    {
                        'CLAUDE_CODE_AUTO_COMPACT_WINDOW': str(env_int('CC_AUTO_COMPACT_WINDOW', 200000)),
                        'CLAUDE_AUTOCOMPACT_PCT_OVERRIDE': str(env_int('CC_AUTOCOMPACT_PCT_OVERRIDE', 80)),
                        'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(env_int('CC_MAX_OUTPUT_TOKENS', 16384)),
                    }
                    if choice.model_family == 'hunyuan'
                    else {}
                ),
                **{str(k): str(v) for k, v in json.loads(os.getenv('CC_EXTRA_ENV_JSON') or '{}').items()},
            },
        }
        if choice.model_family == 'hunyuan':
            env = options['extra_env']
            assert isinstance(env, dict)
            for key in ('CLAUDE_CODE_AUTO_COMPACT_WINDOW', 'CLAUDE_CODE_MAX_OUTPUT_TOKENS'):
                if int(env[key]) <= 0:
                    raise ValueError(f'{key} must be positive')
            if not 0 < int(env['CLAUDE_AUTOCOMPACT_PCT_OVERRIDE']) <= 100:
                raise ValueError('CLAUDE_AUTOCOMPACT_PCT_OVERRIDE must be between 1 and 100')
        return 'claude_code_multiturn', {
            'execution_profile': 'osworld_v2',
            'execution_options': options,
            'working_dir': '/home/user',
            'max_turns': env_int('OSWORLD_CC_USER_TURNS', 20),
            'timeout': env_float('CC_TIMEOUT', 1800),
            'enable_user_input_hooks': True,
        }
    from nexus.extensions.tasks.os_world_v2.execution.codex_profile import ASSETS, OSWorldCodexOptions

    model = os.getenv('CODEX_MODEL', 'gpt-6-astra')
    catalog_path = os.getenv(
        'CODEX_MODEL_CATALOG_JSON',
        str(ASSETS / ('gpt56_sol_catalog.json' if model == 'gpt-5.6-sol' else 'gpt6_catalog_v8.json')),
    )
    catalog = json.loads(Path(catalog_path).read_text())
    window, output, margin = (
        env_int('CODEX_CONTEXT_LIMIT', 262144),
        env_int('CODEX_MAX_OUTPUT_TOKENS', 64000),
        env_int('CODEX_CONTEXT_TOKEN_MARGIN', 4096),
    )
    compact = env_int('CODEX_AUTO_COMPACT_TOKEN_LIMIT', window - output - margin)
    if compact <= 0 or compact > window - output - margin:
        raise ValueError('Codex compact threshold must reserve output and margin')
    for entry in catalog['models']:
        if entry['slug'] == model:
            entry.update(context_window=window, max_context_window=window, auto_compact_token_limit=compact)
    options = OSWorldCodexOptions(
        user_prompt='',
        model=model,
        version=os.getenv('CODEX_VERSION', '0.154.0'),
        guide_text=Path(os.getenv('CODEX_GUIDE_FILE', str(ASSETS / 'guide_codex_linux.md'))).read_text(),
        model_catalog_json=catalog,
        timeout=env_int('TASK_TIMEOUT', 5400),
        reasoning_effort=os.getenv('CODEX_REASONING_EFFORT', 'none'),
        auto_compact_token_limit=compact,
        binary_url=os.getenv('CODEX_BINARY_URL'),
        binary_checksum=os.getenv('CODEX_BINARY_SHA256'),
    ).model_dump(exclude={'user_prompt'})
    options['extra_config_toml'] = (
        'model_reasoning_summary = '
        + json.dumps(os.getenv('CODEX_REASONING_SUMMARY', 'none'))
        + '\n'
        + options['extra_config_toml']
    )
    options.update(
        session_mode=os.getenv('OSWORLD_CODEX_SESSION_MODE', 'auto'),
        max_turns=env_int('OSWORLD_CODEX_USER_TURNS', 20),
        max_user_requests=env_int('OSWORLD_CODEX_MAX_USER_REQUESTS', 100),
        hook_input_timeout=env_float('OSWORLD_CODEX_INPUT_TIMEOUT', 300),
    )
    return 'codex_native_bridge', {'execution_profile': 'osworld_v2', 'execution_options': options}
