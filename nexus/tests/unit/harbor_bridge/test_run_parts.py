from __future__ import annotations

import json

from pathlib import Path

import pytest

from nexus.extensions.harbor_bridge.bad_pattern_operator import ToolCallBadPatternPolicy
from nexus.extensions.harbor_bridge.claude_unattended_hooks import install_unattended_interaction_hooks
from nexus.extensions.harbor_bridge.run_parts import harbor_bridge_run_part
from nexus.extensions.harbor_bridge.task import AntiHackPolicy
from nexus.runner.compose.merge import deep_merge
from nexus.runner.compose.model import RawRunPart, RunPart
from nexus.runner.compose.parser import load_run_file, parse_run_composition
from nexus.runner.compose.resolver import resolve_non_data_part
from nexus.runner.single import RunInput


PROFILE_ROOT = Path(__file__).resolve().parents[3] / 'nexus' / 'extensions' / 'harbor_bridge' / 'configs'


@pytest.mark.parametrize('mode', ['agent', 'oracle', 'nop'])
def test_harbor_bridge_run_part_selects_mode(mode: str) -> None:
    part = harbor_bridge_run_part({'mode': mode})

    assert part.source == f'harbor_bridge:mode={mode}'
    assert part.body == {
        'imports': ['nexus.extensions.harbor_bridge.task'],
        'task_name': 'harbor_bridge_task',
        'task_config': {
            'mode': mode,
        },
    }
    assert part.merge == {'imports': 'append_unique'}


def test_harbor_bridge_run_part_defaults_to_agent_mode() -> None:
    part = harbor_bridge_run_part({})

    assert part.body['task_config'] == {'mode': 'agent'}


def test_harbor_bridge_run_part_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match='does not accept parameters'):
        harbor_bridge_run_part({'unknown': True})

    with pytest.raises(ValueError, match='mode must be agent, oracle, or nop'):
        harbor_bridge_run_part({'mode': 'invalid'})

    with pytest.raises(ValueError, match='does not accept parameters'):
        harbor_bridge_run_part({'anti_hack_prompt': True})


def test_code_anti_hack_policy_profile_is_valid() -> None:
    part = _load_profile('policies/code-anti-hack-v1.yaml')
    policy = AntiHackPolicy.model_validate(part.body['task_config']['anti_hack_policy'])

    assert part.source == 'code-anti-hack'
    assert policy.id == 'code-anti-hack-v1'
    assert policy.delivery == 'instruction_suffix'
    assert policy.trajectory_strategy == 'restricted-source-access-v1'
    assert policy.prompt.startswith('<IMPORTANT>')
    assert policy.prompt.strip().endswith('</IMPORTANT>')
    assert part.body['processors']['harbor_agent_trajectory'] == {
        'name': 'harbor_agent_trajectory_tracker',
        'args': {'tools_mode': 'branch_last'},
    }


def test_bad_pattern_policy_profile_is_valid() -> None:
    part = _load_profile('policies/bad-pattern-v2.yaml')
    policy = ToolCallBadPatternPolicy.model_validate(part.body['task_config']['tool_call_bad_pattern_policy'])

    assert part.source == 'bad-pattern-v2'
    assert policy.id == 'bad-pattern-v2'
    assert policy.penalty == 0.5
    assert policy.reward_threshold == 0.5
    assert policy.max_hits == 3
    assert policy.enabled_legacy_deferred() == set()
    assert {
        'final_turn_language',
        'final_turn_language_ch',
        'tool_call_args_invalid',
        'reasoning_selftalk_flood',
        'trajectory_truncation',
    } <= policy.enabled_rules()
    assert {
        'edit_call_failure',
        'edit_call_never_recovered',
        'edit_noop_call',
        'trajectory_truncation',
    } <= policy.penalized_rules()
    assert policy.penalty_for('trajectory_truncation') == 0.1
    assert 'reasoning_flood' not in policy.patterns
    assert 'reasoning_selftalk_flood' in policy.penalized_rules()
    assert policy.penalty_for('reasoning_selftalk_flood') == 0.5
    assert policy.reward_threshold_for('reasoning_selftalk_flood') == 0.5
    assert policy.reward_threshold_for('trajectory_truncation') == 0.0
    assert {'think_protocol_leak_all_family', 'tool_call_protocol_leak_all_family'}.isdisjoint(
        policy.penalized_rules()
    )
    assert part.body['processors']['harbor_agent_trajectory'] == {
        'name': 'harbor_agent_trajectory_tracker',
        'args': {'tools_mode': 'branch_last'},
    }


@pytest.mark.parametrize(
    ('policy', 'message'),
    [
        ({'id': '', 'prompt': 'notice'}, 'non-empty string'),
        ({'id': 'policy-v1', 'prompt': 'notice', 'trajectory_strategy': '  '}, 'non-empty string'),
        ({'id': 'policy-v1', 'prompt': 'notice', 'reward_on_detection': 0.0}, 'Extra inputs'),
        (
            {
                'id': 'policy-v1',
                'prompt': 'notice',
                'termination_reason_on_detection': 'CODE_REWARD_HACKING',
            },
            'Extra inputs',
        ),
        ({'id': 'policy-v1', 'delivery': 'system_suffix', 'prompt': 'notice'}, 'delivery'),
        ({'id': 'policy-v1', 'prompt': 'notice', 'unknown': True}, 'Extra inputs'),
    ],
)
def test_anti_hack_policy_rejects_invalid_config(policy: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AntiHackPolicy.model_validate(policy)


def test_anti_hack_policy_allows_detector_only_delivery() -> None:
    policy = AntiHackPolicy.model_validate({
        'id': 'detector-only',
        'delivery': 'none',
        'prompt': '',
        'trajectory_strategy': 'restricted-source-access-v1',
    })

    assert policy.delivery == 'none'
    assert policy.prompt == ''
    assert policy.trajectory_strategy == 'restricted-source-access-v1'


def test_harbor_profiles_compose_into_run_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'sandbox-token')
    monkeypatch.setenv('OPENAI_API_KEY', 'llm-token?provider=aws_third')
    monkeypatch.delenv('AWS_OPUS48_BASE_URL', raising=False)
    monkeypatch.delenv('AWS_OPUS48_MODEL', raising=False)

    config = {
        'task_config': {
            'instance_id': 'example-task',
            'task_dir': '/tmp/example-task',
        },
        'runtime_configs': {
            'main': {
                'image': 'example/task:v1',
            },
        },
    }
    parts = [
        harbor_bridge_run_part({}),
        _load_profile('runtime/gongfeng-prod.yaml'),
        _load_profile('agents/claude-code.yaml'),
        _load_profile('llm/aws-opus48-litellm.yaml'),
    ]
    for part in parts:
        config = deep_merge(config, part.body, part.merge)

    run_input = RunInput.from_dict(config)

    assert run_input.task_name == 'harbor_bridge_task'
    assert run_input.task_config['mode'] == 'agent'
    assert run_input.runtime_provider_name == 'gongfeng'
    assert run_input.runtime_provider_config['token__env'] == 'GONGFENG_RUNTIME_TOKEN'
    assert 'token' not in run_input.runtime_provider_config
    assert 'runtime_provider' not in run_input.runtime_provider_config
    assert run_input.agent_name == 'claude_code'
    assert run_input.openai_clients['main']['name'] == 'litellm'
    assert run_input.agent_config['no_thinking_request_args'] == {
        'model': 'bedrock/anthropic.claude-opus-4-8',
        'max_tokens': 128000,
    }
    assert run_input.agent_config['haiku_model'] == 'bedrock/anthropic.claude-opus-4-8'
    assert run_input.agent_config['extra_env']['ANTHROPIC_MODEL'] == 'claude-opus-4-8'
    client_args = run_input.openai_clients['main']['config']['client_args']
    assert client_args['cache_control_enabled'] is True
    assert client_args['cache_control_strategy'] == 'claude_code'
    assert client_args['log_llm_usage'] is True
    assert client_args['api_key__env'] == 'OPENAI_API_KEY'
    assert 'api_key' not in client_args
    assert 'cache_task_id' not in client_args


def test_harbor_profiles_document_and_validate_complete_component_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nexus.extensions.agents.browser_gym.agent import BrowserGymAgentInput
    from nexus.extensions.agents.claude_code.agent import ClaudeCodeInput
    from nexus.extensions.agents.codex.agent import CodexInput
    from nexus.extensions.agents.openclaw.agent import OpenClawInput
    from nexus.extensions.agents.swe_agent.agent import SWEAgentInput
    from nexus.extensions.agents.terminus2.agent import Terminus2Input

    monkeypatch.setenv('GONGFENG_RUNTIME_TOKEN', 'sandbox-token')
    monkeypatch.setenv('OPENAI_API_KEY', 'llm-token?provider=aws_third')
    # Profiles interpolate these. A developer shell must not change the defaults under test.
    for name in (
        'GONGFENG_API_BASE_URL',
        'GONGFENG_RUNTIME_IMAGE',
        'GONGFENG_STARTUP_JITTER_MAX',
        'CLAUDE_CODE_VERSION',
        'CLAUDE_CODE_NPM_REGISTRY',
        'CLAUDE_CODE_TIMEOUT',
        'AGENT_WORKING_DIR',
        'ANTHROPIC_MODEL',
        'API_FORCE_IDLE_TIMEOUT',
        'API_TIMEOUT_MS',
        'CLAUDE_STREAM_IDLE_TIMEOUT_MS',
        'CODEX_VERSION',
        'CODEX_NPM_REGISTRY',
        'CODEX_TIMEOUT',
        'OPENCLAW_VERSION',
        'OPENCLAW_NPM_REGISTRY',
        'OPENCLAW_CONTEXT_WINDOW',
        'OPENCLAW_MAX_TOKENS',
        'OPENCLAW_LLM_IDLE_TIMEOUT_MS',
        'OPENCLAW_ONBOARD_TIMEOUT',
        'OPENCLAW_SUPPORTS_IMAGES',
        'OPENCLAW_TIMEOUT',
    ):
        monkeypatch.delenv(name, raising=False)

    runtime = _load_profile('runtime/gongfeng-prod.yaml')
    runtime_config = _resolve_profile_body(runtime).runtime_provider_config
    assert runtime_config['runtime_image'] == 'current'
    assert runtime_config['purpose'] == 'prod'
    assert runtime_config['token__env'] == 'GONGFENG_RUNTIME_TOKEN'

    claude = _load_profile('agents/claude-code.yaml')
    claude_config = _resolve_profile_body(claude).agent_config
    ClaudeCodeInput.model_validate({
        **claude_config,
        'user_prompt': 'Complete the task.',
    })
    assert set(claude_config) == set(ClaudeCodeInput.model_fields) - {'user_prompt'}
    assert claude_config['mcp_config_mode'] == 'cli'
    assert claude_config['mcp_servers_mode'] == 'strict_file'
    assert claude_config['no_thinking_request_args_mode'] == 'inherit'
    assert claude_config['pre_start_scripts'] == []
    assert claude_config['enable_unattended_interaction_hooks'] is True

    codex = _load_profile('agents/codex.yaml')
    codex_config = _resolve_profile_body(codex).agent_config
    CodexInput.model_validate({**codex_config, 'user_prompt': 'Complete the task.'})
    assert set(codex_config) == set(CodexInput.model_fields) - {'user_prompt'}
    assert codex_config['mode'] == 'exec_json'
    assert codex_config['goal_mode'] is False
    assert codex_config['goal_token_budget'] is None
    assert codex_config['timeline_collect_tool_args'] is False
    assert codex_config['supports_tool_references'] is None
    assert codex_config['tool_search_output_format'] == 'tool_reference'

    openclaw = _load_profile('agents/openclaw.yaml')
    openclaw_config = _resolve_profile_body(openclaw).agent_config
    OpenClawInput.model_validate({
        **openclaw_config,
        'user_prompt': 'Complete the task.',
    })
    assert set(openclaw_config) == set(OpenClawInput.model_fields) - {'user_prompt'}
    assert openclaw_config['version'] == '2026.6.5'
    assert openclaw.body.get('processors') is None

    browser_gym = _load_profile('agents/browser-gym.yaml')
    browser_gym_config = _resolve_profile_body(browser_gym).agent_config
    BrowserGymAgentInput.model_validate({
        **browser_gym_config,
        'user_prompt': 'Complete the task.',
    })
    assert browser_gym_config == {'max_iterations': 30}

    terminus = _load_profile('agents/terminus2.yaml')
    terminus_config = _resolve_profile_body(terminus).agent_config
    Terminus2Input.model_validate({
        **terminus_config,
        'user_prompt': 'Complete the task.',
    })
    assert set(terminus_config) == set(Terminus2Input.model_fields) - {'user_prompt'}
    assert 'working_dir' not in terminus_config

    swe = _load_profile('agents/swe-agent.yaml')
    swe_config = _resolve_profile_body(swe).agent_config
    SWEAgentInput.model_validate({
        **swe_config,
        'user_prompt': 'Complete the task.',
        'working_dir': '/testbed',
    })
    assert set(swe_config) == set(SWEAgentInput.model_fields) - {'user_prompt', 'working_dir'}


def test_chat_and_token_observability_profile() -> None:
    run_input = _resolve_profile_body(_load_profile('policies/chat-and-token-tracking.yaml'))

    assert run_input.processors['chat'].name == 'chat_tracker'
    assert run_input.processors['chat'].args == {'tools_mode': 'branch_last'}
    assert run_input.processors['token_counter'].name == 'token_counter'
    assert run_input.processors['token_counter'].args == {}
    assert run_input.imports == [
        'nexus.trace.processors.chat_tracker',
        'nexus.trace.processors.token_counter',
    ]


def test_harbor_claude_unattended_hooks_append_and_are_idempotent(tmp_path: Path) -> None:
    settings_path = tmp_path / '.claude' / 'settings.json'
    hook_path = tmp_path / '_nexus_askuser_hook.py'
    settings_path.parent.mkdir()
    existing_hook = {
        'matcher': 'Bash',
        'hooks': [{'type': 'command', 'command': 'audit-command'}],
    }
    settings_path.write_text(
        json.dumps({'permissions': {'defaultMode': 'bypassPermissions'}, 'hooks': {'PreToolUse': [existing_hook]}})
    )

    install_unattended_interaction_hooks(settings_path=settings_path, hook_script_path=hook_path)
    install_unattended_interaction_hooks(settings_path=settings_path, hook_script_path=hook_path)

    settings = json.loads(settings_path.read_text())
    hooks = settings['hooks']['PreToolUse']
    assert settings['permissions'] == {'defaultMode': 'bypassPermissions'}
    assert hooks[0] == existing_hook
    assert [entry['matcher'] for entry in hooks] == ['Bash', 'ExitPlanMode', 'AskUserQuestion']
    assert len(hooks) == 3
    assert 'json.load(sys.stdin)' in hook_path.read_text()


@pytest.mark.parametrize(
    'profile',
    [
        'llm/aws-opus48-litellm.yaml',
        'llm/aws-opus5-litellm.yaml',
        'llm/gcp-opus48-litellm.yaml',
        'llm/gcp-opus5-litellm.yaml',
    ],
)
def test_claude_profiles_all_select_litellm(profile: str) -> None:
    run_input = _resolve_profile_body(_load_profile(profile))

    assert run_input.openai_clients['main']['name'] == 'litellm'
    assert 'nexus.extensions.openai_clients.litellm' in run_input.imports


@pytest.mark.parametrize(
    'profile',
    [
        'llm/aihub-hy3-litellm.yaml',
        'llm/aihub-hy3.yaml',
        'llm/aws-opus48-litellm.yaml',
        'llm/aws-opus5-litellm.yaml',
        'llm/gcp-opus48-litellm.yaml',
        'llm/gcp-opus5-litellm.yaml',
        'llm/gpt56-sol-aihub-2.0.yaml',
        'llm/infer-prod-hy4.yaml',
        'llm/opus5-aihub-2.0.yaml',
        'llm/taiji-hy3.yaml',
    ],
)
def test_harbor_llm_profiles_explicitly_enforce_their_output_budget(
    profile: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('INFER_PROD_HY4_MODEL', 'hy4-test-model')
    run_input = _resolve_profile_body(_load_profile(profile))
    config = run_input.openai_clients['main']['config']

    budget_key = 'max_tokens' if 'max_tokens' in config['request_args'] else 'max_output_tokens'
    assert config['enforced_request_args'][budget_key] == config['request_args'][budget_key]


def test_infer_prod_hy4_profile_enables_per_task_session_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('INFER_PROD_HY4_BASE_URL', raising=False)
    monkeypatch.setenv(
        'INFER_PROD_HY4_MODEL',
        'HY4.0_A49B_770B_SFT0803_1M_Opus_offline_HF_wza_jayson_all',
    )
    run_input = _resolve_profile_body(_load_profile('llm/infer-prod-hy4.yaml'))
    config = run_input.openai_clients['main']['config']

    assert config['client_args']['nexus_trace_id_from_run_id'] is True
    assert config['client_args']['nexus_llm_lb_strategy'] == 'ConsistentHash'
    assert config['request_args']['model'] == 'openai/HY4.0_A49B_770B_SFT0803_1M_Opus_offline_HF_wza_jayson_all'


def test_aws_opus48_litellm_profile_uses_bedrock_invoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('AWS_OPUS48_BASE_URL', raising=False)
    monkeypatch.delenv('AWS_OPUS48_MODEL', raising=False)
    run_input = _resolve_profile_body(_load_profile('llm/aws-opus48-litellm.yaml'))
    config = run_input.openai_clients['main']['config']

    assert config['client_args']['api_key__env'] == 'OPENAI_API_KEY'
    assert config['client_args']['base_url'] == 'http://trpc-gpt-eval.production.polaris:8080'
    assert config['client_args']['cache_task_id_from_run_id'] is True
    assert config['request_args']['model'] == 'bedrock/anthropic.claude-opus-4-8'
    assert config['request_args']['thinking']['type'] == 'adaptive'
    assert config['request_args']['output_config']['effort'] == 'xhigh'
    assert run_input.agent_config['extra_env']['ANTHROPIC_MODEL'] == 'claude-opus-4-8'


def test_gcp_opus48_litellm_profile_uses_chat_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('AIHUB_GOOGLE_OPUS48_BASE_URL', raising=False)
    monkeypatch.delenv('AIHUB_GOOGLE_OPUS48_MODEL', raising=False)
    run_input = _resolve_profile_body(_load_profile('llm/gcp-opus48-litellm.yaml'))
    client = run_input.openai_clients['main']
    config = client['config']

    assert client['name'] == 'litellm'
    assert config['client_args']['api_key__env'] == 'OPENAI_API_KEY'
    assert config['client_args']['base_url'] == 'http://llm-api.model-eval.woa.com/v1'
    assert config['client_args']['cache_task_id_from_run_id'] is True
    assert 'request_id_from_cache_task_id' not in config['client_args']
    assert config['request_args']['model'] == 'api_google_claude-opus-4-8'
    assert config['request_args']['custom_llm_provider'] == 'openai'
    assert config['request_args']['max_tokens'] == 128000
    assert config['request_args']['extra_body']['thinking']['type'] == 'adaptive'
    assert config['request_args']['extra_body']['output_config']['effort'] == 'xhigh'
    assert run_input.agent_config['thinking_preset'] == 'bedrock-claude'
    assert run_input.agent_config['extra_env']['ANTHROPIC_MODEL'] == 'claude-opus-4-8'


def test_aws_opus5_litellm_profile_uses_bedrock_invoke_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('AWS_OPUS5_BASE_URL', raising=False)
    monkeypatch.delenv('AWS_OPUS5_MODEL', raising=False)
    run_input = _resolve_profile_body(_load_profile('llm/aws-opus5-litellm.yaml'))
    client = run_input.openai_clients['main']
    config = client['config']

    assert run_input.imports == ['nexus.extensions.openai_clients.litellm']
    assert client['name'] == 'litellm'
    assert config['client_args']['api_key__env'] == 'OPENAI_API_KEY'
    assert config['client_args']['base_url'] == 'http://trpc-gpt-eval.production.polaris:8080'
    assert config['client_args']['cache_task_id_from_run_id'] is True
    assert config['client_args']['cache_control_enabled'] is True
    assert config['client_args']['cache_control_strategy'] == 'default'
    assert config['client_args']['cache_control_last_n_messages'] == 1
    assert config['request_args']['model'] == 'bedrock/invoke/anthropic.claude-opus-5'
    assert 'custom_llm_provider' not in config['request_args']
    assert 'extra_body' not in config['request_args']
    assert config['request_args']['thinking']['type'] == 'adaptive'
    assert config['request_args']['output_config']['effort'] == 'max'
    assert run_input.agent_config['extra_env']['ANTHROPIC_MODEL'] == 'claude-opus-5'

    import litellm

    resolved_model, provider, _, _ = litellm.get_llm_provider(model=config['request_args']['model'])
    assert (resolved_model, provider) == ('invoke/anthropic.claude-opus-5', 'bedrock')


def test_gcp_opus5_litellm_profile_uses_chat_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('AIHUB_GOOGLE_OPUS5_BASE_URL', raising=False)
    monkeypatch.delenv('AIHUB_GOOGLE_OPUS5_MODEL', raising=False)
    run_input = _resolve_profile_body(_load_profile('llm/gcp-opus5-litellm.yaml'))
    config = run_input.openai_clients['main']['config']

    assert config['client_args']['api_key__env'] == 'OPENAI_API_KEY'
    assert config['client_args']['base_url'] == 'http://llm-api.model-eval.woa.com/v1'
    assert config['client_args']['cache_task_id_from_run_id'] is True
    assert config['request_args']['model'] == 'api_google_claude-opus-5'
    assert config['request_args']['custom_llm_provider'] == 'openai'
    assert config['request_args']['max_tokens'] == 128000
    assert config['request_args']['extra_body']['thinking'] == {
        'type': 'adaptive',
        'display': 'summarized',
    }
    assert config['request_args']['extra_body']['output_config']['effort'] == 'max'
    assert run_input.agent_config['thinking_preset'] == 'bedrock-claude'
    assert run_input.agent_config['extra_env']['ANTHROPIC_MODEL'] == 'claude-opus-5'


def test_aihub2_opus5_profile_uses_standard_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('AIHUB2_OPUS5_BASE_URL', raising=False)
    monkeypatch.delenv('AIHUB2_OPUS5_MODEL', raising=False)
    run_input = _resolve_profile_body(_load_profile('llm/opus5-aihub-2.0.yaml'))
    client = run_input.openai_clients['main']
    config = client['config']

    assert run_input.imports == ['nexus.extensions.openai_clients.openai']
    assert client['name'] == 'openai'
    assert config['client_args']['api_key__env'] == 'OPENAI_API_KEY'
    assert config['client_args']['base_url'] == 'http://api.aihub.woa.com/standard/v1'
    assert config['request_args']['model'] == 'claude-opus-5'
    assert config['request_args']['max_tokens'] == 128000
    assert config['client_args']['cache_control_strategy'] == 'default'
    assert config['client_args']['cache_control_last_n_messages'] == 1
    assert config['request_args']['extra_body']['thinking'] == {
        'type': 'adaptive',
        'display': 'summarized',
    }
    assert config['request_args']['extra_body']['output_config']['effort'] == 'max'
    assert run_input.agent_config['thinking_preset'] == 'bedrock-claude'
    assert run_input.agent_config['extra_env']['ANTHROPIC_MODEL'] == 'claude-opus-5'


def test_aihub2_gpt56_sol_profile_uses_standard_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('AIHUB2_GPT56_SOL_BASE_URL', raising=False)
    monkeypatch.delenv('AIHUB2_GPT56_SOL_MODEL', raising=False)
    monkeypatch.delenv('AIHUB2_GPT56_SOL_REASONING_EFFORT', raising=False)
    monkeypatch.delenv('AIHUB2_GPT56_SOL_MAX_OUTPUT_TOKENS', raising=False)
    run_input = _resolve_profile_body(_load_profile('llm/gpt56-sol-aihub-2.0.yaml'))
    client = run_input.openai_clients['main']
    config = client['config']

    assert run_input.imports == ['nexus.extensions.openai_clients.openai_responses']
    assert client['name'] == 'openai_responses'
    assert config['client_args']['api_key__env'] == 'OPENAI_API_KEY'
    assert config['client_args']['base_url'] == 'http://api.aihub.woa.com/standard/v1'
    assert config['request_args'] == {
        'model': 'gpt-5.6-sol',
        'max_output_tokens': 32768,
        'reasoning': {'effort': 'max'},
        'temperature': 1.0,
    }


def test_aihub_hy3_profile_uses_standard_reasoning_content_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('HY3_REASONING_EFFORT', raising=False)
    run_input = _resolve_profile_body(_load_profile('llm/aihub-hy3.yaml'))
    client = run_input.openai_clients['main']['config']

    assert client['client_args']['api_key__env'] == 'OPENAI_API_KEY'
    assert client['client_args']['base_url'] == 'http://llm-api.model-eval.woa.com/v1'
    assert client['request_args'] == {
        'model': 'api_taiji_hy3',
        'max_tokens': 128000,
        'reasoning_effort': 'high',
    }
    assert 'thinking' not in client['request_args']
    assert run_input.agent_config['thinking_preset'] == 'hunyuan'
    assert run_input.agent_config['extra_env']['ANTHROPIC_MODEL'] == 'api_taiji_hy3'


def test_taiji_gpu_runtime_profile_preserves_task_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TAIJI_APP_GROUP', 'TaiJi_HYAide_T3_Nexus_Sandbox')
    monkeypatch.setenv('TAIJI_APP_GROUP_LOCATION', 'gz')
    monkeypatch.delenv('TAIJI_PURPOSE', raising=False)

    task_resources = {
        'main': {
            'image': 'agent:v1',
            'resource_spec': {'cpu': 8, 'memory': 123, 'gpu': 1, 'gpu_card_type': 'H20'},
        },
        'verifier': {
            'image': 'verifier:v1',
            'resource_spec': {'cpu': 8, 'memory': 123, 'gpu': 1, 'gpu_card_type': 'H20'},
        },
    }
    config = {
        'task_name': 'harbor_bridge_task',
        'runtime_provider_name': 'gongfeng',
        'runtime_provider_config': {'runtime_provider': 'suanli'},
        'runtime_configs': task_resources,
    }
    profile = _load_profile('runtime/taiji-gpu.yaml')

    assert 'runtime_configs' not in profile.body
    merged = RunInput.from_dict(deep_merge(config, profile.body, profile.merge))

    assert merged.runtime_provider_name == 'taiji'
    assert merged.runtime_provider_config['token__env'] == 'GONGFENG_RUNTIME_TOKEN'
    assert merged.runtime_provider_config['taiji_app_group'] == 'TaiJi_HYAide_T3_Nexus_Sandbox'
    assert merged.runtime_provider_config['taiji_app_group_location'] == 'gz'
    assert merged.runtime_provider_config['purpose'] == 'prod'
    assert 'runtime_provider' not in merged.runtime_provider_config
    assert merged.runtime_configs == task_resources


def test_all_harbor_profile_files_resolve_and_experiment_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('TAIJI_APP_GROUP', 'TaiJi_HYAide_T3_Nexus_Sandbox')
    monkeypatch.setenv('TAIJI_APP_GROUP_LOCATION', 'gz')
    profile_dirs = ('runtime', 'agents', 'llm')
    profile_paths = sorted(path for directory in profile_dirs for path in (PROFILE_ROOT / directory).glob('*.yaml'))

    assert profile_paths
    for path in profile_paths:
        part = _load_profile(str(path.relative_to(PROFILE_ROOT)))
        assert part.body
        assert part.source.startswith('harbor-')

    experiment_part_counts = {
        'opus48-claude-code-prod.yaml': 6,
        'opus-5-claude-code-prod.yaml': 6,
        'search-hy3-prod.yaml': 7,
    }
    for experiment_name, expected_parts in experiment_part_counts.items():
        experiment_path = PROFILE_ROOT / 'experiments' / experiment_name
        data, source_path = load_run_file(experiment_path)
        composition = parse_run_composition(data, source_path=source_path)
        assert len(composition.parts) == expected_parts


def _load_profile(relative_path: str) -> RunPart:
    path = PROFILE_ROOT / relative_path
    return resolve_non_data_part(
        RawRunPart(
            kind='use',
            value=str(path),
            source=f'--use {path}',
        )
    )


def _resolve_profile_body(part: RunPart) -> RunInput:
    return RunInput.from_dict({'task_name': 'profile_validation_task', **part.body})
