"""RunInput -> identity -> single_turn -> configured real model, with both system prompt sources."""

import copy
import json
import os
import uuid

from pathlib import Path

import pytest
import yaml

from nexus.runner.single import RunHooks, RunInput, run


@pytest.mark.e2e
@pytest.mark.llm
async def test_single_turn_combines_system_prompts_and_continues_history(monkeypatch):
    config_path = os.environ.get('NEXUS_SINGLE_TURN_E2E_CONFIG')
    if not config_path:
        pytest.skip('Set NEXUS_SINGLE_TURN_E2E_CONFIG to a RunInput YAML containing openai_clients.main')
    source = yaml.safe_load(Path(config_path).read_text())
    client = copy.deepcopy(source['openai_clients']['main'])
    client_name = client.get('name', 'openai')
    client_config = client.setdefault('config', {})
    client_config.setdefault('client_args', {}).setdefault('timeout', 90)
    client_config.setdefault('request_args', {}).update(temperature=0, max_tokens=2048)

    expected = {name: f'{name}-{uuid.uuid4().hex[:12]}' for name in ('agent', 'dataset', 'history')}
    system_prompt = f'Agent marker: {expected["agent"]}. Answer the final request with JSON only.'
    prompt = [
        {'role': 'system', 'content': f'Dataset marker: {expected["dataset"]}. Retain all markers.'},
        {'role': 'user', 'content': 'Keep the next historical marker available for our follow-up.'},
        {'role': 'assistant', 'content': f'Historical marker: {expected["history"]}.'},
        {
            'role': 'user',
            'content': 'Return a JSON object with keys agent, dataset and history and their corresponding marker values. '
            'Use the markers from both system messages and the earlier assistant message. No Markdown or explanation.',
        },
    ]
    original_prompt = copy.deepcopy(prompt)
    requests = []

    def observe_request(context):
        real_completion = context.openai_client.create_completion

        async def complete(messages, **kwargs):
            requests.append(copy.deepcopy(messages))
            return await real_completion(messages=messages, **kwargs)

        monkeypatch.setattr(context.openai_client, 'create_completion', complete)

    run_input = RunInput.from_dict(
        {
            'imports': [
                'nexus.extensions.tasks.identity.task',
                'nexus.extensions.agents.single_turn.agent',
                f'nexus.extensions.openai_clients.{client_name}',
            ],
            'task_name': 'identity',
            'agent_name': 'single_turn',
            'agent_config': {'system_prompt': system_prompt, 'prompt': prompt},
            'openai_clients': {'main': client},
            'runner': {'timeout': 120},
        }
    )
    result = await run(run_input, hooks=RunHooks(on_context_ready=observe_request))
    assert result.success, result.error
    expected_messages = [{'role': 'system', 'content': system_prompt}, *original_prompt]
    assert requests == [expected_messages]
    assert prompt == original_prompt
    output = result.agent_output
    assert output.main_trajectory[:-1] == expected_messages
    assert output.main_trajectory[-1]['role'] == 'assistant'
    assert json.loads(output.response.strip()) == expected

    output_dir = Path(os.environ.get('NEXUS_SINGLE_TURN_E2E_OUTPUT', 'workspace/single-turn-system-prompt-e2e'))
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        'success': result.success,
        'model': run_input.openai_clients['main']['config']['request_args']['model'],
        'expected': expected,
        'response': output.response,
        'request_messages': requests[0],
        'trajectory': output.main_trajectory,
    }
    evidence_path = output_dir / 'result.json'
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(f'Single-turn system prompt E2E evidence: {evidence_path.resolve()}')
