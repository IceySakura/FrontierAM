"""Run Kimi Code 2.3.2 through the runtime proxy against a scripted local model.

Set KIMI_CODE_TEST_ENTRY to an installed package's dist/main.mjs and optionally
KIMI_CODE_TEST_NODE to Node.js >=22.19.0. No model credentials are required.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.kimi_code import runtime_setup
from nexus.extensions.agents.kimi_code.installer import DEFAULT_VERSION, MIN_NODE_VERSION, parse_node_version
from nexus.extensions.agents.kimi_code.process import run_process


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def kimi_cli():
    entry_value = os.environ.get('KIMI_CODE_TEST_ENTRY')
    if not entry_value:
        pytest.skip('Set KIMI_CODE_TEST_ENTRY to an installed Kimi Code 2.3.2 dist/main.mjs')
    entry = Path(entry_value).resolve()
    assert entry.is_file()
    metadata = json.loads((entry.parent.parent / 'package.json').read_text())
    assert metadata['name'] == '@moonshot-ai/kimi-code'
    assert metadata['version'] == DEFAULT_VERSION
    node = shutil.which(os.environ.get('KIMI_CODE_TEST_NODE', 'node'))
    assert node is not None, 'Kimi Code requires Node.js >=22.19.0'
    version = subprocess.run([node, '--version'], capture_output=True, text=True, check=True).stdout
    assert parse_node_version(version) >= MIN_NODE_VERSION
    return Path(node), entry


class ScriptedModelBridge:
    """Answer the real proxy's host requests without contacting a model gateway."""

    def __init__(self, reply_with_tool: bool = False):
        self.handlers = {}
        self.requests = []
        self.reply_with_tool = reply_with_tool

    def on(self, name):
        def register(handler):
            self.handlers[name] = handler
            return handler

        return register

    async def emit(self, name, data):
        assert name == 'llm_request'
        self.requests.append(data)
        message = {'role': 'assistant', 'content': 'LONG_PROMPT_OK'}
        if self.reply_with_tool:
            message = {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': f'tool-{len(self.requests)}',
                        'type': 'function',
                        'function': {
                            'name': 'Bash',
                            'arguments': json.dumps({'command': 'printf tool-ok'}),
                        },
                    }
                ],
            }
        response = {
            'id': f'chatcmpl-local-{len(self.requests)}',
            'object': 'chat.completion',
            'created': 0,
            'model': 'nexus-model',
            'choices': [
                {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if self.reply_with_tool else 'stop'}
            ],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110},
        }
        await self.handlers['llm_response']({'request_id': data['request_id'], 'success': True, 'response': response})


def user_text(message):
    content = message.get('content')
    if isinstance(content, str):
        return content
    return ''.join(block.get('text', '') for block in content or [] if block.get('type') == 'text')


@pytest.mark.parametrize('case', ['short', 'ascii140k', 'unicode'])
async def test_real_cli_receives_complete_prompt(tmp_path, monkeypatch, kimi_cli, case):
    import nexus.runtime.context as runtime_context

    payloads = {'short': 'Verify the prompt transport.', 'ascii140k': 'x' * 140000, 'unicode': '中文🙂' * 20000}
    task = 'BEGIN_LONG_PROMPT\n' + payloads[case] + '\nEND_LONG_PROMPT'
    node, entry = kimi_cli
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    monkeypatch.setattr(runtime_setup, 'install_kimi_code', AsyncMock(return_value=(node, entry)))
    bridge = ScriptedModelBridge()
    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: SimpleNamespace(sio=bridge))

    if case == 'ascii140k':
        with pytest.raises(OSError, match='Argument list too long') as error:
            await run_process(
                [str(node), str(entry), '--prompt', task],
                cwd=str(workspace),
                env=dict(os.environ),
                timeout=10,
            )
        assert error.value.errno == errno.E2BIG
        assert not bridge.requests

    result = await runtime_setup.setup_and_run_kimi_code(
        task=task,
        working_dir=str(workspace),
        timeout=60,
        context_window=1000000,
        max_steps=3,
    )

    assert result['success'], result['stderr']
    assert result['exit_code'] == 0
    assert result['exit_status'] == 'finished'
    assert result['assistant_text'] == 'LONG_PROMPT_OK'
    assert result['session_id']
    assert result['trace']['malformed_records'] == 0
    users = [
        user_text(message)
        for request in bridge.requests
        for message in request['messages']
        if message['role'] == 'user'
    ]
    assert task in users, 'The model must receive the complete task, not a file path or truncated text'
    assert (Path(result['artifact_dir']) / 'prompt.txt').read_bytes() == task.encode('utf-8')
    (tmp_path / 'verification.json').write_text(
        json.dumps(
            {
                'case': case,
                'task_bytes': len(task.encode('utf-8')),
                'task_sha256': hashlib.sha256(task.encode('utf-8')).hexdigest(),
                'model_requests': len(bridge.requests),
                'exact_prompt_received': True,
                'exit_status': result['exit_status'],
                'artifact_dir': result['artifact_dir'],
            },
            indent=2,
        )
    )


async def test_real_cli_step_limit_remains_a_failure(tmp_path, monkeypatch, kimi_cli):
    import nexus.runtime.context as runtime_context

    node, entry = kimi_cli
    monkeypatch.setattr(runtime_setup, 'install_kimi_code', AsyncMock(return_value=(node, entry)))
    bridge = ScriptedModelBridge(reply_with_tool=True)
    monkeypatch.setattr(runtime_context, 'require_runtime_control', lambda: SimpleNamespace(sio=bridge))

    result = await runtime_setup.setup_and_run_kimi_code(
        task='Use Bash to print tool-ok, then finish.',
        working_dir=str(tmp_path / 'workspace'),
        timeout=60,
        max_steps=1,
    )

    (tmp_path / 'result.json').write_text(json.dumps(result, indent=2))

    assert len(bridge.requests) == 1
    assert not result['success']
    assert result['exit_code'] == 1
    assert result['exit_status'] == 'error'
    assert 'loop.max_steps_exceeded' in result['stderr']
    assert result['trace']['tool_calls'] == {'Bash': 1}
    assert result['trace']['tool_results'] == 1
    assert result['trace']['malformed_records'] == 0
