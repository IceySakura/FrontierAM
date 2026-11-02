"""Exercise the actual pinned backend contract without external model calls."""

import subprocess
import sys

from pathlib import Path

import pytest


ROOT = Path('/data/workspace/OSWorld-V2')


@pytest.mark.skipif(not ROOT.exists(), reason='Pinned upstream checkout not available')
def test_aws_third_wire_contract_and_retry_boundary():
    # Isolate upstream modules and registration from the rest of the unit suite.
    source = r"""
import sys
from unittest.mock import AsyncMock
sys.path.insert(0, sys.argv[1])
from nexus.extensions.tasks.os_world_v2.adapter.aws_third import register, AWSThirdRequestError
from nexus.extensions.openai_clients.anthropic_messages import AnthropicMessagesClient, AnthropicMessagesRequestError
from desktop_env.evaluators.backends import BackendConfig, create_backend
from desktop_env.evaluators.model_client import generate_text, generate_chat
register()
calls = []
async def post(self, endpoint, headers, payload, model=''):
    calls.append((endpoint, headers, payload))
    return {'id':'fixture', 'type':'message', 'role':'assistant',
            'content':[{'type':'text','text':'OK'}], 'stop_reason':'end_turn',
            'usage':{'input_tokens':1,'output_tokens':1}}
AnthropicMessagesClient._post_once = post
options = {'provider':'aws_third','model':'anthropic.claude-sonnet-4-6',
           'api_key':'fixture-key','base_url':'https://fixture.invalid','max_tokens':77}
assert generate_text('hello', options=options, system='system') == 'OK'
assert generate_chat([{'role':'system','content':'rules'},
                      {'role':'user','content':'first'},
                      {'role':'assistant','content':'previous'},
                      {'role':'user','content':'next'}], options=options) == 'OK'
endpoint, headers, body = calls[0]
assert endpoint.endswith('/model/anthropic.claude-sonnet-4-6/invoke')
assert 'provider=aws_third' in headers['Authorization']
assert 'fixture-key' in headers['Authorization']
assert body['anthropic_version'] == 'bedrock-2023-05-31'
assert body['max_tokens'] == 77 and 'model' not in body
assert len(calls[1][2]['messages']) == 3
backend = create_backend(BackendConfig(**{k:v for k,v in options.items() if k != 'max_tokens'},
                                      max_tokens=77,retry_attempts=180,retry_delay=0))
assert backend.config.retry_attempts == 3 and backend.client.max_retries == 1
mock = AsyncMock(side_effect=AnthropicMessagesRequestError(400))
backend.client._create_completion = mock
try:
    backend.generate('test',[])
except AWSThirdRequestError as exc:
    assert exc.status_code == 400 and 'fixture-key' not in str(exc)
else: raise AssertionError('400 must fail')
assert mock.await_count == 1
mock = AsyncMock(side_effect=AnthropicMessagesRequestError(503))
backend.client._create_completion = mock
try: backend.chat([{'role':'user','content':'test'}])
except AWSThirdRequestError: pass
else: raise AssertionError('503 must fail after bounded retries')
assert mock.await_count == 3
from types import SimpleNamespace
backend.client._create_completion = AsyncMock(return_value=SimpleNamespace(
    choices=[SimpleNamespace(finish_reason='length',message=SimpleNamespace(content='truncated'))]))
try: backend.generate('test',[])
except RuntimeError as exc: assert 'truncated' in str(exc)
else: raise AssertionError('Truncated judge text must not be scored')
assert backend.client._create_completion.await_count == 1
print('CONTRACT_OK')
"""
    result = subprocess.run([sys.executable, '-c', source, str(ROOT)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert 'CONTRACT_OK' in result.stdout


def test_worker_registers_backend_before_simulator_creation():
    import ast

    from nexus.extensions.tasks.os_world_v2.adapter import worker

    tree = ast.parse(Path(worker.__file__).read_text())
    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {'register', 'Environment'}
    ]
    positions = {name: line for line, name in calls}
    assert positions['register'] < positions['Environment']
