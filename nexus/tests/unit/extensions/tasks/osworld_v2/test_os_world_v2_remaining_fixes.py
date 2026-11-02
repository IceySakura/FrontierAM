"""Regression coverage for dynamic prompts and task-local service isolation."""

import json
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.os_world_v2.adapter.prompts import RESOURCE_POLICY, build_os_world_system_prompt


def test_resource_policy_in_bare_system_prompt():
    text = build_os_world_system_prompt(client_password='dummy')
    assert RESOURCE_POLICY in text
    assert '64 GB' in text
    assert '37 GB' in text
    assert 'initially' in text
    assert 'change during execution' in text


def test_cc_resource_policy_uses_system_argument():
    import ast

    from nexus.extensions.tasks.os_world_v2.adapter import claude_multiturn

    tree = ast.parse(Path(claude_multiturn.__file__).read_text())
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.List)
        and len(node.elts) >= 2
        and isinstance(node.elts[0], ast.Constant)
        and node.elts[0].value == '--append-system-prompt'
    ]
    assert len(matches) == 1
    assert isinstance(matches[0].elts[1], ast.Name)
    assert matches[0].elts[1].id == 'RESOURCE_POLICY'


def test_visual_evaluation_records_failure_and_identical_images(tmp_path):
    from nexus.extensions.tasks.os_world_v2.adapter.evaluation_diagnostics import record_evaluation

    for name in ('task078_interact_default.png', 'task078_interact_rotated.png'):
        (tmp_path / name).write_bytes(b'same test image')

    def fail():
        with record_evaluation('078', tmp_path):
            raise RuntimeError('secret must not be copied')

    with pytest.raises(RuntimeError, match='secret'):
        fail()
    text = (tmp_path / 'evaluation-diagnostics.jsonl').read_text()
    data = json.loads(text)
    assert data['ok'] is False
    assert data['interaction_images_identical'] is True
    assert data['error_type'] == 'RuntimeError'
    assert 'secret' not in text


def test_codex_resource_policy_enters_config(monkeypatch):
    from nexus.extensions.tasks.os_world_v2.adapter.codex import OSWorldV2CodexProfile
    from nexus.extensions.tasks.os_world_v2.generators.policy import agent_config, selection

    _, config = agent_config(selection(model_family='hunyuan', harness='codex'))
    assert config['execution_profile'] == 'osworld_v2'
    profile = OSWorldV2CodexProfile(config['execution_options'], {}, 'task')
    assert 'developer_instructions = ' + json.dumps(RESOURCE_POLICY) in profile.options.extra_config_toml


def test_task_no_longer_installs_systemd_launch_interceptor():
    from nexus.extensions.tasks.os_world_v2.adapter import adapter
    from nexus.extensions.tasks.os_world_v2.adapter.claude_multiturn import InteractiveAGSAdapter

    source = Path(adapter.__file__).read_text()
    assert 'IsolatedServices' not in source
    instance = object.__new__(InteractiveAGSAdapter)
    instance.session_spec = '/tmp/session/session.json'
    command = instance._build_launch_command(
        SimpleNamespace(working_dir='/home/user'),
        SimpleNamespace(
            dir='/tmp/session', stdout_log='/tmp/out', stderr_log='/tmp/err', exit_code='/tmp/exit', pid='/tmp/pid'
        ),
    )
    assert 'nohup bash' in command
    assert 'sudo' not in command
    assert 'systemd-run' not in command


@pytest.mark.skipif(not Path('/data/workspace/OSWorld-V2').exists(), reason='Requires pinned source')
def test_actual_task041_worker_returns_generated_credentials(tmp_path):
    # Actual subprocess Worker load, no GitLab network call or printed credentials.
    code = r"""
import asyncio
from types import SimpleNamespace
from nexus.core import ExecutionContext
from nexus.extensions.tasks.os_world_v2.adapter.adapter import Worker
async def test():
    async with ExecutionContext(), Worker(SimpleNamespace(), 60) as worker:
        metadata = await worker.call('load', root='/data/workspace/OSWorld-V2',
            task_id='041', host='127.0.0.1', cache=CACHE, password='dummy',
            environment={'GITLAB_URL':'http://localhost','GITLAB_PRIVATE_TOKEN':'dummy'},
            sites=[], suffix='localhost')
        assert "My GitLab account is 'user-" in metadata['instruction']
        assert "and the password is 'Pw" in metadata['instruction']
    print('DYNAMIC_PROMPT_OK')
asyncio.run(test())
"""
    code = code.replace('CACHE', repr(str(tmp_path)))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert 'DYNAMIC_PROMPT_OK' in result.stdout


@pytest.mark.skipif(not Path('/data/workspace/OSWorld-V2').exists(), reason='Requires pinned source')
def test_503_backoff_and_redacted_diagnostics(tmp_path):
    code = r"""
import json,os,sys
from unittest.mock import patch
sys.path.insert(0,'/data/workspace/OSWorld-V2')
os.environ['OSWORLD_AUX_DIAGNOSTIC_DIR']=OUT
from nexus.extensions.tasks.os_world_v2.adapter.aws_third import register,AWSThirdRequestError
from desktop_env.evaluators.backends import BackendConfig
backend=register()(BackendConfig(provider='aws_third',model='test',api_key='private-secret',
    base_url='http://not-contacted.invalid',retry_attempts=180,retry_delay=2))
responses=iter([503,503,'YES'])
def call(*args,**kwargs):
    v=next(responses)
    if isinstance(v,int): raise AWSThirdRequestError(v)
    return v
with patch.object(backend,'_generate_once',side_effect=call), \
     patch('nexus.extensions.tasks.os_world_v2.adapter.aws_third.time.sleep') as sleep, \
     patch('nexus.extensions.tasks.os_world_v2.adapter.aws_third.random.uniform',return_value=0):
    assert backend.generate('private prompt',[])=='YES'
    assert [x.args[0] for x in sleep.call_args_list]==[2,4]
with patch.object(backend,'_generate_once',side_effect=AWSThirdRequestError(401)) as call, \
     patch('nexus.extensions.tasks.os_world_v2.adapter.aws_third.time.sleep') as sleep:
    try: backend.generate('private prompt',[])
    except AWSThirdRequestError: pass
    else: raise AssertionError('must fail')
    assert call.call_count==1 and sleep.call_count==0
text=open(OUT+'/auxiliary-model-calls.jsonl').read()
rows=[json.loads(x) for x in text.splitlines()]
assert [x.get('status') for x in rows]==[503,503,None,401]
assert all(x['max_attempts']==3 for x in rows)
assert 'private-secret' not in text and 'private prompt' not in text
print('RETRY_DIAGNOSTICS_OK')
"""
    code = code.replace('OUT', repr(str(tmp_path)))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'RETRY_DIAGNOSTICS_OK' in result.stdout
