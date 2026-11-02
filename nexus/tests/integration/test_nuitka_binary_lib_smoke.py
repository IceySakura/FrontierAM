import os
import subprocess
import sys

from pathlib import Path

import pytest


def _binary_dir() -> Path:
    value = os.getenv('NEXUS_BINARY_LIB_DIR')
    if not value:
        pytest.skip('set NEXUS_BINARY_LIB_DIR to run binary lib smoke tests')
    return Path(value)


def _docker_runtime_enabled() -> bool:
    return os.getenv('NEXUS_BINARY_DOCKER_RUNTIME', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _docker_runtime_image() -> str:
    return os.getenv('NEXUS_BINARY_DOCKER_RUNTIME_IMAGE', 'nexus-runtime:1.1.3')


def test_binary_import_and_core_registries(tmp_path):
    binary_dir = _binary_dir()
    script = (
        'import nexus\n'
        'from nexus.core import agent_registry, tool_registry\n'
        "assert agent_registry.get_entry('main_agent')\n"
        "assert tool_registry.get_entry('str_replace_editor')\n"
        "assert tool_registry.get_entry('jupyter_code_interpreter')\n"
    )

    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(binary_dir)},
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_binary_main_agent_runs_with_real_docker_runtime(tmp_path):
    if not _docker_runtime_enabled():
        pytest.skip('set NEXUS_BINARY_DOCKER_RUNTIME=1 to run real Docker runtime binary e2e')

    binary_dir = _binary_dir()
    runtime_image = _docker_runtime_image()
    script = """
import asyncio
import inspect
import json
import os
import sys

import nexus
from nexus.core import ExecutionContext, OpenAICompatibleClient, OpenAIConfig, tool_registry
from nexus.extensions.agents.main.agent import MainAgentInput, run_main_agent
from nexus.runtime.providers.docker import DockerRuntimeProvider


binary_dir = os.environ['NEXUS_BINARY_LIB_DIR']
assert binary_dir in sys.path, sys.path
assert '/workspace' not in sys.path, sys.path
assert '/data/projects/nexus' not in sys.path, sys.path
assert type(nexus.__loader__).__name__ == 'nuitka_module_loader'
try:
    inspect.getsource(nexus)
except (OSError, TypeError):
    pass
else:
    raise AssertionError('inspect.getsource(nexus) unexpectedly returned source')


class MockToolCall:
    def __init__(self, call_id, name, args):
        self.id = call_id
        self.function = type('Function', (), {'name': name, 'arguments': json.dumps(args)})()


class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, **kwargs):
        result = {'role': 'assistant'}
        if self.content:
            result['content'] = self.content
        if self.tool_calls:
            result['tool_calls'] = [
                {
                    'id': call.id,
                    'function': {
                        'name': call.function.name,
                        'arguments': call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return result


class MockClient(OpenAICompatibleClient):
    def __init__(self, messages):
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self._messages = iter(messages)

    async def _create_completion(self, messages, model=None, **kwargs):
        message = next(self._messages)
        choice = type('Choice', (), {'message': message})()
        return type('Response', (), {'choices': [choice]})()


async def main():
    runtime_image = os.environ['NEXUS_BINARY_DOCKER_RUNTIME_IMAGE']
    run_id = f'binary-docker-runtime-integration-{os.getpid()}'
    client = MockClient([
        MockMessage(tool_calls=[
            MockToolCall('tc1', 'jupyter_code_interpreter', {'code': 'print(20 + 22)'}),
            MockToolCall(
                'tc2',
                'str_replace_editor',
                {'command': 'create', 'path': '/tmp/nexus-docker-e2e.txt', 'file_text': 'docker runtime e2e'},
            ),
        ]),
        MockMessage(content='done'),
    ])
    provider = DockerRuntimeProvider(
        image=runtime_image,
        runtime_image=runtime_image,
        python_version='3.12',
        auto_connect_ws=False,
    )
    ctx = ExecutionContext(
        runtime_provider=provider,
        openai_clients={'main': client},
        run_id=run_id,
        tool_state={'working_dir': '/tmp'},
    )
    async with ctx:
        output = await run_main_agent(MainAgentInput(
            user_prompt='run docker runtime tools',
            tools=['jupyter_code_interpreter', 'str_replace_editor'],
            observation_time_timeout=240,
        ))

    assert output.exit_status == 'finished', output
    assert output.final_answer == 'done', output
    tool_messages = [msg for msg in output.main_trajectory if msg.get('role') == 'tool']
    assert any('42' in str(msg.get('content')) for msg in tool_messages), tool_messages
    assert any('File created successfully' in str(msg.get('content')) for msg in tool_messages), tool_messages


asyncio.run(main())
"""

    env = {
        **os.environ,
        'PYTHONPATH': str(binary_dir),
        'NEXUS_BINARY_LIB_DIR': str(binary_dir),
        'NEXUS_BINARY_DOCKER_RUNTIME_IMAGE': runtime_image,
    }
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=360,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_binary_executes_file_editor_and_jupyter_tools(tmp_path):
    binary_dir = _binary_dir()
    script = """
import asyncio
from pathlib import Path

import nexus
from nexus.core import ToolContext, tool_registry
from nexus.extensions.tools.code_interpreter.jupyter_code_interpreter import JupyterCodeInterpreterInput
from nexus.extensions.tools.file_editor.tool import StrReplaceEditorInput


async def main():
    ctx = ToolContext(tool_configs={}, tool_state={'working_dir': str(Path.cwd())})
    jupyter_tool = tool_registry.get('jupyter_code_interpreter')
    editor_tool = tool_registry.get('str_replace_editor')

    jupyter_resp = await jupyter_tool(JupyterCodeInterpreterInput(code='print(21 * 2)'), ctx)
    assert '42' in str(jupyter_resp.output)

    target = Path.cwd() / 'note.txt'
    editor_resp = await editor_tool(
        StrReplaceEditorInput(command='create', path=str(target), file_text='hello from binary'),
        ctx,
    )
    assert 'Error:' not in editor_resp.output[0]['text']
    assert target.read_text() == 'hello from binary'


asyncio.run(main())
"""

    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(binary_dir)},
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_binary_main_agent_runs_file_editor_and_jupyter_tools(tmp_path):
    binary_dir = _binary_dir()
    script = """
import asyncio
import importlib
import json
from pathlib import Path

import nexus
from nexus.core import ExecutionContext, OpenAICompatibleClient, OpenAIConfig
from nexus.core.tool.registry import get_tool_meta
from nexus.extensions.agents.main.agent import MainAgentInput, run_main_agent
from nexus.runtime import RuntimeProvider
from nexus.runtime.client.base import Runtime


class MockToolCall:
    def __init__(self, call_id, name, args):
        self.id = call_id
        self.function = type('Function', (), {'name': name, 'arguments': json.dumps(args)})()


class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, **kwargs):
        result = {'role': 'assistant'}
        if self.content:
            result['content'] = self.content
        if self.tool_calls:
            result['tool_calls'] = [
                {
                    'id': call.id,
                    'function': {
                        'name': call.function.name,
                        'arguments': call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return result


class MockClient(OpenAICompatibleClient):
    def __init__(self, messages):
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self._messages = iter(messages)

    async def _create_completion(self, messages, model=None, **kwargs):
        message = next(self._messages)
        choice = type('Choice', (), {'message': message})()
        return type('Response', (), {'choices': [choice]})()


class LocalRuntime(Runtime):
    async def _invoke(self, request):
        raise NotImplementedError

    async def call_tool(self, import_path, params, context):
        module_name, function_name = import_path.split('::', 1)
        tool_func = getattr(importlib.import_module(module_name), function_name)
        meta = get_tool_meta(tool_func)
        parsed_params = meta.input_model.model_validate(params)
        return await tool_func(parsed_params, context)


class LocalRuntimeProvider(RuntimeProvider):
    async def acquire(self, name, **params):
        runtime = LocalRuntime()
        self._active_runtimes[name] = runtime
        return runtime

    async def release(self, name):
        self._active_runtimes.pop(name, None)


async def main():
    target = Path.cwd() / 'agent-note.txt'
    client = MockClient([
        MockMessage(tool_calls=[
            MockToolCall('tc1', 'jupyter_code_interpreter', {'code': 'print(6 * 7)'}),
            MockToolCall(
                'tc2',
                'str_replace_editor',
                {'command': 'create', 'path': str(target), 'file_text': 'created by main agent'},
            ),
        ]),
        MockMessage(content='done'),
    ])
    ctx = ExecutionContext(
        runtime_provider=LocalRuntimeProvider(),
        openai_clients={'main': client},
        tool_state={'working_dir': str(Path.cwd())},
    )
    async with ctx:
        output = await run_main_agent(MainAgentInput(
            user_prompt='run tools',
            tools=['jupyter_code_interpreter', 'str_replace_editor'],
            parallel_tool_calls=True,
        ))

    assert output.exit_status == 'finished'
    assert output.final_answer == 'done'
    assert target.read_text() == 'created by main agent'
    tool_messages = [msg for msg in output.main_trajectory if msg.get('role') == 'tool']
    assert any('42' in str(msg.get('content')) for msg in tool_messages)


asyncio.run(main())
"""

    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(binary_dir)},
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr or result.stdout
