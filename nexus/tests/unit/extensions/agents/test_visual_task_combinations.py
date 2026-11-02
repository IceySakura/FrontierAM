"""Run identity/visual through the real agent loop and local runtime."""

from __future__ import annotations

import base64
import copy
import json
import zipfile

from pathlib import Path
from typing import Any

import pytest
import yaml

from PIL import Image
from pydantic import BaseModel

from nexus.core import OpenAICompatibleClient, OpenAIConfig, ToolContext, ToolResponse, tool_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.agents.visual_agent.resources.images import upload_locations
from nexus.runner.single import RunInput, run


class ReadFileInput(BaseModel):
    path: str


@tool_registry.register('visual_test_read_file')
async def read_file(params: ReadFileInput, context: ToolContext) -> ToolResponse:
    """Read the local runtime workspace without optional editor dependencies."""
    return ToolResponse(output=[{'type': 'text', 'text': Path(params.path).read_text()}])


class ScriptedVisualClient(OpenAICompatibleClient):
    def __init__(self, read_paths: list[Path]) -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.read_paths = read_paths
        self.requests: list[list[dict[str, Any]]] = []

    async def _create_completion(self, messages, model=None, **kwargs) -> ExtendedChatCompletion:
        self.requests.append(copy.deepcopy(messages))
        if len(self.requests) == 1 and self.read_paths:
            message = {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': f'read_{index}',
                        'type': 'function',
                        'function': {
                            'name': 'visual_test_read_file',
                            'arguments': json.dumps({'path': str(path)}),
                        },
                    }
                    for index, path in enumerate(self.read_paths)
                ],
            }
            reason = 'tool_calls'
        else:
            message = {'role': 'assistant', 'content': 'VISUAL_COMBINATION_OK'}
            reason = 'stop'
        return ExtendedChatCompletion.model_validate({
            'id': f'completion_{len(self.requests)}',
            'created': 0,
            'model': 'scripted-local',
            'choices': [{'index': 0, 'finish_reason': reason, 'message': message}],
        })


def build_run_input(
    tmp_path: Path, task_name: str, data: dict, client: ScriptedVisualClient, *, runtime: bool
) -> RunInput:
    config = {
        'task_name': task_name,
        'task_config': {key: value for key, value in data.items() if key != 'workspace'}
        if task_name == 'visual'
        else {},
        'agent_name': 'visual_agent',
        'processors': {'metrics': {'name': 'main_agent_metrics'}},
        'agent_config': {
            'tools': ['visual_test_read_file'],
            'max_iterations': 3,
            'tool_error_continue': False,
            'show_image_dimensions': True,
            'workspace': data.get('workspace', '/tmp'),
        },
    }
    if task_name == 'identity':
        config['agent_config'].update(data)
    if runtime:
        config['runtime_provider_name'] = 'local'
    path = tmp_path / f'{task_name}.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    result = RunInput.from_file(path)
    result.openai_clients = {'main': {'instance': client}}
    return result


@pytest.mark.parametrize('task_name', ['identity', 'visual'])
async def test_task_combination_runs_text_without_runtime(tmp_path: Path, task_name: str) -> None:
    data = {'user_prompt': 'Say hello', 'history': [{'role': 'assistant', 'content': 'Previous answer'}]}
    client = ScriptedVisualClient([])
    result = await run(build_run_input(tmp_path, task_name, data, client, runtime=False))

    assert result.success, result.error
    assert result.agent_output.exit_status == 'finished'
    assert result.processor_results['metrics']['agent.iterations'] == 1
    assert result.agent_output.iterations == 1
    assert result.agent_output.final_answer == 'VISUAL_COMBINATION_OK'
    assert result.agent_output.workspace == '/tmp'
    assert result.task_output.model_dump() == (
        {'score': None, 'evaluation_details': None} if task_name == 'visual' else {}
    )
    assert client.requests[0][1] == data['history'][0]
    assert client.requests[0][-1] == {'role': 'user', 'content': 'Say hello'}


@pytest.mark.parametrize('task_name', ['identity', 'visual'])
@pytest.mark.parametrize('zipped', [False, True])
async def test_task_combination_runs_media_workspace_and_real_tool_loop(
    tmp_path: Path, task_name: str, monkeypatch: pytest.MonkeyPatch, zipped: bool
) -> None:
    from nexus.extensions.agents.visual_agent.resources import images as images_module

    monkeypatch.setattr(images_module, 'RUNTIME_UPLOAD_DIR', str(tmp_path / 'media'))
    picture = tmp_path / 'picture.png'
    Image.new('RGB', (12, 8), 'red').save(picture)
    document = tmp_path / 'document.txt'
    document.write_text('DOCUMENT_MARKER')
    source = tmp_path / 'inputs'
    (source / 'empty').mkdir(parents=True)
    (source / 'file.txt').write_text('FILE_MARKER')
    (source / '.hidden').write_text('DIRECTORY_MARKER')
    (source / 'nested').mkdir()
    (source / 'nested' / 'data.txt').write_text('ZIP_MARKER')
    if zipped:
        archive = tmp_path / 'project.zip'
        with zipfile.ZipFile(archive, 'w') as bundle:
            for path in source.rglob('*'):
                bundle.write(path, path.relative_to(source).as_posix())
        source = archive
    target = tmp_path / 'sandbox'
    read_paths = [target / 'file.txt', target / 'nested' / 'data.txt']
    data = {
        'user_prompt': [
            {'type': 'text', 'text': 'Inspect the picture and files'},
            {'type': 'image_url', 'image_url': {'url': str(picture)}},
            {'type': 'file', 'file': {'url': str(document)}},
        ],
        'history': [{'role': 'assistant', 'content': 'Previous observation'}],
        'workspace': str(target),
        'upload_workspace': str(source),
    }
    client = ScriptedVisualClient(read_paths)
    run_input = build_run_input(tmp_path, task_name, data, client, runtime=True)
    before = copy.deepcopy(run_input.task_config)
    result = await run(run_input)

    assert result.success, result.error
    assert result.agent_output.exit_status == 'finished'
    assert result.agent_output.final_answer == 'VISUAL_COMBINATION_OK'
    assert result.processor_results['metrics']['agent.iterations'] == 2
    assert result.processor_results['metrics']['agent.tool_call_count'] == 2
    assert result.processor_results['metrics']['agent.exit_status'] == 'finished'
    assert result.agent_output.iterations == 2
    assert result.task_output.model_dump() == (
        {'score': None, 'evaluation_details': None} if task_name == 'visual' else {}
    )
    assert len(client.requests) == 2
    assert run_input.task_config == before

    assert read_paths[0].read_text() == 'FILE_MARKER'
    assert read_paths[1].read_text() == 'ZIP_MARKER'
    assert (target / '.hidden').read_text() == 'DIRECTORY_MARKER'
    assert (target / 'empty').is_dir()
    assert result.agent_output.workspace == str(target)

    parts = client.requests[0][-1]['content']
    image_part = next(part for part in parts if part['type'] == 'image_url')
    assert image_part['image_url']['url'].startswith('data:image/png;base64,')
    image_index = parts.index(image_part)
    note_index = image_index - 1
    assert '![picture](' in parts[note_index]['text']
    assert '(12x8 px)' in parts[note_index]['text']
    assert base64.b64decode(image_part['image_url']['url'].split(',', 1)[1]) == picture.read_bytes()
    locations = upload_locations([str(picture), str(document)])
    assert Path(locations[str(picture)]).read_bytes() == picture.read_bytes()
    assert Path(locations[str(document)]).read_text() == 'DOCUMENT_MARKER'
    notes = '\n'.join(part['text'] for part in parts if part['type'] == 'text')
    assert '(12x8 px)' in notes
    assert locations[str(document)] in notes
    assert str(target) in notes
    tool_messages = [message for message in client.requests[1] if message['role'] == 'tool']
    assert len(tool_messages) == 2
    assert 'FILE_MARKER' in json.dumps(tool_messages)
    assert 'ZIP_MARKER' in json.dumps(tool_messages)
    assert result.agent_output.main_trajectory[-1]['content'] == 'VISUAL_COMBINATION_OK'


@pytest.mark.parametrize('task_name', ['identity', 'visual'])
async def test_task_combination_stops_before_model_if_workspace_fails(tmp_path: Path, task_name: str) -> None:
    data = {
        'user_prompt': 'Inspect files',
        'upload_workspace': str(tmp_path / 'missing'),
        'workspace': str(tmp_path / 'sandbox'),
    }
    client = ScriptedVisualClient([])
    result = await run(build_run_input(tmp_path, task_name, data, client, runtime=True))
    assert result.success is False
    assert result.error.type == 'FileNotFoundError'
    assert client.requests == []
