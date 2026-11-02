"""AGS file transport for the shared Codex app-server protocol/session."""

import asyncio
import base64
import hashlib
import io
import json
import shlex
import zipfile

from dataclasses import replace
from pathlib import Path

from nexus.core import UserInputRequest
from nexus.extensions.agents.codex import app_server_protocol, app_server_session, app_server_turn_session, events
from nexus.extensions.tasks.os_world_v2.adapter import codex_hitl_guest
from nexus.extensions.tasks.os_world_v2.adapter.control_reads import read_control
from nexus.extensions.tasks.os_world_v2.adapter.preparation import run_preparation
from nexus.extensions.tasks.os_world_v2.adapter.prompts import RESOURCE_POLICY
from nexus.extensions.tasks.os_world_v2.execution.codex import TaskCodexAdapter


PYTHON_DEPS = '/tmp/nexus-codex-hitl-pydeps'


def protocol_bundle():
    """Stage existing source modules unchanged, without importing Nexus in guest."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('codex_session/__init__.py', '')
        for module in (app_server_protocol, app_server_session, app_server_turn_session, events):
            archive.writestr('codex_session/' + Path(module.__file__).name, Path(module.__file__).read_bytes())
    return output.getvalue()


class HITLCodexAdapter(TaskCodexAdapter):
    def __init__(self, runtime, *, diagnostic_dir, options, callback, has_simulator):
        super().__init__(runtime, diagnostic_dir=diagnostic_dir)
        self.options = options
        self.callback = callback
        self.has_simulator = has_simulator
        self.session_result = None
        self.appserver_events = None

    async def _prepare(self, request, paths):
        await super()._prepare(request, paths)
        python_probe = f'PYTHONPATH={PYTHON_DEPS} python3 -c ' + shlex.quote(
            'import pydantic; assert int(pydantic.__version__.split(".")[0])>=2'
        )
        probe = await self._command(python_probe)
        if not self._ok(probe):
            installed = await run_preparation(
                self.runtime,
                command=f'python3 -m pip install --disable-pip-version-check --target {PYTHON_DEPS} pydantic==2.12.5',
                timeout=180,
            )
            if installed.return_code != 0 or not self._ok(await self._command(python_probe)):
                raise RuntimeError('Cannot prepare isolated Codex app-server Python dependencies')
        await self._stage(paths.dir + '/protocol.zip', protocol_bundle())
        spec = {
            'executable': request.codex_executable,
            'working_dir': request.working_dir,
            'model': request.model,
            'timeout': request.timeout,
            'max_turns': self.options.max_turns,
            'max_user_requests': self.options.max_user_requests,
            'hook_timeout': self.options.hook_input_timeout,
            'has_simulator': self.has_simulator,
            'reasoning_effort': self.options.reasoning_effort,
            'developer_instructions': RESOURCE_POLICY,
            'python_dependencies': PYTHON_DEPS,
        }
        await self._stage(paths.dir + '/session.json', json.dumps(spec).encode())
        if self.diagnostic_dir:
            (self.diagnostic_dir / 'codex_protocol_manifest.json').write_text(
                json.dumps(
                    {
                        module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                        for module in (
                            app_server_protocol,
                            app_server_session,
                            app_server_turn_session,
                            events,
                            codex_hitl_guest,
                        )
                    },
                    indent=2,
                )
            )
        await self._stage(
            paths.dir + '/driver.py',
            b'#!/usr/bin/python3\n' + Path(codex_hitl_guest.__file__).read_bytes(),
        )
        result = await self._command('chmod 700 ' + shlex.quote(paths.dir + '/driver.py'))
        if not self._ok(result):
            raise RuntimeError('Cannot make Codex app-server driver executable')

    def _safe_launch(self, request, paths):
        # Reuse the verified PID/process-group lifecycle; driver launches the
        # actual Codex app-server as its child in that same group.
        return super()._safe_launch(replace(request, codex_executable=paths.dir + '/driver.py'), paths)

    async def _interactions(self, paths):
        sequence = 0
        while True:
            script = (
                'import pathlib,json;'
                f'p=pathlib.Path({(paths.dir + "/interaction.json")!r});'
                'print(p.read_text() if p.exists() else "null")'
            )
            response = await read_control(
                self.runtime,
                'python3 -c ' + shlex.quote(script),
                label='codex_hitl_requests',
                timeout=20,
                diagnostic_dir=self.diagnostic_dir,
            )
            if response.return_code != 0:
                raise RuntimeError('Cannot read Codex interaction mailbox')
            request = json.loads(response.stdout or 'null')
            if request and request['sequence'] != sequence:
                if request['sequence'] != sequence + 1:
                    raise ValueError('Codex interaction sequence gap')
                sequence = request['sequence']
                params = request['params']
                record = {'sequence': sequence, 'kind': request['kind'], 'params': params}
                try:
                    if request['kind'] == 'turn':
                        answer = (
                            await asyncio.wait_for(
                                self.callback(UserInputRequest(source='turn', hint='Codex turn complete')),
                                self.options.hook_input_timeout,
                            )
                            if self.callback
                            else None
                        )
                    elif request['kind'] == 'user_input' and self.has_simulator and self.callback:
                        questions = params.get('questions')
                        if not isinstance(questions, list):
                            raise ValueError('Codex question list missing')
                        # Validate schema before requesting any simulator answer.
                        codex_hitl_guest.encode_answers(
                            questions, {q.get('id'): 'schema probe' for q in questions if isinstance(q, dict)}
                        )
                        answers = {}
                        for question in questions:
                            user_request = UserInputRequest(
                                source='hook',
                                hint=question['question'],
                                hook={
                                    'tool_name': 'request_user_input',
                                    'tool_input': {
                                        'questions': [question],
                                    },
                                },
                                metadata={'interaction_id': str(sequence), 'question_id': question['id']},
                            )
                            value = await asyncio.wait_for(self.callback(user_request), self.options.hook_input_timeout)
                            if not isinstance(value, str) or not value.strip():
                                raise RuntimeError('V2 user simulator did not answer the Codex question')
                            # V2 shared callback returns JSON keyed by question text.
                            try:
                                decoded = json.loads(value)
                            except ValueError:
                                decoded = None
                            if isinstance(decoded, dict) and question['question'] in decoded:
                                value = decoded[question['question']]
                            answers[question['id']] = value
                        answer = codex_hitl_guest.encode_answers(questions, answers)
                    else:
                        raise RuntimeError('Unsupported Codex interaction')
                    result = {'sequence': sequence, 'ok': True, 'answer': answer}
                    record.update(ok=True, answer=answer)
                except Exception as exc:
                    result = {'sequence': sequence, 'ok': False, 'error_type': type(exc).__name__}
                    record.update(ok=False, error_type=type(exc).__name__)
                if self.diagnostic_dir:
                    with (self.diagnostic_dir / 'codex_interactions.jsonl').open('a') as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                path = f'{paths.dir}/answer-{sequence}.json'
                data = json.dumps(result).encode()
                await self._stage(path + '.tmp', data)
                moved = await self._command(f'mv {shlex.quote(path + ".tmp")} {shlex.quote(path)}')
                if not self._ok(moved):
                    raise RuntimeError('Cannot deliver Codex user answer')
            await asyncio.sleep(1)

    async def _poll(self, paths, timeout):
        process = asyncio.create_task(super()._poll(paths, timeout))
        interactions = asyncio.create_task(self._interactions(paths))
        try:
            done, _ = await asyncio.wait((process, interactions), return_when=asyncio.FIRST_COMPLETED)
            if interactions in done:
                interactions.result()
                raise RuntimeError('Codex interaction service stopped')
            return process.result()
        finally:
            process.cancel()
            interactions.cancel()
            await asyncio.gather(process, interactions, return_exceptions=True)

    async def _capture(self, paths):
        # Save structured evidence before the inherited cleanup removes guest files.
        for filename in ('session-result.json', 'codex_appserver_events.jsonl', 'appserver_stderr.txt'):
            parts = []
            for offset in range(0, 64 * 1024**2, 48 * 1024):
                script = (
                    'import base64;'
                    f'f=open({(paths.dir + "/" + filename)!r},"rb");f.seek({offset});'
                    'print(base64.b64encode(f.read(49152)).decode())'
                )
                result = await self._command('python3 -c ' + shlex.quote(script), timeout=10)
                if not self._ok(result):
                    self.diagnostics.setdefault('structured_read_errors', {})[filename] = {
                        'offset': offset,
                        'return_code': result.return_code,
                    }
                    break
                block = base64.b64decode((result.stdout or '').strip(), validate=True)
                parts.append(block)
                if len(block) < 49152:
                    break
            else:
                raise RuntimeError('Codex structured evidence exceeds 64 MiB bound')
            if parts:
                value = b''.join(parts).decode()
                if self.diagnostic_dir:
                    (self.diagnostic_dir / filename).write_text(value)
                if filename == 'session-result.json':
                    parsed = json.loads(value)
                    if parsed.get('exit_status') not in {'finished', 'max_iterations', 'tool_error'}:
                        raise ValueError('Invalid Codex session status')
                    self.session_result = parsed
                if filename == 'codex_appserver_events.jsonl':
                    self.appserver_events = [
                        events.CodexEvent.model_validate_json(line) for line in value.splitlines() if line.strip()
                    ]
        await super()._capture(paths)
