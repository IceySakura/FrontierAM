"""Guest app-server driver using the shared Codex JSON-RPC client and session.

This file and the shared pure protocol modules are staged into a private zip.
The mailbox carries questions and answers only; model traffic uses the normal
Nexus Responses proxy. No PTY or additional guest network service is required.
"""

import asyncio
import json
import os
import re
import sys
import time

from contextlib import suppress
from pathlib import Path


def encode_answers(questions, answer):
    """Encode simulator replies by question ID without selecting an option."""
    if not isinstance(questions, list) or not questions:
        raise ValueError('Codex user request has no questions')
    ids = [q.get('id') for q in questions if isinstance(q, dict)]
    if len(ids) != len(questions) or any(not isinstance(q, str) or not q for q in ids) or len(set(ids)) != len(ids):
        raise ValueError('Codex question IDs must be unique nonempty strings')
    values = answer
    if isinstance(values, str):
        values = {ids[0]: values} if len(ids) == 1 else json.loads(values)
    if isinstance(values, dict) and isinstance(values.get('answers'), dict):
        values = values['answers']
    if not isinstance(values, dict):
        raise ValueError('Simulator reply must map question IDs to answers')
    result = {}
    for key in ids:
        value = values.get(key)
        if isinstance(value, dict):
            value = value.get('answers')
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value):
            raise ValueError('Simulator omitted a Codex question')
        result[key] = {'answers': value}
    return {'answers': result}


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    os.replace(tmp, path)


def executed_event(event):
    """Project completed app-server observations into the existing exec schema."""
    item = event.data.get('item') or {}
    if event.direction != 'inbound' or event.type != 'item.completed':
        return None
    if item.get('type') == 'commandExecution':
        return {
            'type': 'item.completed',
            'item': {
                'type': 'command_execution',
                'id': item.get('id'),
                'command': item.get('command'),
                'exit_code': item.get('exitCode'),
                'aggregated_output': item.get('aggregatedOutput'),
                'status': item.get('status'),
            },
        }
    if item.get('type') == 'agentMessage':
        return {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': item.get('text', '')}}
    return None


async def run(root):
    # These modules are copied byte-for-byte from nexus.extensions.agents.codex.
    from codex_session.app_server_protocol import AppServerClient
    from codex_session.app_server_turn_session import AppServerTurnSession
    from codex_session.events import reduce_app_server_events

    spec = json.loads((root / 'session.json').read_text())
    started = time.monotonic()
    deadline = started + spec['timeout']
    sequence = 0
    questions_count = 0
    interaction_error = None
    turns = 0
    client = None
    session = None
    input_lock = asyncio.Lock()
    report = {'success': False, 'exit_status': 'tool_error', 'error': None, 'turns': 0}
    journal = (root / 'codex_appserver_events.jsonl').open('w')

    class JournalEvents(list):
        def append(self, event):
            super().append(event)
            journal.write(event.model_dump_json() + '\n')
            journal.flush()
            projection = executed_event(event)
            if projection is not None:
                print(json.dumps(projection), flush=True)

    class JournalClient(AppServerClient):
        def __init__(self, process):
            super().__init__(process)
            self.event_stream.events = JournalEvents()

    async def ask_unlocked(kind, params):
        nonlocal sequence
        sequence += 1
        atomic(root / 'interaction.json', {'sequence': sequence, 'kind': kind, 'params': params})
        response = root / f'answer-{sequence}.json'
        until = min(deadline, time.monotonic() + spec['hook_timeout'])
        while time.monotonic() < until:
            if response.is_file():
                value = json.loads(response.read_text())
                if value.get('sequence') != sequence:
                    raise ValueError('Codex response sequence mismatch')
                if not value.get('ok'):
                    raise RuntimeError('Codex user-input provider failed')
                return value.get('answer')
            await asyncio.sleep(0.2)
        raise TimeoutError('Codex user-input deadline exceeded')

    async def ask(kind, params):
        async with input_lock:
            return await ask_unlocked(kind, params)

    async def user_input(params):
        nonlocal questions_count, interaction_error
        assert session is not None
        try:
            if not spec['has_simulator']:
                raise RuntimeError('No simulator configured for Codex user request')
            if params.get('threadId') != session.thread_id or params.get('turnId') != session.turn_id:
                raise ValueError('Codex question does not belong to the active thread/turn')
            questions_count += 1
            if questions_count > spec['max_user_requests']:
                raise RuntimeError('Codex user-request limit exceeded')
            answer = await ask('user_input', params)
            return encode_answers(params.get('questions'), answer)
        except Exception as exc:
            interaction_error = f'{type(exc).__name__}: {exc}'
            # Fail the owning session too, not just one JSON-RPC tool request.
            await session._on_error(exc)
            raise

    async def unexpected(params):
        nonlocal interaction_error
        exc = RuntimeError('Unexpected approval/elicitation request in unattended OSWorld session')
        interaction_error = str(exc)
        if session is not None:
            await session._on_error(exc)
        raise exc

    try:
        client = await JournalClient.start(
            [spec['executable'], 'app-server', '-c', 'model_reasoning_effort=' + json.dumps(spec['reasoning_effort'])],
            cwd=spec['working_dir'],
            env=dict(os.environ),
        )
        client.register_request_handler('item/tool/requestUserInput', user_input)
        for method in (
            'item/commandExecution/requestApproval',
            'item/fileChange/requestApproval',
            'item/permissions/requestApproval',
            'mcpServer/elicitation/request',
        ):
            client.register_request_handler(method, unexpected)
        session = AppServerTurnSession(client)
        async with session:
            await asyncio.wait_for(
                session.start_thread(
                    working_dir=spec['working_dir'],
                    model=spec['model'],
                    developer_instructions=spec['developer_instructions'],
                ),
                timeout=min(60, max(0.01, deadline - time.monotonic())),
            )
            prompt = (root / 'prompt.txt').read_text()
            while prompt is not None and turns < spec['max_turns']:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Codex dialogue budget exhausted')
                turns += 1
                result = await asyncio.wait_for(session.run(prompt), timeout=remaining)
                reduced = reduce_app_server_events(
                    client.event_stream.events, thread_id=session.thread_id, turn_id=session.turn_id
                )
                if interaction_error:
                    raise RuntimeError(interaction_error)
                if not result.success:
                    raise RuntimeError(result.error)
                report.update(success=True, exit_status='finished', final_answer=reduced.final_message)
                if re.search(r'^\s*TASK_DONE\s*$', reduced.final_message, re.MULTILINE) or not spec['has_simulator']:
                    break
                if turns >= spec['max_turns']:
                    report['exit_status'] = 'max_iterations'
                    break
                prompt = await ask('turn', {'threadId': session.thread_id, 'turnId': session.turn_id})
                if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
                    raise ValueError('Invalid continuation prompt')
    except BaseException as exc:
        report.update(success=False, exit_status='tool_error', error=f'{type(exc).__name__}: {exc}')
    finally:
        if session is not None:
            with suppress(Exception):
                await session.interrupt()
        if client is not None:
            try:
                await client.close()
            except Exception as exc:
                report.update(
                    success=False, exit_status='tool_error', error=f'Codex close failed: {type(exc).__name__}'
                )
            (root / 'appserver_stderr.txt').write_text('\n'.join(client.stderr_lines))
        journal.close()
        report.update(
            turns=turns,
            user_requests=questions_count,
            thread_id=session.thread_id if session else None,
            elapsed_s=time.monotonic() - started,
        )
        atomic(root / 'session-result.json', report)
        print(
            json.dumps({
                'type': 'turn.completed' if report['success'] else 'turn.failed',
                **({} if report['success'] else {'error': {'message': report['error']}}),
            }),
            flush=True,
        )
    return 0 if report['success'] else 1


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    os.umask(0o077)
    spec = json.loads((root / 'session.json').read_text())
    sys.path.insert(0, spec['python_dependencies'])
    sys.path.insert(0, str(root / 'protocol.zip'))
    raise SystemExit(asyncio.run(run(root)))
