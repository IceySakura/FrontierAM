import asyncio
import io
import json
import sys
import threading
import time

from pathlib import Path

import pytest

from nexus.extensions.agents.claude_code.interactive import hooks as hook_bridge


def test_ask_user_question_single_answer_preserves_original_input():
    tool_input = {
        'questions': [
            {
                'question': 'Which option?',
                'header': 'Choice',
                'options': [{'label': 'Alpha'}, {'label': 'Beta'}],
            }
        ],
        'extra': 'keep-me',
    }

    output = hook_bridge.build_hook_output('AskUserQuestion', tool_input, 'Beta')

    specific = output['hookSpecificOutput']
    assert specific['permissionDecision'] == 'allow'
    assert specific['updatedInput']['questions'] == tool_input['questions']
    assert specific['updatedInput']['extra'] == 'keep-me'
    assert specific['updatedInput']['answers'] == {'Which option?': 'Beta'}


def test_ask_user_question_multiple_answers_accept_json_and_multiselect():
    tool_input = {
        'questions': [
            {'question': 'Framework?'},
            {'question': 'Features?', 'multiSelect': True},
        ]
    }
    answer = json.dumps({'Framework?': 'React', 'Features?': ['Cache', 'Retry']})

    output = hook_bridge.build_hook_output('AskUserQuestion', tool_input, answer)

    assert output['hookSpecificOutput']['updatedInput']['answers'] == {
        'Framework?': 'React',
        'Features?': 'Cache, Retry',
    }


def test_ask_user_question_none_uses_unattended_fallback():
    output = hook_bridge.build_hook_output('AskUserQuestion', {'questions': []}, None)

    specific = output['hookSpecificOutput']
    assert specific['permissionDecision'] == 'deny'
    assert 'unattended task' in specific['permissionDecisionReason']


@pytest.mark.parametrize(
    ('answer', 'decision'),
    [
        ('approve', 'allow'),
        (None, 'allow'),
        ('reject', 'deny'),
        ('reject: add tests', 'deny'),
        ('please revise compatibility', 'deny'),
    ],
)
def test_exit_plan_mode_answers(answer, decision):
    output = hook_bridge.build_hook_output('ExitPlanMode', {'plan': 'plan'}, answer)
    assert output['hookSpecificOutput']['permissionDecision'] == decision


@pytest.mark.asyncio
async def test_file_bridge_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    interaction_dir = tmp_path / 'interactions'
    turn_file = tmp_path / 'turn'
    turn_file.write_text('3', encoding='utf-8')
    monkeypatch.setenv(hook_bridge.INTERACTION_DIR_ENV, str(interaction_dir))
    monkeypatch.setenv(hook_bridge.SESSION_ID_ENV, 'session-1')
    monkeypatch.setenv(hook_bridge.TURN_FILE_ENV, str(turn_file))
    monkeypatch.setattr(
        sys,
        'stdin',
        io.StringIO(
            json.dumps(
                {
                    'tool_use_id': 'tooluse-ask-1',
                    'tool_input': {
                        'questions': [
                            {
                                'question': 'Which option?',
                                'options': [{'label': 'Alpha'}, {'label': 'Beta'}],
                            }
                        ]
                    },
                }
            )
        ),
    )
    bridge = hook_bridge.FileHookInteractionBridge(interaction_dir)

    hook_outputs: list[dict] = []
    hook_errors: list[BaseException] = []

    def invoke_hook() -> None:
        try:
            hook_outputs.append(hook_bridge.run_hook('AskUserQuestion', wait_timeout=5.0))
        except BaseException as exc:  # pragma: no cover - re-raised below
            hook_errors.append(exc)

    hook_thread = threading.Thread(target=invoke_hook, daemon=True)
    hook_thread.start()
    request = await bridge.wait_request(timeout=2.0)

    assert request is not None
    assert request.session_id == 'session-1'
    assert request.turn_id == 3
    assert request.interaction_id == 'tooluse-ask-1'
    assert request.tool_name == 'AskUserQuestion'
    assert request.timeout_seconds == 5.0
    await bridge.respond(request.interaction_id, 'Beta')

    deadline = time.monotonic() + 6.0
    while hook_thread.is_alive() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert not hook_thread.is_alive(), 'Hook process did not consume the response before its deadline'
    if hook_errors:
        raise hook_errors[0]
    output = hook_outputs[0]
    assert output['hookSpecificOutput']['updatedInput']['answers'] == {'Which option?': 'Beta'}

    # After hook consumes, both request and response files are cleaned up.
    assert await bridge.wait_request(timeout=0.0) is None
    remaining = list(interaction_dir.iterdir())
    assert remaining == []


@pytest.mark.asyncio
async def test_file_bridge_marks_delivered_missing_request_as_expired(tmp_path: Path):
    interaction_dir = tmp_path / 'interactions'
    interaction_dir.mkdir()
    interaction_id = 'tooluse-expired'
    key = hook_bridge.interaction_key(interaction_id)
    request_path = interaction_dir / f'{key}.request.json'
    request_path.write_text(
        json.dumps(
            {
                'version': 1,
                'session_id': 'session-expired',
                'turn_id': 1,
                'interaction_id': interaction_id,
                'tool_name': 'AskUserQuestion',
                'tool_input': {'questions': [{'question': 'Proceed?'}]},
                'created_at': time.time(),
                'timeout_seconds': 10.0,
            }
        ),
        encoding='utf-8',
    )
    bridge = hook_bridge.FileHookInteractionBridge(interaction_dir)

    request = await bridge.wait_request(timeout=0.0)
    assert request is not None
    request_path.unlink()

    expired = await bridge.respond(interaction_id, 'yes')
    unknown = await bridge.respond('never-delivered', 'yes')

    assert expired == {'status': 'expired', 'interaction_id': interaction_id}
    assert unknown == {'status': 'unknown', 'interaction_id': 'never-delivered'}
    assert list(interaction_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_file_bridge_rejects_response_past_request_deadline(tmp_path: Path):
    interaction_dir = tmp_path / 'interactions'
    interaction_dir.mkdir()
    interaction_id = 'tooluse-deadline-expired'
    key = hook_bridge.interaction_key(interaction_id)
    request_path = interaction_dir / f'{key}.request.json'
    request_path.write_text(
        json.dumps(
            {
                'version': 1,
                'session_id': 'session-expired',
                'turn_id': 1,
                'interaction_id': interaction_id,
                'tool_name': 'AskUserQuestion',
                'tool_input': {'questions': [{'question': 'Proceed?'}]},
                'created_at': time.time() - 11.0,
                'timeout_seconds': 10.0,
            }
        ),
        encoding='utf-8',
    )
    bridge = hook_bridge.FileHookInteractionBridge(interaction_dir)

    assert await bridge.wait_request(timeout=0.0) is not None
    result = await bridge.respond(interaction_id, 'late answer')

    assert result == {'status': 'expired', 'interaction_id': interaction_id}
    assert not (interaction_dir / f'{key}.response.json').exists()


@pytest.mark.asyncio
async def test_file_bridge_cleans_response_written_after_hook_removed_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    interaction_dir = tmp_path / 'interactions'
    interaction_dir.mkdir()
    interaction_id = 'tooluse-cleanup-race'
    key = hook_bridge.interaction_key(interaction_id)
    request_path = interaction_dir / f'{key}.request.json'
    response_path = interaction_dir / f'{key}.response.json'
    request_path.write_text(
        json.dumps(
            {
                'version': 1,
                'session_id': 'session-race',
                'turn_id': 1,
                'interaction_id': interaction_id,
                'tool_name': 'AskUserQuestion',
                'tool_input': {'questions': [{'question': 'Proceed?'}]},
                'created_at': time.time(),
                'timeout_seconds': 10.0,
            }
        ),
        encoding='utf-8',
    )
    bridge = hook_bridge.FileHookInteractionBridge(interaction_dir)
    assert await bridge.wait_request(timeout=0.0) is not None

    real_atomic_write = hook_bridge._atomic_write_json

    def remove_request_then_write(path: Path, value: dict) -> None:
        if path == response_path:
            request_path.unlink()
        real_atomic_write(path, value)

    monkeypatch.setattr(hook_bridge, '_atomic_write_json', remove_request_then_write)

    result = await bridge.respond(interaction_id, 'boundary answer')

    assert result == {'status': 'written', 'interaction_id': interaction_id}
    assert list(interaction_dir.iterdir()) == []


def test_hook_timeout_uses_unattended(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    interaction_dir = tmp_path / 'interactions'
    monkeypatch.setenv(hook_bridge.INTERACTION_DIR_ENV, str(interaction_dir))
    monkeypatch.setenv(hook_bridge.SESSION_ID_ENV, 'session-timeout')
    monkeypatch.delenv(hook_bridge.TURN_FILE_ENV, raising=False)
    monkeypatch.setattr(
        sys,
        'stdin',
        io.StringIO(
            json.dumps(
                {
                    'tool_use_id': 'tooluse-timeout',
                    'tool_input': {'questions': [{'question': 'Proceed?'}]},
                }
            )
        ),
    )

    output = hook_bridge.run_hook('AskUserQuestion', wait_timeout=0.0)

    assert output['hookSpecificOutput']['permissionDecision'] == 'deny'
    # Both files cleaned up even on timeout.
    remaining = [p for p in interaction_dir.iterdir() if not p.name.startswith('.')]
    assert remaining == []


@pytest.mark.asyncio
async def test_response_after_real_hook_timeout_is_expired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    interaction_dir = tmp_path / 'interactions'
    monkeypatch.setenv(hook_bridge.INTERACTION_DIR_ENV, str(interaction_dir))
    monkeypatch.setenv(hook_bridge.SESSION_ID_ENV, 'session-late')
    monkeypatch.setattr(
        sys,
        'stdin',
        io.StringIO(
            json.dumps(
                {
                    'tool_use_id': 'tooluse-late',
                    'tool_input': {'questions': [{'question': 'Proceed?'}]},
                }
            )
        ),
    )
    bridge = hook_bridge.FileHookInteractionBridge(interaction_dir)
    # The hook removes its request file as soon as ``wait_timeout`` elapses, so that value is
    # the window in which ``wait_request`` (which polls every ``POLL_INTERVAL`` == 0.05s) can
    # observe it. Keep the window an order of magnitude above the poll interval: a value equal
    # to it makes the test lose the race whenever the loop is busy (e.g. the full suite).
    hook_wait_timeout = hook_bridge.POLL_INTERVAL * 20
    hook_thread = threading.Thread(
        target=hook_bridge.run_hook,
        args=('AskUserQuestion',),
        kwargs={'wait_timeout': hook_wait_timeout},
        daemon=True,
    )
    hook_thread.start()

    request = await bridge.wait_request(timeout=5.0)
    assert request is not None
    await asyncio.to_thread(hook_thread.join, hook_wait_timeout + 5.0)
    assert not hook_thread.is_alive(), 'Hook thread did not exit in time'

    response = await bridge.respond(request.interaction_id, 'late answer')

    assert response == {'status': 'expired', 'interaction_id': 'tooluse-late'}
    assert list(interaction_dir.iterdir()) == []
