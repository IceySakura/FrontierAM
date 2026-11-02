"""V2 Codex structured HITL routing and simulator-answer integrity."""

import io
import json
import zipfile

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.codex.agent import CodexNativeBridgeInput
from nexus.extensions.tasks.os_world_v2.execution.codex_profile import OSWorldCodexExecutionProfile
from nexus.extensions.tasks.os_world_v2.adapter.codex import OSWorldV2CodexProfile
from nexus.extensions.tasks.os_world_v2.adapter.codex_hitl import protocol_bundle
from nexus.extensions.tasks.os_world_v2.adapter.codex_hitl_guest import encode_answers, executed_event


def options(**kwargs):
    return {
        'guide_text': 'Complete the task; end with TASK_DONE.',
        'model_catalog_preset': 'gpt6_catalog_v8',
        'version': '0.154.0',
        **kwargs,
    }


def test_answers_preserve_question_ids_and_free_text():
    questions = [
        {'id': 'elective', 'question': 'Which elective?'},
        {'id': 'credits', 'question': 'How many credits?'},
    ]
    answer = encode_answers(questions, {'elective': 'VI only', 'credits': ['No pass/fail credits']})
    assert answer == {
        'answers': {
            'elective': {'answers': ['VI only']},
            'credits': {'answers': ['No pass/fail credits']},
        }
    }
    assert encode_answers(questions, json.dumps(answer)) == answer
    assert encode_answers(questions[:1], 'Answer in free text')['answers']['elective']['answers'] == [
        'Answer in free text'
    ]


@pytest.mark.parametrize(
    ('questions', 'answer'),
    [
        ([], 'answer'),
        ([{'id': 'a'}, {'id': 'a'}], {'a': 'answer'}),
        ([{'id': ''}], 'answer'),
        ([{'id': 'a'}, {'id': 'b'}], {'a': 'answer'}),
        ([{'id': 'a'}], ''),
        ([{'id': 'a'}], None),
    ],
)
def test_invalid_or_missing_answers_never_become_unattended_approval(questions, answer):
    with pytest.raises((ValueError, TypeError)):
        encode_answers(questions, answer)


def test_protocol_bundle_reuses_original_sources():
    from nexus.extensions.agents.codex import app_server_protocol, app_server_session, app_server_turn_session, events

    with zipfile.ZipFile(io.BytesIO(protocol_bundle())) as archive:
        for module in (app_server_protocol, app_server_session, app_server_turn_session, events):
            path = Path(module.__file__)
            assert archive.read('codex_session/' + path.name) == path.read_bytes()


@pytest.mark.asyncio
async def test_non_hitl_auto_preserves_exec_path(monkeypatch):
    original = AsyncMock(return_value={'success': True})
    monkeypatch.setattr(OSWorldCodexExecutionProfile, 'execute', original)
    profile = OSWorldV2CodexProfile(options(), {'osworld_user_simulator': False}, 'task')
    assert await profile.execute(runtime='runtime', launch='launch', timeout=20) == {'success': True}
    original.assert_awaited_once_with(runtime='runtime', launch='launch', timeout=20)


def test_native_input_transports_callback_without_serializing_it():
    callback = AsyncMock()
    value = CodexNativeBridgeInput.model_validate({
        'execution_profile': 'osworld_v2',
        'execution_options': options(max_turns=2, max_user_requests=3),
        'user_prompt': 'task',
        'get_user_input': callback,
    })
    assert value.get_user_input is callback
    assert 'get_user_input' not in value.model_dump()


def test_hitl_cannot_explicitly_select_noninteractive_exec():
    with pytest.raises(ValueError, match='requires auto or app_server'):
        OSWorldV2CodexProfile(options(session_mode='exec'), {'osworld_user_simulator': True}, 'task')


@pytest.mark.asyncio
async def test_missing_simulator_callback_fails_before_launch():
    profile = OSWorldV2CodexProfile(options(), {'osworld_user_simulator': True}, 'task')
    with pytest.raises(ValueError, match='callback is missing'):
        await profile.execute(runtime=None, launch=None, timeout=20)


def test_user_turn_and_request_limits_are_separate():
    profile = OSWorldV2CodexProfile(options(max_turns=2, max_user_requests=5), {}, 'task')
    assert profile.options.max_turns == 2
    assert profile.options.max_user_requests == 5
    with pytest.raises(ValueError, match='max_turns'):
        OSWorldV2CodexProfile(options(max_turns=0), {}, 'task')


def test_executed_history_projects_completed_commands_not_proposals(tmp_path):
    from nexus.extensions.agents.codex.events import CodexEvent
    from nexus.extensions.tasks.os_world_v2.adapter.adapter import executed_cli_actions

    def event(direction, kind):
        return CodexEvent(
            seq=0,
            direction=direction,
            type=kind,
            data={
                'item': {
                    'type': 'commandExecution',
                    'command': 'printf executed',
                    'exitCode': 0,
                    'status': 'completed',
                    'aggregatedOutput': 'executed',
                }
            },
        )

    assert executed_event(event('outbound', 'item.completed')) is None
    assert executed_event(event('inbound', 'item.started')) is None
    projection = executed_event(event('inbound', 'item.completed'))
    (tmp_path / 'codex_stream.jsonl').write_text(json.dumps(projection) + '\n')
    assert executed_cli_actions(tmp_path, 'codex_native_bridge') == [{'command': 'printf executed', 'exit_code': 0}]


def test_shared_v1_profile_does_not_accept_v2_dialogue_options():
    with pytest.raises(ValueError, match='max_user_requests'):
        OSWorldCodexExecutionProfile(options(max_user_requests=3), {}, 'task')
