from __future__ import annotations

import copy
import json

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.projects.hy_live.example.cases import get_case
from nexus.extensions.projects.hy_live.example.validate import TRAJECTORY_FILES, validate_run
from hylive.prompts import (
    FAILURE_TEMPLATE,
    NOTICE_TEMPLATE,
    PERMISSION_GUIDANCE,
    notice_fact,
)


SNAKE = 'task_1'


def _user(text: str) -> dict[str, Any]:
    return {'role': 'user', 'content': text}


def _say(text: str) -> dict[str, Any]:
    return {'role': 'assistant', 'content': text}


def _call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    """An assistant tool call as ``cerebellum.jsonl`` exports it, arguments parsed."""
    return {
        'role': 'assistant',
        'content': '',
        'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}],
    }


def _result(call_id: str, name: str, content: Any) -> dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return {'role': 'tool', 'content': text, 'tool_call_id': call_id, 'name': name}


def _notice(fact: str) -> dict[str, Any]:
    return _user(NOTICE_TEMPLATE.format(fact=fact))


PERMISSION_FACT = notice_fact(
    'permission', {'work': '贪吃蛇', 'operation': 'List working directory contents', 'detail': '{"command": "ls"}'}
)
COMPLETION_FACT = notice_fact('completed', {'work': '贪吃蛇', 'summary': '贪吃蛇做好了，产物是 snake.html。'})


@pytest.fixture
def canonical() -> dict[str, Any]:
    """One delegate, one permission allowed for good, one completion: every generic check holds."""
    cerebellum = [
        {'role': 'system', 'content': 'sys'},
        _user('写一个贪吃蛇'),
        _call('c1', 'delegate', text='写一个贪吃蛇'),
        _result('c1', 'delegate', {'status': 'submitted'}),
        _say('我这就去做一个贪吃蛇。'),
        _notice(f'{PERMISSION_FACT}\n{PERMISSION_GUIDANCE}'),
        _say('贪吃蛇要先看一下工作目录，这次允许吗？以后要不要都放行？'),
        _user('都允许'),
        _call('c2', 'answer_permission', text='贪吃蛇这项工作以后都允许'),
        _result('c2', 'answer_permission', {'status': 'submitted'}),
        _say('好，以后都放行。'),
        _notice(COMPLETION_FACT),
        _say('贪吃蛇做好了。'),
    ]
    midbrain = [
        {'role': 'system', 'content': 'midbrain'},
        _call('m1', 'reply', status='accepted', say='已受理「贪吃蛇」。'),
        _result('m1', 'reply', {'ok': True}),
    ]
    return {
        'cerebellum_trajectory': {'tools': [], 'messages': cerebellum},
        'midbrain_trajectory': {'tools': [], 'messages': midbrain},
        'dialogue_trajectory': {'tools': {}, 'system': {}, 'turns': []},
        'tasks': [{'task_id': SNAKE, 'status': 'done'}],
        'delivery_events': [
            {'kind': 'permission_request', 'task_id': SNAKE},
            {'kind': 'task_delivery', 'task_id': SNAKE},
        ],
    }


def _cerebellum(output: dict[str, Any]) -> list[dict[str, Any]]:
    return output['cerebellum_trajectory']['messages']


def test_canonical_run_passes_every_check(canonical: dict[str, Any]) -> None:
    result = validate_run(canonical)
    assert result['passed'], result['error']
    assert all(result['checks'].values())


def _web_trajectory(*, search: str, fetch: str) -> list[dict[str, Any]]:
    return [
        _call('search', 'WebSearch', query='Typesafe JEV'),
        _result('search', 'WebSearch', search),
        _call('fetch1', 'WebFetch', url='https://docs.typesafe.ai/introduction'),
        _result('fetch1', 'WebFetch', fetch),
        _call('fetch2', 'WebFetch', url='https://typesafe.ai/blog/introducing-system-one-models-and-jev'),
        _result('fetch2', 'WebFetch', fetch),
    ]


def test_web_regression_checks_native_results_and_distinct_source_urls(canonical: dict[str, Any]) -> None:
    canonical['tasks'][0]['result'] = {
        'trajectory': _web_trajectory(
            search='# Search Results\nhttps://docs.typesafe.ai/introduction\n' + 'Official source summary. ' * 10,
            fetch='# Introduction\n' + 'Jev is TypeSafe flagship model and the first System One model. ' * 10,
        )
    }
    checks = ('native_web_search_succeeded', 'native_web_fetch_succeeded', 'native_web_fetch_two_sources', 'tasks_done')
    result = validate_run(canonical, case_checks=checks)

    assert result['passed'], result['error']


@pytest.mark.parametrize(
    ('search', 'fetch', 'failed'),
    [
        (
            'Error: HTTP 401, invalid search credential. ' * 10,
            'Fetched page body. ' * 20,
            'native_web_search_succeeded',
        ),
        ('https://docs.typesafe.ai/introduction ' * 10, 'Error: fetch failed. ' * 20, 'native_web_fetch_two_sources'),
        ('https://docs.typesafe.ai/introduction ' * 10, '', 'native_web_fetch_two_sources'),
    ],
)
def test_web_regression_rejects_failed_or_empty_native_results(
    canonical: dict[str, Any], search: str, fetch: str, failed: str
) -> None:
    canonical['tasks'][0]['result'] = {'trajectory': _web_trajectory(search=search, fetch=fetch)}
    result = validate_run(canonical, case_checks=('native_web_search_succeeded', 'native_web_fetch_two_sources'))

    assert not result['passed'], 'Web calls alone do not prove successful internet access'
    assert result['checks'][failed] is False


def test_web_fetch_regression_requires_two_distinct_sources_and_a_done_task(canonical: dict[str, Any]) -> None:
    trajectory = _web_trajectory(search='', fetch='Jev introduction page content. ' * 10)
    trajectory[4] = _call('fetch2', 'WebFetch', url='https://docs.typesafe.ai/introduction')
    canonical['tasks'][0].update(status='failed', result={'trajectory': trajectory})
    result = validate_run(
        canonical, case_checks=('native_web_fetch_succeeded', 'native_web_fetch_two_sources', 'tasks_done')
    )

    assert result['checks']['native_web_fetch_succeeded'] is True
    assert result['checks']['native_web_fetch_two_sources'] is False
    assert result['checks']['tasks_done'] is False


def _replace(index: int, message: dict[str, Any]) -> Callable[[dict[str, Any]], None]:
    def mutate(output: dict[str, Any]) -> None:
        _cerebellum(output)[index] = message

    return mutate


def _drop_midbrain(output: dict[str, Any]) -> None:
    output['midbrain_trajectory']['messages'] = []


def _drop_tool_result(output: dict[str, Any]) -> None:
    del _cerebellum(output)[3]


def _drop_last_reply(output: dict[str, Any]) -> None:
    _cerebellum(output).pop()


def _task_status(status: str) -> Callable[[dict[str, Any]], None]:
    def mutate(output: dict[str, Any]) -> None:
        output['tasks'][0]['status'] = status

    return mutate


def _no_tasks(output: dict[str, Any]) -> None:
    output['tasks'] = []


def _no_delivery(output: dict[str, Any]) -> None:
    output['delivery_events'] = [event for event in output['delivery_events'] if event['kind'] != 'task_delivery']


@pytest.mark.parametrize(
    ('mutate', 'failing_check'),
    [
        (_drop_midbrain, 'trajectories_present'),
        (
            _replace(3, _result('c1', 'delegate', {'status': 'failed', 'error': '没有这个工具'})),
            'async_tools_submitted',
        ),
        (_drop_tool_result, 'async_tools_submitted'),
        (_replace(11, _user('[进展]\n「贪吃蛇」正在写代码。')), 'notices_well_formed'),
        (_replace(11, _notice('')), 'notices_well_formed'),
        (_replace(11, _user('[通知]\n有一项工作做完了。')), 'notices_well_formed'),
        (_replace(11, _notice(f'{COMPLETION_FACT}\npermission_id: 7214922ab288')), 'notices_hide_ledger_ids'),
        (_replace(11, _notice(f'{COMPLETION_FACT}\n路径：{SNAKE}')), 'notices_hide_ledger_ids'),
        (_replace(12, _say('<｜｜DSML｜｜invoke name="delegate">')), 'no_tool_markup_leaked'),
        (_replace(12, _say('<tool_calls>[{"name": "delegate"}]</tool_calls>')), 'no_tool_markup_leaked'),
        (_drop_last_reply, 'every_notice_answered'),
        (_replace(12, _say('  ')), 'every_notice_answered'),
        (_task_status('running'), 'tasks_finished'),
        (_no_tasks, 'tasks_finished'),
        (_no_delivery, 'finished_tasks_delivered'),
    ],
    ids=[
        'midbrain-not-exported',
        'hand-off-failed',
        'hand-off-without-result',
        'progress-notice-is-never-sent',
        'empty-notice-fact',
        'notice-without-template-head',
        'notice-leaks-permission-id',
        'notice-leaks-task-path',
        'dsml-in-reply',
        'tool-calls-tag-in-reply',
        'last-notice-unanswered',
        'blank-reply-to-notice',
        'task-still-running',
        'no-task',
        'finished-task-never-delivered',
    ],
)
def test_validator_names_the_broken_check(
    canonical: dict[str, Any], mutate: Callable[[dict[str, Any]], None], failing_check: str
) -> None:
    broken = copy.deepcopy(canonical)
    mutate(broken)
    result = validate_run(broken)
    assert not result['passed']
    assert not result['checks'][failing_check]
    assert failing_check in result['error']


def test_web_search_results_are_not_hand_offs(canonical: dict[str, Any]) -> None:
    _cerebellum(canonical)[2:2] = [
        _call('s1', 'web_search', query=['上海今天天气']),
        _result('s1', 'web_search', '搜索失败：搜索失败：All connection attempts failed'),
        _say('没查到上海今天的温度。'),
    ]
    assert validate_run(canonical)['passed']


def test_a_failure_message_is_well_formed(canonical: dict[str, Any]) -> None:
    _cerebellum(canonical)[11:11] = [
        _user(FAILURE_TEMPLATE.format(detail='model error')),
        _say('刚才这一下没做成。'),
    ]
    assert validate_run(canonical)['checks']['notices_well_formed']


def test_a_cancelled_task_needs_no_delivery(canonical: dict[str, Any]) -> None:
    canonical['tasks'].append({'task_id': 'task_3', 'status': 'cancelled'})
    assert validate_run(canonical)['passed']


def _write_files(directory: Path, output: dict[str, Any]) -> None:
    for name in TRAJECTORY_FILES:
        record = output[f'{name.removesuffix(".jsonl")}_trajectory']
        (directory / name).write_text(json.dumps(record, ensure_ascii=False) + '\n', encoding='utf-8')


def test_trajectory_files_are_checked_only_when_a_directory_is_given(canonical: dict[str, Any], tmp_path: Path) -> None:
    assert 'trajectory_files_parse' not in validate_run(canonical)['checks']
    _write_files(tmp_path, canonical)
    assert validate_run(canonical, output_dir=tmp_path)['checks']['trajectory_files_parse']


@pytest.mark.parametrize('damage', ['missing', 'not-json', 'empty'])
def test_trajectory_files_must_exist_and_parse(canonical: dict[str, Any], tmp_path: Path, damage: str) -> None:
    _write_files(tmp_path, canonical)
    target = tmp_path / 'midbrain.jsonl'
    if damage == 'missing':
        target.unlink()
    else:
        target.write_text('{"tools": [' if damage == 'not-json' else '', encoding='utf-8')
    result = validate_run(canonical, output_dir=tmp_path)
    assert not result['checks']['trajectory_files_parse']


# ── Case-specific checks ──────────────────────────────────────────────────────


TETRIS = 'task_2'
ANSWER_FACT = notice_fact(
    'answer', {'work': '俄罗斯方块', 'question': '接下来还做哪几步', 'answer': '还要写消行和计分。'}
)
UNANSWERED_FACT = notice_fact('answer', {'work': '俄罗斯方块', 'question': '接下来还做哪几步', 'answer': ''})
TETRIS_DONE_FACT = notice_fact('completed', {'work': '俄罗斯方块', 'summary': '俄罗斯方块做好了，带计分板。'})
CASE_CHECKS = ('question_answered_during_run', 'steer_replaced_running_run')


def _delivery(content: str) -> dict[str, Any]:
    return {'kind': 'task_delivery', 'task_id': TETRIS, 'payload': {'content': NOTICE_TEMPLATE.format(fact=content)}}


@pytest.fixture
def followup() -> dict[str, Any]:
    """A follow-up answered beside the running turn, then an update that steered the running turn."""
    midbrain = [
        {'role': 'system', 'content': 'midbrain'},
        _call('m1', 'TaskCreate', subject='俄罗斯方块', description='写一个俄罗斯方块'),
        _result('m1', 'TaskCreate', {'status': 'accepted', 'task_id': TETRIS, 'subject': '俄罗斯方块'}),
        _call('m2', 'ask_backend', taskId=TETRIS, question='接下来还打算做哪几步？'),
        _result('m2', 'ask_backend', {'status': 'accepted', 'task_id': TETRIS}),
        _call('m3', 'TaskUpdate', taskId=TETRIS, description='加一个计分板，消一行加 100 分'),
        _result('m3', 'TaskUpdate', {'status': 'accepted', 'task_id': TETRIS, 'steered': True}),
    ]
    return {
        'midbrain_trajectory': {'tools': [], 'messages': midbrain},
        'tasks': [{'task_id': TETRIS, 'status': 'done'}],
        'delivery_events': [
            {'kind': 'permission_request', 'task_id': TETRIS, 'payload': {'content': 'permission'}},
            _delivery(ANSWER_FACT),
            _delivery(TETRIS_DONE_FACT),
        ],
    }


def test_case_checks_pass_when_the_answer_precedes_completion_and_the_steer_stopped_a_run(
    followup: dict[str, Any],
) -> None:
    checks = validate_run(followup, case_checks=CASE_CHECKS)['checks']
    assert checks['question_answered_during_run']
    assert checks['steer_replaced_running_run']


def test_case_checks_run_only_when_the_case_asks_for_them(canonical: dict[str, Any]) -> None:
    result = validate_run(canonical)
    assert result['passed'], result['error']
    assert not set(CASE_CHECKS) & set(result['checks']), 'A CodeBuddy case must never be judged by them'


ARTIFACT_CHECK = 'acp_artifact_presented_and_opened'


def _presented_output(canonical: dict[str, Any]) -> dict[str, Any]:
    canonical['tasks'][0]['activity'] = [
        {'kind': 'artifact', 'detail': 'index.html'},
        {'kind': 'display', 'detail': '展示产物：index.html'},
    ]
    canonical['midbrain_trajectory']['messages'].extend([
        _call('open', 'present_files', artifact_id='index'),
        _result(
            'open',
            'present_files',
            {
                'status': 'opened',
                'artifact_id': 'index',
                'title': 'index.html',
                'url': 'https://example.invalid/artifacts/index',
            },
        ),
    ])
    return canonical


@pytest.mark.parametrize('display', ['展示产物：index.html', '展示产物：index.html、README.md'])
def test_acp_artifact_check_passes_when_the_presented_page_is_opened(canonical: dict[str, Any], display: str) -> None:
    output = _presented_output(canonical)
    output['tasks'][0]['activity'][1]['detail'] = display
    result = validate_run(output, case_checks=(ARTIFACT_CHECK,))

    assert result['passed'], result['error']


def test_acp_artifact_check_rejects_directory_scanning_without_presentation(canonical: dict[str, Any]) -> None:
    output = _presented_output(canonical)
    output['tasks'][0]['activity'].pop()
    result = validate_run(output, case_checks=(ARTIFACT_CHECK,))

    assert result['checks'][ARTIFACT_CHECK] is False, (
        'A registered and opened file alone does not prove ACP presentation'
    )


@pytest.mark.parametrize(
    'receipt',
    [
        {'status': 'failed'},
        {'status': 'opened', 'artifact_id': 'index', 'title': 'README.md', 'url': 'https://example.invalid/readme'},
        {'status': 'opened', 'artifact_id': 'other', 'title': 'index.html', 'url': 'https://example.invalid/other'},
    ],
)
def test_acp_artifact_check_rejects_failed_or_different_artifact_opening(
    canonical: dict[str, Any], receipt: dict[str, Any]
) -> None:
    output = _presented_output(canonical)
    output['midbrain_trajectory']['messages'][-1] = _result('open', 'present_files', receipt)
    result = validate_run(output, case_checks=(ARTIFACT_CHECK,))

    assert result['checks'][ARTIFACT_CHECK] is False, 'The presented primary page must be opened successfully'


def test_workbuddy_case_requires_fork_steer_and_native_artifact_presentation(canonical: dict[str, Any]) -> None:
    checks = get_case('workbuddy_followup_and_steer').case_checks

    assert checks == (*CASE_CHECKS, ARTIFACT_CHECK), 'Keep ACP v2 regression coverage in the existing case'
    assert ARTIFACT_CHECK not in validate_run(canonical)['checks'], 'Other cases do not require presentation'


def _answer_after_completion(output: dict[str, Any]) -> None:
    events = output['delivery_events']
    events[1], events[2] = events[2], events[1]


def _no_answer(output: dict[str, Any]) -> None:
    output['delivery_events'][1] = _delivery(UNANSWERED_FACT)


def _never_asked(output: dict[str, Any]) -> None:
    del output['midbrain_trajectory']['messages'][3:5]


def _steer_after_run_ended(output: dict[str, Any]) -> None:
    receipt = json.loads(output['midbrain_trajectory']['messages'][6]['content'])
    receipt['steered'] = False
    output['midbrain_trajectory']['messages'][6]['content'] = json.dumps(receipt)


def _steered_task_failed(output: dict[str, Any]) -> None:
    output['tasks'][0]['status'] = 'failed'


@pytest.mark.parametrize(
    ('mutate', 'failing_check'),
    [
        (_answer_after_completion, 'question_answered_during_run'),
        (_no_answer, 'question_answered_during_run'),
        (_never_asked, 'question_answered_during_run'),
        (_steer_after_run_ended, 'steer_replaced_running_run'),
        (_steered_task_failed, 'steer_replaced_running_run'),
    ],
    ids=[
        'answer-queued-behind-the-turn',
        'question-never-answered',
        'midbrain-never-asked-the-backend',
        'update-after-the-turn-ended-steers-nothing',
        'steered-task-did-not-finish',
    ],
)
def test_case_checks_name_the_broken_flow(
    followup: dict[str, Any], mutate: Callable[[dict[str, Any]], None], failing_check: str
) -> None:
    broken = copy.deepcopy(followup)
    mutate(broken)
    result = validate_run(broken, case_checks=CASE_CHECKS)
    assert not result['checks'][failing_check]
    assert failing_check in result['error']
