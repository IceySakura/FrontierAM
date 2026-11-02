"""门控接入的单测：gates.py 常量、gate_judge 判定逻辑、total_reward 门控短路。

``gate_judge`` / ``gates`` / ``grm_rubrics`` / ``grm_transport`` 都已从 hytuner 迁移为
nexus 内独立实现（见 docs/developing/search-s2t-rl-yb-glue-reward-migration.md），
不再需要 stub 掉任何 hytuner/recipe 依赖，直接跑：

    uv run pytest tests/unit/extensions/projects/search_s2t_rl_yb_glue/test_gate_reward.py -v
"""

import asyncio

from typing import Any

import pytest

from nexus.extensions.projects.search_s2t_rl_yb_glue import task as search_task
from nexus.extensions.projects.search_s2t_rl_yb_glue.reward import gate_judge, grm_rubrics, total_reward
from nexus.extensions.projects.search_s2t_rl_yb_glue.reward.gates import GATES, _append_gate_rules, _find_gate_result


GATE_LABELS = [label for label, _dimension, _item in GATES]
GATE_IDS = {dimension: index for index, (_label, dimension, _item) in enumerate(GATES, start=1)}


def test_gates_has_five_entries_including_logic():
    # 迁移时点：hytuner 分支已给 GATES 新增了 logic 门控（早于本项目文档/测试上次更新），
    # 迁移必须原样保留全部五个，而不是只搬文档里写的四个，否则会悄悄收窄判分口径。
    assert GATE_LABELS == ['format', 'safety', 'length', 'natural', 'logic']


def test_append_gate_rules_ids_are_1_based_and_stable():
    checklist, gate_ids = _append_gate_rules([])
    assert len(checklist) == len(GATES)
    assert list(gate_ids.values()) == list(range(1, len(GATES) + 1))


def _rule_results(overrides=None):
    """构造 5 个门控的 rule_results，overrides={dimension: passed}。"""
    base = [{'id': idx, 'name': dimension, 'passed': True} for idx, (_label, dimension, _item) in enumerate(GATES, 1)]
    for rule in base:
        if overrides and rule['name'] in overrides:
            rule['passed'] = overrides[rule['name']]
    return base


def test_judge_gates_all_pass():
    verdict = gate_judge._judge_gates(_rule_results(), GATE_IDS)
    assert verdict['gate_passed'] is True
    assert verdict['failed_gates'] == []
    assert verdict['gate_status'] == {label: 'PASS' for label in GATE_LABELS}


def test_judge_gates_safety_fail():
    verdict = gate_judge._judge_gates(_rule_results({'基础安全-内容安全': False}), GATE_IDS)
    assert verdict['gate_passed'] is False
    assert verdict['failed_gates'] == ['safety']
    assert verdict['gate_status']['safety'] == 'FAIL'


def test_judge_gates_logic_fail():
    verdict = gate_judge._judge_gates(_rule_results({'口语对话-逻辑表达': False}), GATE_IDS)
    assert verdict['gate_passed'] is False
    assert verdict['failed_gates'] == ['logic']
    assert verdict['gate_status']['logic'] == 'FAIL'


def test_judge_gates_absent_treated_as_pass():
    # 只返回 format 一个门控，其余缺席应视为 PASS（不拦）。
    rr = [{'id': 1, 'name': '口语对话-回答格式约束', 'passed': True}]
    verdict = gate_judge._judge_gates(rr, GATE_IDS)
    assert verdict['gate_passed'] is True
    expected = {'format': 'PASS'} | {label: 'ABSENT' for label in GATE_LABELS if label != 'format'}
    assert verdict['gate_status'] == expected


def test_find_gate_result_matches_by_name_then_id():
    _checklist, gate_ids = _append_gate_rules([])
    dimension = GATES[0][1]
    gate_id = gate_ids[dimension]

    by_name = [{'id': 999, 'name': dimension, 'passed': True}]
    assert _find_gate_result(by_name, dimension, gate_id) is by_name[0]

    by_id_only = [{'id': gate_id, 'name': 'judge 没照抄 dimension', 'passed': False}]
    assert _find_gate_result(by_id_only, dimension, gate_id) is by_id_only[0]

    assert _find_gate_result([], dimension, gate_id) is None


def test_build_gate_prompt_renders_all_gate_check_points():
    # 真实走一遍 _build_gate_prompt -> grm_rubrics.build_reward_prompt，确认迁移后的
    # gates.py check_point 文案真的被拼进了发给 GRM 的 prompt（而不是被 stub 掉）。
    prompt, gate_ids = gate_judge._build_gate_prompt(
        solution_str='<result>今天天气不错</result>',
        ground_truth='',
        extra_info={'query': '今天天气怎么样', 'history': []},
    )
    assert '今天天气不错' in prompt
    assert '今天天气怎么样' in prompt
    for _label, dimension, item in GATES:
        assert item['check_point'] in prompt
        assert dimension in gate_ids


def test_grm_rubrics_parse_scorer_output_round_trip():
    # parse_scorer_output 是真实解析逻辑（非 stub），覆盖 ```json 围栏 + <think> 剥离。
    raw = (
        '<think>ignore me</think>\n'
        '```json\n'
        '{"rule_results": [{"id": 1, "name": "口语对话-回答格式约束", "passed": false, "reason": "x"}]}\n'
        '```'
    )
    parsed = grm_rubrics.parse_scorer_output(raw)
    assert parsed == [{'id': 1, 'name': '口语对话-回答格式约束', 'passed': False, 'reason': 'x'}]
    assert grm_rubrics._rule_passed(parsed[0]) is False


def _make_extra_info():
    return {
        'traj_id': 't1',
        'valid_search_count': 1,
        'search_count': 1,
        'subquery_count': 1,
        'tool_call_count': 1,
        'reward_system_router_id': 'AudioGRMDS-SimpleFusion-0',
        'reward_system_user_token': 'dummy',
        'query': '测试问题',
    }


def _run(coro):
    return asyncio.run(coro)


def _make_task_input(**overrides):
    async def unused_run_agent(**kwargs):
        raise AssertionError('run_agent should not be called in oracle mode')

    values: dict[str, Any] = {
        'run_agent': unused_run_agent,
        'run_oracle': True,
        'ground_truth': '参考答案',
        'extra_info': {'question': '测试问题'},
    }
    values.update(overrides)
    return search_task.SearchS2TYBTaskInput(**values)


def test_task_zero_reward_mode_skips_reward_calculation(monkeypatch):
    called = {'reward': False}

    async def fake_compute_score_async(**kwargs):
        called['reward'] = True
        return {'score': 1.0, 'judgment': 'unexpected'}

    monkeypatch.setattr(search_task, '_compute_reward', fake_compute_score_async)

    result = _run(search_task.run_search_s2t_task_yb_glue(_make_task_input(reward_mode='zero')))

    assert called['reward'] is False
    assert result.score == 0.0
    assert result.judgment == ''
    assert result.extra_fields['reward_extra_info']['reward_mode'] == 'zero'
    assert result.extra_fields['reward_extra_info']['reward_skipped'] is True


@pytest.mark.parametrize(
    ('reward_mode', 'is_validation'),
    [('compute', False), ('zero', True)],
)
def test_task_computes_reward_by_default_and_during_validation(monkeypatch, reward_mode, is_validation):
    called = {'reward': False}

    async def fake_compute_score_async(**kwargs):
        called['reward'] = True
        return {'score': 0.5, 'judgment': 'computed'}

    monkeypatch.setattr(search_task, '_compute_reward', fake_compute_score_async)

    result = _run(
        search_task.run_search_s2t_task_yb_glue(_make_task_input(reward_mode=reward_mode, is_validation=is_validation))
    )

    assert called['reward'] is True
    assert result.score == 0.5
    assert result.judgment == 'computed'
    assert result.extra_fields['reward_extra_info']['reward_mode'] == reward_mode
    assert result.extra_fields['reward_extra_info']['reward_skipped'] is False


def test_total_reward_gate_fail_zeros_score(monkeypatch):
    async def fake_rule(**kwargs):
        return {'planner_rule_check': 0}

    async def fake_llm(**kwargs):
        return {'closed_qa_score': 1.0, 'judgment': '完全匹配', 'confidence': '高', 'reason': 'ok'}

    async def fake_gates(**kwargs):
        return {
            'gate_passed': False,
            'failed_gates': ['safety'],
            'gate_status': {label: ('FAIL' if label == 'safety' else 'PASS') for label in GATE_LABELS},
            'gate_rm_request_id': 'req-1',
        }

    monkeypatch.setattr(total_reward.rule_based_judge, 'compute_score', fake_rule)
    monkeypatch.setattr(total_reward.llm_judge, 'compute_score', fake_llm)
    monkeypatch.setattr(total_reward.gate_judge, 'evaluate_gates', fake_gates)

    result = _run(
        total_reward.compute_score_async(
            solution_str='<result>答案</result>',
            ground_truth='参考',
            extra_info=_make_extra_info(),
            data_source='agts_closed_qa',
        )
    )
    assert result['score'] == 0.0
    assert result['gate_passed'] is False
    assert result['failed_gates'] == ['safety']
    # 语义分仍保留用于观测
    assert result['closed_qa_score'] == 1.0


def test_total_reward_gate_pass_keeps_semantic_score(monkeypatch):
    async def fake_rule(**kwargs):
        return {'planner_rule_check': 0}

    async def fake_llm(**kwargs):
        return {'closed_qa_score': 1.0, 'judgment': '完全匹配', 'confidence': '高', 'reason': 'ok'}

    async def fake_gates(**kwargs):
        return {
            'gate_passed': True,
            'failed_gates': [],
            'gate_status': {label: 'PASS' for label in GATE_LABELS},
            'gate_rm_request_id': 'req-2',
        }

    monkeypatch.setattr(total_reward.rule_based_judge, 'compute_score', fake_rule)
    monkeypatch.setattr(total_reward.llm_judge, 'compute_score', fake_llm)
    monkeypatch.setattr(total_reward.gate_judge, 'evaluate_gates', fake_gates)

    # subquery_count == search_count == 1 → query_cost 0，final == 1.0。
    result = _run(
        total_reward.compute_score_async(
            solution_str='<result>答案</result>',
            ground_truth='参考',
            extra_info=_make_extra_info(),
            data_source='agts_closed_qa',
        )
    )
    assert result['score'] == 1.0
    assert result['gate_passed'] is True
    assert result['failed_gates'] == []


def test_total_reward_no_search_skips_gate(monkeypatch):
    # 未搜索早退分支应在门控之前触发，gate_judge 不应被调用。
    called = {'gate': False}

    async def fake_rule(**kwargs):
        return {'planner_rule_check': 0}

    async def fake_gates(**kwargs):
        called['gate'] = True
        return {'gate_passed': True, 'failed_gates': [], 'gate_status': {}, 'gate_rm_request_id': ''}

    monkeypatch.setattr(total_reward.rule_based_judge, 'compute_score', fake_rule)
    monkeypatch.setattr(total_reward.gate_judge, 'evaluate_gates', fake_gates)

    extra = _make_extra_info()
    extra['valid_search_count'] = 0
    result = _run(
        total_reward.compute_score_async(
            solution_str='<result>答案</result>',
            ground_truth='参考',
            extra_info=extra,
            data_source='agts_closed_qa',
        )
    )
    assert result['score'] == 0.0
    assert result['judgment'] == '未搜索'
    assert called['gate'] is False
