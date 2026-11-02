"""产物预检：两条判分路径必须给出同一个结论。

"这次 rollout 有没有东西可判" 不是某个 judge 的问题，所以它属于
``registry.run_judge`` 的 wrapper，两条路径（宿主 ``task.artifact_task`` /
容器 ``harbor_entry``）自动共用。

它检查的**只有「有没有东西可判」**。产物*内容*的策略（技术栈匹配、反 hack 规则）
不在这里 —— 那些是判分决策，归拥有那个 judge 的人。

这里钉两件事：

1. 预检是**可选加入**的：只有真的采集了工作区的调用方（设了 ``file_list``）才会
   被判。多数 judge 不读 ``archive_b64``（它们评部署好的 URL / 截图），所以
   "没有 archive" 不能当作 "没有产物"。
2. 空产物按 agent 结局分流：正常收尾 → 罚分且留在训练；没跑完 → maskout。
"""

from __future__ import annotations

import asyncio
import base64
import io
import tarfile

import pytest

from nexus.extensions.tasks.artifact.harbor_entry import verdict_to_reward
from nexus.extensions.tasks.artifact.judges.base import JudgeInput
from nexus.extensions.tasks.artifact.judges.registry import (
    EVALUATOR_FAULT_KINDS,
    _artifact_precheck,
)


def _archive(files: dict[str, bytes]) -> str:
    """真实 tar.gz —— 走真实解包路径，不 mock。"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def _check(*, files=None, termination='ENV_DONE', track_files=True):
    archive = _archive(files) if files else ''
    instance: dict = {'infer_metadata': {'agent_termination_reason': termination}}
    if track_files:
        instance['file_list'] = sorted(files or {})
    return _artifact_precheck(
        'svg',
        JudgeInput(
            instance_id='i',
            problem_statement='p',
            openai_clients={},
            archive_b64=archive,
            instance=instance,
        ),
    )


# ────────────────────── 1. 可选加入，不臆测 ──────────────────────


def test_callers_that_do_not_track_files_are_not_judged_empty():
    """没设 ``file_list`` → 放行。

    多数 judge 不读 ``archive_b64``（评的是部署好的 URL / 截图），复合 judge 的
    子调用也常手搓 ``JudgeInput``。把"没提供文件清单"当成"交付为空"会让这些调用
    全部拿到一个假的产物惩罚。
    """
    assert _check(files=None, track_files=False) is None


def test_empty_file_list_is_judged_when_explicitly_supplied():
    """显式给了空清单 → 才判为空。区别在于"有人回答了这个问题"。"""
    verdict = _check(files={}, track_files=True)

    assert verdict is not None
    assert verdict.metadata['artifact_precheck'] == 'empty'


# ────────────────────── 2. 空产物按 agent 结局分流 ──────────────────────


@pytest.mark.parametrize(
    ('label', 'termination', 'kind', 'masked'),
    [
        ('agent 正常收尾', 'ENV_DONE', 'no_artifact', False),
        ('超时', 'TIMEOUT', 'agent_failed', True),
        ('步数上限', 'MAX_STEPS', 'agent_failed', True),
        ('截断', 'TRUNCATION', 'agent_failed', True),
        ('未分类', '', 'agent_failed', True),
    ],
)
def test_empty_artifact_routing(label: str, termination: str, kind: str, masked: bool):
    """正常收尾却没交付 = 模型的问题，必须留在训练里。

    反过来，超时/步数上限的 rollout 根本没机会完成 —— 罚它等于教模型一件它无法
    控制的事。这个区分是 penalty 与 maskout 的分界。
    """
    verdict = _check(files={}, termination=termination)

    assert verdict is not None
    assert verdict.failure_kind == kind, label
    assert (verdict.failure_kind in EVALUATOR_FAULT_KINDS) is masked, label

    reward = verdict_to_reward(verdict, judge_name='svg')['reward']
    assert reward == (-1.0 if masked else 0.0), f'{label}: reward={reward}'


# ────────────────────── 4. 两条路径同一个结论 ──────────────────────


def test_host_short_circuit_delegates_to_the_same_precheck():
    """``task.py`` 只是**提前**调用同一个预检，不是第二份实现。

    两份实现必然漂移 —— 这正是 Harbor 路径当初什么都没有的原因。
    """
    import inspect

    from nexus.extensions.tasks.artifact import task as artifact_task

    source = inspect.getsource(artifact_task._archive_short_circuit)
    assert '_artifact_precheck' in source, 'task.py 又自己实现了一遍'
    # 判分策略不该再出现在 task.py 里 —— 归属 judge，不归入口。
    assert 'required_stack' not in source
    assert 'artifact_empty_cause' not in source


def test_container_path_supplies_the_same_agent_facts():
    """容器侧必须能填出同样的归因事实，否则每个空 rollout 都会被 maskout。

    Harbor 通过 ``T3_AGENT_RESULT_PATH`` 提供 agent 结局；
    ``load_agent_outcome`` 把它翻成 ``artifact_empty_cause`` 认的键。
    """
    from nexus.extensions.tasks.artifact.harbor_entry import _AGENT_STATUS_TO_TERMINATION

    # 关键的一条：agent 成功收尾 → ENV_DONE → no_files_written → 罚分而非 maskout
    assert _AGENT_STATUS_TO_TERMINATION['success'] == 'ENV_DONE'
    assert _AGENT_STATUS_TO_TERMINATION['timeout'] == 'TIMEOUT'
    assert _AGENT_STATUS_TO_TERMINATION['incomplete'] == 'MAX_STEPS'


def test_agent_outcome_is_optional(monkeypatch: pytest.MonkeyPatch):
    """拿不到 agent 结局不是错误 —— 判分继续，归因退化成 unknown（偏 maskout，安全方向）。"""
    from nexus.extensions.tasks.artifact.harbor_entry import load_agent_outcome

    monkeypatch.delenv('T3_AGENT_RESULT_PATH', raising=False)
    assert load_agent_outcome() == {}

    monkeypatch.setenv('T3_AGENT_RESULT_PATH', '/nonexistent/result.json')
    assert load_agent_outcome() == {}


def test_agent_outcome_parses_harbors_envelope(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import json

    from nexus.extensions.tasks.artifact.harbor_entry import load_agent_outcome
    from nexus.extensions.tasks.artifact.reward_protocol import artifact_empty_cause

    result = tmp_path / 'result.json'
    result.write_text(json.dumps({'mode': 'agent', 'status': 'success'}), encoding='utf-8')
    monkeypatch.setenv('T3_AGENT_RESULT_PATH', str(result))

    outcome = load_agent_outcome()

    # 端到端：Harbor 的 status → 归因 → penalty 而不是 maskout
    assert artifact_empty_cause(outcome) == 'no_files_written'


def test_precheck_runs_before_capabilities():
    """预检必须排在 capability 准备之前 —— 否则空产物会先白花一次部署。"""
    import inspect

    from nexus.extensions.tasks.artifact.judges import registry

    source = inspect.getsource(registry.run_judge)
    assert source.index('_artifact_precheck') < source.index('_prepare_capabilities')


def test_precheck_runs_before_the_judges_own_validate():
    """也要排在 judge 自己的 validate 之前：没有产物时，判它的参数对不对没有意义。"""
    import inspect

    from nexus.extensions.tasks.artifact.judges import registry

    source = inspect.getsource(registry.run_judge)
    assert source.index('_artifact_precheck') < source.index('spec.validate(inp)')


def test_async_precheck_is_not_needed():
    """预检是纯函数（只解包 + 查字典），不碰网络/容器 —— 所以能放在最前面。"""
    import inspect

    assert not inspect.iscoroutinefunction(_artifact_precheck)


def test_precheck_result_is_a_valid_judge_result():
    """返回值要能直接进 reward 契约，不需要调用方再翻译一次。"""
    verdict = _check(files={}, termination='ENV_DONE')

    assert verdict is not None
    payload = verdict_to_reward(verdict, judge_name='svg')
    assert payload['reward'] == 0.0
    assert payload['failure_kind'] == 'no_artifact'


def test_precheck_is_reached_through_run_judge():
    """走真实 ``run_judge`` 入口，不只测内部函数。"""
    from nexus.extensions.tasks.artifact.judges.registry import run_judge

    result = asyncio.run(
        run_judge(
            'svg',
            JudgeInput(
                instance_id='i',
                problem_statement='p',
                openai_clients={},
                instance={'file_list': [], 'infer_metadata': {'agent_termination_reason': 'ENV_DONE'}},
            ),
        )
    )

    # 没有 client 也该先被产物预检拦下 —— 证明它排在 client 检查之前。
    assert result.failure_kind == 'no_artifact'
    assert result.metadata['artifact_precheck'] == 'empty'


# ────────────────────── web_judger 自己读契约（入口判分无关） ──────────────────────


def test_web_judger_reads_its_contract_from_the_conventional_path(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Harbor 环境下，契约由 judge 自己从约定路径读，入口不参与。

    这是「Harbor 不该知道 flow 格式」的落点：``test.sh`` 只调判分入口，不传任何
    judge 专属的 flag。
    """
    import json

    from nexus.extensions.tasks.artifact.judges.strategies import web_judger

    contract = tmp_path / 'flow.json'
    contract.write_text(json.dumps({'task_type': 'web', 'flows': [{'fid': 'F1'}]}), encoding='utf-8')
    monkeypatch.setattr(web_judger, 'HARBOR_FLOW_PATHS', (str(contract),))

    # instance 里没有 flow —— 完全模拟入口不读参数的情形
    flow, error = web_judger._flow_contract({'instance_id': 'i'})

    assert error == ''
    assert flow is not None and flow['flows'][0]['fid'] == 'F1'


def test_web_judger_says_where_to_put_the_contract_when_it_is_missing(monkeypatch: pytest.MonkeyPatch):
    """契约缺失的报错要指明约定路径 —— 容器侧看不到 host 的数据生成过程。"""
    from nexus.extensions.tasks.artifact.judges.registry import get_judge
    from nexus.extensions.tasks.artifact.judges.strategies import web_judger

    monkeypatch.setattr(web_judger, 'HARBOR_FLOW_PATHS', ('/nonexistent/flow.json',))

    reason = get_judge('web_judger').validate(
        JudgeInput(instance_id='i', problem_statement='p', openai_clients={}, instance={})
    )

    assert reason is not None
    assert 'tests/flow.json' in reason, '要告诉使用者文件该放哪'


def test_unreadable_contract_reports_the_real_cause(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """文件在那儿但 JSON 打错 → 报真因，不要退化成「没有契约」。"""
    from nexus.extensions.tasks.artifact.judges.strategies import web_judger

    broken = tmp_path / 'flow.json'
    broken.write_text('{not json', encoding='utf-8')
    monkeypatch.setattr(web_judger, 'HARBOR_FLOW_PATHS', (str(broken),))

    flow, error = web_judger._flow_contract({})

    assert flow is None
    assert '读取失败' in error


def test_explicit_bad_flow_does_not_silently_fall_back(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """数据里显式写了 ``flow`` 但值非法 → 报错，不回落到文件。

    「写错了」和「没写」是两件事：静默回落会让一个打错的字段表现成「用了另一份
    契约」，判分照常出分，而没人知道跑的不是预期那份。
    """
    import json

    from nexus.extensions.tasks.artifact.judges.strategies import web_judger

    contract = tmp_path / 'flow.json'
    contract.write_text(json.dumps({'flows': [{'fid': 'FROM_FILE'}]}), encoding='utf-8')
    monkeypatch.setattr(web_judger, 'HARBOR_FLOW_PATHS', (str(contract),))

    flow, error = web_judger._flow_contract({'flow': 'oops'})

    assert flow is None
    assert error, '应报错而不是用文件里的那份'
