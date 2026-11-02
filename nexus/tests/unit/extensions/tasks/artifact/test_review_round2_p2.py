"""Review 第二轮的 P2：Harbor 适配器、名称边界、预检范围、钩子契约。

这些测试刻意走**真实对象和真实入口**：review 指出上一轮的名称测试只检查源码字
符串、skill 测试只用手造的 ``SimpleNamespace``，所以恰好漏掉了真实
``ClaudeCodeInput`` 根本没有 ``extra_skill_dirs`` 字段、以及 ``null`` 别名在
模型校验阶段就被拒这两件事。
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from pathlib import Path

import pytest

from nexus.core.registry import RegistryError
from nexus.extensions.tasks.artifact.judges.base import JudgeInput, JudgeResult
from nexus.extensions.tasks.artifact.judges.registry import get_judge, register_judge, run_judge


def _inp(**kwargs) -> JudgeInput:
    base = {'instance_id': 'i', 'problem_statement': 'p', 'openai_clients': {}}
    base.update(kwargs)
    return JudgeInput(**base)


async def _noop_agent(*_args, **_kwargs):
    raise AssertionError('agent should not run in these tests')


# ══════════════ R4: Harbor 适配器必须接住真实的准备协议 ══════════════


def test_real_judge_input_has_no_extra_skill_dirs_field():
    """钉住这个事实本身 —— 旧的容器侧实现就是对着它不存在的字段取值。

    ``extra_skill_dirs`` 是 ``default_judge_claude_input`` 的**参数**，工厂把它
    消化成 ``pre_start_scripts`` 里一段 base64 打包的安装脚本。所以
    ``getattr(claude_input, 'extra_skill_dirs', None)`` 永远是 ``None``，而
    flow2code 的 prompt 却要求 agent 按名字加载 ``webjudger-collect``。

    现在两条路径都转发 ``model_dump()`` 全量，不再逐字段挑，这类漏项从根上消失。
    覆盖它的执行测试见 ``test_agent_path_unification.py``。
    """
    from nexus.extensions.tasks.artifact.agent_setup.claude_config import default_judge_claude_input

    ci = default_judge_claude_input(with_playwright=True, with_skill=True)

    assert not hasattr(ci, 'extra_skill_dirs'), '若将来真加了这个字段，转发逻辑要同步检查'
    assert ci.pre_start_scripts, 'skill / MCP 配置都靠 pre_start_scripts 落地'


def test_no_runtime_patching_of_the_agent_capability():
    """容器侧不得再用运行时替换 —— 那是漂移的根源。

    旧实现把 ``capabilities.agent.run_agent_and_read_reward`` 整个换掉，等于维护
    第二份实现，实际漏过 5 件事（skill / extra_env / pre_run_setup / MCP /
    trajectory）。现在分叉在 ``capabilities.agent`` 内部，harbor_entry 不碰它。
    """
    import inspect

    from nexus.extensions.tasks.artifact import harbor_entry

    source = inspect.getsource(harbor_entry)
    assert 'run_agent_and_read_reward =' not in source, 'harbor_entry 又在替换共用函数'
    assert not hasattr(harbor_entry, 'route_agent_runs_through_direct_gateway')


# ══════════════ R5: 名称兼容的三个边界 ══════════════


def test_null_judge_name_reaches_the_resolver():
    """``"judge_name": null`` 必须当「没写」，不能在模型校验阶段就 ValidationError。

    generator / parquet 都用 ``inst.get('judge_name', '')`` 透传，而 JSONL 里显式
    写 ``null`` 时 ``.get`` 返回的是 ``None`` 而不是默认值。字段是 ``str`` 时这里
    直接报错，整行数据根本到不了 ``resolve_judge_name`` —— 而那个函数本来就把空值
    当缺失，会给出一条说得清的错误（``no judge_name``）而不是一个 pydantic 类型错。

    分层是有意的：模型只做空值归一化，「没有分派键」这个判断只在 resolver 里
    定义一次。
    """
    from nexus.extensions.tasks.artifact.judges.registry import resolve_judge_name
    from nexus.extensions.tasks.artifact.task import ArtifactTaskInput

    task_input = ArtifactTaskInput(instance_id='i', run_agent=_noop_agent, judge_name=None)
    assert task_input.judge_name == ''

    with pytest.raises(ValueError, match='no judge_name'):
        resolve_judge_name({'judge_name': task_input.judge_name})


def test_dispatch_config_is_checked_before_the_artifact_short_circuit():
    """判分配置校验必须排在产物短路之前。

    排在后面时，一个空产物会先返回产物惩罚，配置错误永远查不到 —— 于是「忘了写
    judge_name」表现成一个看起来正常的产物低分，只在产物非空时才偶尔暴露。
    """
    import inspect

    from nexus.extensions.tasks.artifact import task as artifact_task

    source = inspect.getsource(artifact_task)
    name_check = source.index('bad judge dispatch config')
    short_circuit = source.index('Short-circuit: skip judges on empty')

    assert name_check < short_circuit, '名称校验被排到了产物短路之后'


def test_skip_eval_does_not_require_a_judge_name():
    """``skip_eval`` 只跑 infer 存档，判分名字可以留空。"""
    import inspect

    from nexus.extensions.tasks.artifact import task as artifact_task

    source = inspect.getsource(artifact_task)
    block = source[source.index('Dispatch key: validate BEFORE') : source.index('bad judge dispatch config')]

    assert 'skip_eval' in block, '名称校验必须受 skip_eval 约束'


def test_flow_generator_writes_the_canonical_field():
    """生成脚本要写规范字段 ``judge_name``。"""
    script = (
        Path(__file__).parents[5]
        / 'nexus/extensions/tasks/artifact/judges/strategies/web_judger/run_1_flow.sh'
    ).read_text(encoding='utf-8')

    assert "inst['judge_name'] = 'web_judger'" in script


# ══════════════ R6: 预检范围 ══════════════


@pytest.mark.parametrize(
    ('label', 'instance', 'expect_reject'),
    [
        ('正常 absolute', {'flow': {'flows': [{'fid': 'F1'}]}}, False),
        ('拼错的模式 pairwize', {'flow': {'flows': [{'fid': 'F1'}]}, 'aesthetic_mode': 'pairwize'}, True),
        ('flows 不是数组', {'flow': {'flows': 1}}, True),
        ('flows 元素不是 object', {'flow': {'flows': [1, 2]}}, True),
        ('flow 不是 object', {'flow': 'oops'}, True),
        ('flows 缺 fid', {'flow': {'flows': [{'criterion': 'x'}]}}, True),
        ('完全没有 flow', {}, True),
    ],
)
def test_web_judger_validate_covers_config_errors(label: str, instance: dict, expect_reject: bool):
    """能在部署前确定的配置错误，都要在部署前拒。

    ``pairwize`` 拼错以前一路放行，到 verifier 里 ``== 'pairwise'`` 不匹配就静默
    退回 absolute —— 本该跑对比评审的实验全程跑了绝对评分，分数看起来完全正常。
    ``{'flows': 1}`` 则会在 executor 里炸成 TypeError，被记成 internal_error
    （判分代码有 bug），而真因是数据格式错。
    """
    reason = get_judge('web_judger').validate(_inp(instance=instance))

    assert (reason is not None) is expect_reject, f'{label}: reason={reason!r}'


def test_visual_anchored_checks_its_anchor_before_capabilities():
    """独立使用时缺参考图要在截图/部署之前拒。

    以前只在 ``compute_score`` 里查，那时 ``screenshot`` capability 已经跑完：
    白花一次部署+截图，而且部署失败会先短路，于是「数据里没配参考图」被记成
    ``no_sandbox``（基建故障），排查方向完全错。
    """
    spec = get_judge('visual_anchored')

    assert spec.validate is not None, 'visual_anchored 没接入预检'
    assert spec.validate(_inp(instance={})) is not None
    assert spec.validate(_inp(instance={'anchor_screenshot': 'A' * 600})) is None


# ══════════════ R7: validate 钩子自身的契约 ══════════════


def test_validate_returning_true_becomes_internal_error():
    """返回 ``True`` 不得逃出 ``run_judge``。

    ``True`` 是 truthy，于是走到 ``_fail(reason=True)``，``JudgeResult.error``
    拒掉 bool，抛出的 ValidationError 直接穿透 ``run_judge`` —— 调用方拿到异常
    而不是这个 wrapper 承诺的结构化失败。
    """
    @register_judge('_r7_returns_true', requires=(), validate=lambda _inp: True)
    async def compute_score(inp):
        return JudgeResult(judge_name='_r7_returns_true', score=1.0)

    result = asyncio.run(run_judge('_r7_returns_true', _inp()))

    assert result.failure_kind == 'internal_error'
    assert result.score == 0.0
    assert 'bool' in (result.error or '')


def test_validate_returning_false_is_not_read_as_accept():
    """返回 ``False`` 不能当成「输入合法」。

    ``False`` 是 falsy，于是 judge 在它自己的校验器刚刚拒绝过的数据上跑了起来。
    """
    @register_judge('_r7_returns_false', requires=(), validate=lambda _inp: False)
    async def compute_score(inp):
        return JudgeResult(judge_name='_r7_returns_false', score=1.0)

    result = asyncio.run(run_judge('_r7_returns_false', _inp()))

    assert result.failure_kind == 'internal_error'
    assert result.score != 1.0, 'judge 不该在被拒的数据上跑出满分'


def test_validate_with_extra_required_param_is_rejected_at_registration():
    """带必填额外参数的钩子要在注册期拒，而不是每个实例都 internal_error。

    只数位置参数会漏掉 ``validate(inp, *, required)``：注册干净通过，然后在派发
    时抛 TypeError —— 把一次数据检查变成了每个实例都屏蔽。
    """
    def validate_needs_more(inp, *, required):
        return None

    with pytest.raises(RegistryError) as exc_info:
        @register_judge('_r7_extra_param', requires=(), validate=validate_needs_more)
        async def compute_score(inp):
            return JudgeResult(judge_name='_r7_extra_param', score=1.0)

    assert 'required parameter' in str(exc_info.value)


def test_validate_with_defaulted_extra_param_is_allowed():
    """有默认值的额外参数是可调用的，不该拦 —— 便于测试注入。"""
    def validate_with_default(inp, *, strict=True):
        return None

    @register_judge('_r7_defaulted', requires=(), validate=validate_with_default)
    async def compute_score(inp):
        return JudgeResult(judge_name='_r7_defaulted', score=0.5)

    assert asyncio.run(run_judge('_r7_defaulted', _inp())).score == 0.5


def test_legal_validate_contract_still_works():
    """``None`` 放行、非空字符串拒 —— 正常契约不受影响。"""
    @register_judge('_r7_accept', requires=(), validate=lambda _inp: None)
    async def accept(inp):
        return JudgeResult(judge_name='_r7_accept', score=0.9)

    @register_judge('_r7_reject', requires=(), validate=lambda _inp: 'bad data')
    async def reject(inp):
        return JudgeResult(judge_name='_r7_reject', score=1.0)

    assert asyncio.run(run_judge('_r7_accept', _inp())).score == 0.9

    rejected = asyncio.run(run_judge('_r7_reject', _inp()))
    assert rejected.failure_kind == 'config_error'
    assert rejected.error == 'bad data'
