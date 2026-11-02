"""``judge_name`` 是唯一的分派键，所有数据行入口用同一条规则。

保留的 ``task.artifact_task`` 从 ``ArtifactTaskInput`` 解析分派键。
当前 Harbor 数据通过 ``[verifier.env].T3_JUDGE_NAME`` 传入容器；旧 generator 已退役。

它们各自写一遍「键叫什么、缺了怎么办」就会漂移 —— 曾经 harbor 做了冲突检查、
task 没做，同一份数据在两条路径上行为不同：一边报配置错误，一边照常跑完出分。
所以规则只在 ``registry.resolve_judge_name`` 定义一次，这里验证每个入口都用它。

``pipeline_name`` 别名已删除：``pipelines/`` 目录删掉后所有数据生成路径都改写
``judge_name``，别名只剩测试在用 —— 留着等于维护一条没有真实数据走的分支，外加
一个「两个键不一致怎么办」的伪问题。

**Harbor 容器侧不在此列**：那条路径的分派键来自 ``[verifier.env]`` 的
``T3_JUDGE_NAME``，不是数据行 —— 见 ``test_harbor_entry_load_instance.py``。
容器里读不到数据集，而且不带 flow 契约的 judge 连参数文件都没有，分派键必须
有个与文件无关的落脚点。
"""

from __future__ import annotations

import pytest

from nexus.extensions.tasks.artifact.judges.registry import resolve_judge_name


async def _noop_agent(*_args, **_kwargs):
    """``ArtifactTaskInput.run_agent`` 的占位 —— 这些测试不跑 agent。"""
    raise AssertionError('agent should not run in a name-resolution test')


# ────────────────────── 规则本身 ──────────────────────


@pytest.mark.parametrize(
    ('label', 'row', 'expected'),
    [
        ('正常', {'judge_name': 'svg'}, 'svg'),
        ('首尾空白', {'judge_name': '  svg  '}, 'svg'),
    ],
)
def test_resolve_judge_name_accepts(label: str, row: dict, expected: str):
    assert resolve_judge_name(row) == expected


@pytest.mark.parametrize(
    ('label', 'row'),
    [
        ('键缺失', {}),
        ('空串', {'judge_name': ''}),
        ('显式 null', {'judge_name': None}),
        ('只有空白', {'judge_name': '   '}),
    ],
)
def test_missing_judge_name_is_rejected(label: str, row: dict):
    """没有分派键 → 报错，不猜。

    猜一个（历史上默认 ``web_judger``）会让忘配的数据拿另一个 judge 的分数进
    训练，或者以一个与真因无关的理由失败。
    """
    with pytest.raises(ValueError, match='no judge_name'):
        resolve_judge_name(row)


def test_legacy_pipeline_name_is_no_longer_honoured():
    """``pipeline_name`` 不再是分派键。

    钉住这条：留一个「有时生效」的别名比彻底删掉更危险 —— 数据里写了它却不生效
    时，报的是「没有 judge_name」而不是「这个键已废弃」，反而更难查。现在它就是
    一个普通的无关字段。
    """
    with pytest.raises(ValueError, match='no judge_name'):
        resolve_judge_name({'pipeline_name': 'web_judger'})


def test_default_applies_only_when_the_key_is_missing():
    assert resolve_judge_name({}, default='web_judger') == 'web_judger'
    # 显式给了名字时 default 不该生效
    assert resolve_judge_name({'judge_name': 'svg'}, default='web_judger') == 'svg'


# ────────────────────── 每个入口都用这条规则 ──────────────────────


def test_task_input_exposes_only_judge_name():
    """``ArtifactTaskInput`` 只有 ``judge_name`` 这一个分派字段。"""
    from nexus.extensions.tasks.artifact.task import ArtifactTaskInput

    fields = set(ArtifactTaskInput.model_fields)
    assert 'judge_name' in fields
    assert 'pipeline_name' not in fields, '历史别名已删除，不要重新引入'

    task_input = ArtifactTaskInput(instance_id='i', judge_name='svg', run_agent=_noop_agent)
    assert resolve_judge_name({'judge_name': task_input.judge_name}) == 'svg'


def test_task_input_accepts_explicit_null():
    """``"judge_name": null`` 当「没写」，不要在模型校验阶段就 ValidationError。

    generator / parquet 用 ``inst.get('judge_name', '')`` 透传，而 JSONL 里显式
    写 ``null`` 时 ``.get`` 返回的是 ``None`` 而不是默认值。字段声明成 ``str``
    会在这里直接报错，整行数据到不了 ``resolve_judge_name`` —— 而那个函数本来
    就会给出一条说得清的错误。
    """
    from nexus.extensions.tasks.artifact.task import ArtifactTaskInput

    task_input = ArtifactTaskInput(instance_id='i', judge_name=None, run_agent=_noop_agent)
    assert task_input.judge_name == ''

    with pytest.raises(ValueError, match='no judge_name'):
        resolve_judge_name({'judge_name': task_input.judge_name})


@pytest.mark.parametrize('name', ['svg', 'web_judger', '', '   ', None])
def test_harbor_data_forwards_explicit_dispatch_without_a_default(tmp_path, monkeypatch, name):
    """Replace the retired generator source check with the current data/env boundary."""
    import json

    import tomllib

    from nexus.extensions.harbor_bridge.compiler import load_task_definition, resolve_env_mapping
    from nexus.extensions.tasks.artifact.harbor_entry import resolve_judge_name_from_env

    text = 'version = "T3-RL-v1.0"\n[verifier.env]\npipeline_name = "ignored"\n'
    if name is not None:
        text += f'T3_JUDGE_NAME = {json.dumps(name)}\n'
    (tmp_path / 'task.toml').write_text(text)
    (tmp_path / 'instruction.md').write_text('Build a page')
    task, instruction = load_task_definition(tmp_path)
    env = resolve_env_mapping(task['verifier']['env'])
    monkeypatch.delenv('T3_JUDGE_NAME', raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert task['version'] == 'T3-RL-v1.0'
    assert task == tomllib.loads(text)
    assert instruction == 'Build a page'
    if name and name.strip():
        assert env['T3_JUDGE_NAME'] == name
        assert resolve_judge_name_from_env() == name
    else:
        with pytest.raises(ValueError, match='no judge name'):
            resolve_judge_name_from_env()


def test_all_entrypoints_agree_on_the_same_row():
    """同一个数据行，所有数据行入口解析出同一个名字。

    核心断言：不是各自「能跑」，而是**结果一致**。
    """
    from nexus.extensions.tasks.artifact.task import ArtifactTaskInput

    row = {'judge_name': 'aesthetic_general'}

    task_input = ArtifactTaskInput(instance_id='i', run_agent=_noop_agent, **row)
    task_name = resolve_judge_name({'judge_name': task_input.judge_name})
    direct_name = resolve_judge_name(row)

    assert task_name == direct_name == 'aesthetic_general'


def test_no_production_code_still_reads_the_alias():
    """整个 artifact 包里不该再有读取 ``pipeline_name`` 的代码。

    删一个字段最容易漏的是某个没被测试覆盖的分支仍在读它 —— 那种残留不会报错，
    只会静默失效。

    用 AST 而不是文本匹配：docstring 和注释里说明「这个别名为什么删了」是有用
    的，把它们也算成违规会逼着人删掉解释。真正要拦的是**取值**，也就是
    ``row['pipeline_name']`` / ``.get('pipeline_name')`` / ``inst.pipeline_name``
    这三种形态。
    """
    import ast

    from pathlib import Path

    from nexus.extensions.tasks.artifact import task as artifact_task

    root = Path(artifact_task.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            # row['pipeline_name']
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == 'pipeline_name'
            ):
                offenders.append(f'{path.relative_to(root)}:{node.lineno}: subscript')
            # x.pipeline_name
            elif isinstance(node, ast.Attribute) and node.attr == 'pipeline_name':
                offenders.append(f'{path.relative_to(root)}:{node.lineno}: attribute')
            # .get('pipeline_name') — 但允许 .pop(...)，那是清理旧数据而非读取
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'get'
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 'pipeline_name'
            ):
                offenders.append(f'{path.relative_to(root)}:{node.lineno}: .get()')

    assert not offenders, '仍有代码读 pipeline_name:\n' + '\n'.join(offenders)
