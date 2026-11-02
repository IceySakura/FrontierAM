"""``harbor_entry`` 只组装通用事实，不碰任何 judge 的参数。

这是 T3-Harbor 目录布局与 judge 输入之间唯一的映射点，错一处整条 rollout 白跑，
所以三件事钉死在测试里：

* ``problem_statement`` 的来源与优先级（``/tests/instruction.md`` 优先）。
* **入口不读判分参数**。八个 judge 里七个只要 prompt + 工作区，只有
  ``web_judger`` 要 flow 契约 —— 它自己去读自己的约定路径。
* 分派键来自 ``[verifier.env]`` 的 ``T3_JUDGE_NAME``，与任何文件无关。

曾经这里有个 ``--instance <path>`` 参数，把 ``web_judger`` 的 schema 泄漏到了通用
入口：每个任务的 ``test.sh`` 都得知道「这个 judge 要那个文件」，换 judge 还得记得
删掉那个 flag。而 Harbor 本来只需要「调用判分入口」这一件事。
"""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from nexus.extensions.tasks.artifact import harbor_entry
from nexus.extensions.tasks.artifact.harbor_entry import load_instance


_JUDGE = 'web_judger'


@pytest.fixture
def tests_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把容器里的 ``/tests`` 重定向到 tmp —— 入口读的是固定路径。"""
    task_dir = tmp_path / 'some_instance_id'
    uploaded = task_dir / 'tests'
    uploaded.mkdir(parents=True)
    monkeypatch.setattr(harbor_entry, 'TESTS_DIR', uploaded)
    return uploaded


# ────────────────────── problem_statement 来源 ──────────────────────


def test_problem_statement_comes_from_tests_instruction_md(tests_dir: Path):
    """``/tests/instruction.md`` 是首选来源。

    Harbor Verifier 只上传 ``tests/``，task 根目录的 instruction.md 在容器里读不到；
    数据侧拷一份进 ``tests/``，判分就能读到 agent 看到的同一段需求，不必把文本另存成
    JSON 字段或 TOML 字符串（那样两处会各自漂移）。
    """
    (tests_dir / 'instruction.md').write_text('  build a 9:16 animation  \n', encoding='utf-8')

    instance = load_instance(workdir='/workspace', judge_name=_JUDGE)

    assert instance['problem_statement'] == 'build a 9:16 animation', '应 strip 首尾空白'


def test_falls_back_to_verifier_env(tests_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """``[verifier.env]`` 的 T3_PROBLEM_STATEMENT 是兜底。"""
    monkeypatch.setenv('T3_PROBLEM_STATEMENT', 'from verifier env')

    instance = load_instance(workdir='/workspace', judge_name=_JUDGE)

    assert instance['problem_statement'] == 'from verifier env'


def test_instruction_file_wins_over_the_env(tests_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('T3_PROBLEM_STATEMENT', 'from env')
    (tests_dir / 'instruction.md').write_text('from the file', encoding='utf-8')

    instance = load_instance(workdir='/workspace', judge_name=_JUDGE)

    assert instance['problem_statement'] == 'from the file'


def test_empty_instruction_file_falls_through(tests_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """空的 instruction.md 不算「有需求」—— 别让空字符串盖掉兜底。"""
    monkeypatch.setenv('T3_PROBLEM_STATEMENT', 'from env')
    (tests_dir / 'instruction.md').write_text('   \n  ', encoding='utf-8')

    instance = load_instance(workdir='/workspace', judge_name=_JUDGE)

    assert instance['problem_statement'] == 'from env'


def test_missing_problem_statement_is_an_error(tests_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """两条来源都没有 → 报错而不是拿空字符串去判分。

    空需求会让 judge 拿不到任何评判依据，却仍然产出一个看起来合理的分数喂进训练。
    """
    monkeypatch.delenv('T3_PROBLEM_STATEMENT', raising=False)

    with pytest.raises(ValueError, match='no problem statement'):
        load_instance(workdir='/workspace', judge_name=_JUDGE)


# ────────────────────── 入口不读判分参数 ──────────────────────


def test_entry_does_not_read_any_judge_parameters(tests_dir: Path):
    """``/tests`` 里有 flow.json 也不该被入口读进 instance。

    这是入口判分无关的核心：它不知道 flow 是什么，也不该知道。
    ``web_judger`` 自己去读 ``HARBOR_FLOW_PATHS``。
    """
    (tests_dir / 'instruction.md').write_text('build a page', encoding='utf-8')
    (tests_dir / 'flow.json').write_text(json.dumps({'flows': [{'fid': 'F1'}]}), encoding='utf-8')

    instance = load_instance(workdir='/workspace', judge_name=_JUDGE)

    assert 'flows' not in instance, '入口把 judge 的参数读进来了'
    assert 'flow' not in instance, '入口不该替任何 judge 包一层'
    assert set(instance) == {'instance_id', 'problem_statement', 'working_dir', 'judge_name'}


def test_entry_has_no_instance_flag():
    """``--instance`` 必须已经消失 —— 它把 web_judger 的 schema 泄漏到了通用入口。"""
    import inspect

    source = inspect.getsource(harbor_entry.main)
    assert '--instance' not in source
    assert '--flow' not in source


def test_instance_id_falls_back_to_the_task_dir_name(tests_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('T3_INSTANCE_ID', raising=False)
    (tests_dir / 'instruction.md').write_text('q', encoding='utf-8')

    instance = load_instance(workdir='/workspace', judge_name=_JUDGE)

    # ``/tests`` 的父目录就是任务目录
    assert instance['instance_id'] == 'some_instance_id'


def test_env_instance_id_wins(tests_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('T3_INSTANCE_ID', 'from-env')
    (tests_dir / 'instruction.md').write_text('q', encoding='utf-8')

    instance = load_instance(workdir='/workspace', judge_name=_JUDGE)

    assert instance['instance_id'] == 'from-env'


def test_judge_name_comes_from_the_caller(tests_dir: Path):
    """``load_instance`` 不自己解析分派键，用调用方给的那个。"""
    (tests_dir / 'instruction.md').write_text('draw an svg', encoding='utf-8')

    instance = load_instance(workdir='/workspace', judge_name='svg')

    assert instance['judge_name'] == 'svg'


# ────────────────────── 分派键：来自 [verifier.env] ──────────────────────
#
# 分派键曾经从 flow.json 里读，默认 ``web_judger``。那样不带 flow 契约的任务
# （svg / code_quality —— 直接看工作区和 prompt）没地方声明自己要谁来判，于是
# 静默跑了 web_judger，再以「没有 flow 契约」失败：报出来的是文件缺失，真因是
# 分派键无处安放。


def test_resolve_judge_name_from_env_reads_verifier_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('T3_JUDGE_NAME', 'svg')

    assert harbor_entry.resolve_judge_name_from_env() == 'svg'


@pytest.mark.parametrize('value', ['', '   '])
def test_blank_judge_name_is_an_error_not_a_default(value: str, monkeypatch: pytest.MonkeyPatch):
    """没配就报错，不猜。

    猜一个（以前默认 ``web_judger``）会让忘配的任务拿另一个 judge 的分数进训练，
    或者以一个与真因无关的理由失败。
    """
    monkeypatch.setenv('T3_JUDGE_NAME', value)

    with pytest.raises(ValueError, match='no judge name'):
        harbor_entry.resolve_judge_name_from_env()


def test_missing_judge_name_is_an_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('T3_JUDGE_NAME', raising=False)

    with pytest.raises(ValueError, match='no judge name'):
        harbor_entry.resolve_judge_name_from_env()
