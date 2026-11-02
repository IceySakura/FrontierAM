"""静态层：批处理生成器必须产出契约干净的流。

此处固定的是 ``nexus run-batch -g nexus/extensions/tasks/aidata/generator.py``
通过 ``nexus.runner.batch.loader.GeneratorLoader`` 实际加载的内容：

* 三个模块级可调用对象 ``generate_configs`` /
  ``get_total_count`` / ``get_batch_metadata`` 存在且行为正确（ count
  与迭代器长度一致， metadata 是可 JSON 序列化的字典）
* 每个 yield 出的 :class:`BatchItem` 具有 aidata 任务的形状 ——
  ``task_name == 'aidata'`` ， ``task_config == {'case_id': <id>}`` ，
  ``run_id`` 与 case_id 逐字节一致（批处理运行器自身的
  ``generate_run_id`` 会为 SQLite 把 ``/`` 替换为 ``_`` ，但我们
  在此保留原始形态，因为 :mod:`engine.case_spec` 会把它当作
  文件系统路径解析）
* 由环境变量驱动的开关（ ``AIDATA_CASE_IDS`` /
  ``AIDATA_SKIP_PENDING_UPSTREAM`` / ``AIDATA_CONFLICT_GROUPS`` ）会按
  ``generator.py`` 中的说明进行过滤 + 打上 metadata 标记

无网络、无运行时、无 LLM 。 与 ``test_cases_resolve.py`` 一起在远远
不到一秒的时间内跑完。 目的是在真实批次运行 **之前** 就捕获契约回归，
避免在一个静默产出 0 / 形状错误配置的生成器上浪费 gongfeng 配额。
"""

from __future__ import annotations

import json
import os

from pathlib import Path

import pytest

from nexus import BatchItem, RunInput
from nexus.extensions.tasks.aidata import generator as gen
from nexus.runner.batch.loader import GeneratorLoader

from .conftest import ALL_CASE_IDS


# 与 ``test_cases_resolve.test_discovery_finds_cases`` 保持一致：上游
# nexus aidata 扩展有意不打包 case 数据，因此当没有配置
# ``AIDATA_CASES_ROOT`` （或配置的目录树为空）时，生成器的 BatchItem
# 流合法地为空，下面的契约测试也就没什么可断言的了。 与其让每条断言
# 单独失败，不如直接在模块层跳过。
if not ALL_CASE_IDS:
    pytest.skip(
        'no cases discovered (set AIDATA_CASES_ROOT to a downstream Tasks '
        'tree to exercise the generator contract tests)',
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# 引导环境
#
# 生成器 import 时已经把 ``<repo>/.env`` 合并到 ``os.environ`` （见
# ``generator._load_repo_env_once`` ）。 但在没有该文件的主机上（ CI
# 镜像、全新 checkout ）， ``RunInput.from_dict`` 依然会失败，因为
# ``runtime.template.yaml`` 里插值了 ``${oc.env:GONGFENG_TOKEN}`` 等等。
# 仅在真实值尚未设置时注入无害占位符，这样静态套件在任何环境下都能跑，
# 而本地开发机仍可继续使用真实的 .env 值。
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope='module')
def _ensure_env_for_omegaconf() -> None:
    for key in (
        'GONGFENG_API_BASE_URL',
        'GONGFENG_TOKEN',
        'SEARCH_OPENAI_API_KEY',
        'SEARCH_OPENAI_BASE_URL',
        'SEARCH_OPENAI_MODEL',
    ):
        os.environ.setdefault(key, f'placeholder-{key.lower()}')


@pytest.fixture(autouse=True)
def _scrub_aidata_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个测试对 AIDATA_* 开关都从干净的状态开始。

    ``delenv`` 加上 ``raising=False`` 在变量未设置时是空操作，因此在
    任何基线 shell 环境下都能安全运行。
    """
    for key in (
        'AIDATA_CASE_IDS',
        'AIDATA_SKIP_PENDING_UPSTREAM',
        'AIDATA_CONFLICT_GROUPS',
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# 模块级协议
# ---------------------------------------------------------------------------


def test_module_exposes_protocol_functions() -> None:
    """``GeneratorLoader`` 会去查找的三个可调用对象存在。"""
    assert callable(gen.generate_configs)
    assert callable(gen.get_total_count)
    assert callable(gen.get_batch_metadata)


def test_generate_configs_yields_batch_items() -> None:
    items = list(gen.generate_configs())
    assert items, (
        'generate_configs yielded nothing — either cases/ is empty, or '
        'the env-var resolution hit silent skips inside the per-case '
        'try/except. Check the WARNING / ERROR log lines emitted by the '
        'generator for the real cause.'
    )
    for it in items:
        assert isinstance(it, BatchItem), f'expected BatchItem, got {type(it).__name__} for run_id={it!r}'


def test_count_matches_generate() -> None:
    """``get_total_count`` 与 ``len(list(generate_configs()))`` 必须一致 ——
    对 loader 撒谎会破坏 inspect-batch 的进度上报，即使各次运行本身
    是成功的。
    """
    items = list(gen.generate_configs())
    assert gen.get_total_count() == len(items)


def test_get_batch_metadata_shape() -> None:
    md = gen.get_batch_metadata()
    assert md['task_type'] == 'aidata'
    assert md['cases_dir'].endswith('cases')
    assert isinstance(md['selected_case_ids'], list)
    assert md['selected_count'] == len(md['selected_case_ids'])
    # 会作为 JSON 持久化到 SQLite ；不可序列化的值会在生成中途
    # 让整个流程崩溃。
    json.dumps(md)


# ---------------------------------------------------------------------------
# BatchItem 形状
# ---------------------------------------------------------------------------


def test_each_item_has_aidata_task_shape() -> None:
    items = list(gen.generate_configs())
    for it in items:
        cfg = it.config
        assert isinstance(cfg, RunInput), f'{it.run_id}: config must be a RunInput, got {type(cfg).__name__}'
        assert cfg.task_name == 'aidata', f'{it.run_id}: task_name must be aidata'
        assert cfg.task_config == {'case_id': it.run_id}, (
            f'{it.run_id}: task_config must be exactly {{case_id: <id>}}, got {cfg.task_config!r}'
        )
        assert cfg.runtime_provider_name == 'gongfeng'
        # OmegaConf 插值必须已在 from_dict 阶段解析完毕；
        # 若这里还残留 ``${...}`` ，说明 runtime.template.yaml 没有
        # 完成 round-trip ，或上面的环境变量注入被绕过了。
        api_url = cfg.runtime_provider_config['api_base_url']
        assert '${' not in api_url, f'{it.run_id}: api_base_url still has unresolved ${{...}}: {api_url!r}'


def test_metadata_is_json_serializable() -> None:
    """``BatchDatabase.insert_run`` 会把 metadata 作为 JSON 持久化；
    不可序列化的值会在阶段 1 生成过程中崩溃。 静态地捕获作者失误
    （ datetime / Path / pydantic 对象）。
    """
    items = list(gen.generate_configs())
    for it in items:
        json.dumps(it.metadata)


def test_metadata_carries_case_id() -> None:
    items = list(gen.generate_configs())
    for it in items:
        assert it.metadata.get('case_id') == it.run_id


def test_run_id_preserves_nesting() -> None:
    """嵌套 case_id （包含 ``/`` ）在 BatchItem.run_id 中保留斜杠。

    ``nexus.runner.batch.runner.generate_run_id`` 才是那个为了 SQLite
    唯一性把 ``/`` 替换为 ``_`` 的地方；生成器不应预先做这种替换，
    否则 ``engine.case_spec.case_manifest_path`` 中的磁盘路径查找就
    会挂掉。
    """
    items = list(gen.generate_configs())
    nested = [it for it in items if '/' in (it.run_id or '')]
    if not nested:
        pytest.skip('no nested case_ids currently shipping')
    for it in nested:
        assert it.metadata['case_id'] == it.run_id, (
            f'{it.run_id}: metadata.case_id drifted to {it.metadata.get("case_id")!r}'
        )


# ---------------------------------------------------------------------------
# 发现
# ---------------------------------------------------------------------------


def test_discovery_matches_conftest() -> None:
    """generator 与 conftest 对什么算作 case 必须达成一致。

    它们通过不同的遍历器（ os.walk-with-skip 与 Path.rglob ）实现同一条
    "目录中包含 task.yaml 或 case.yaml" 的规则；此处发生漂移意味着其中
    之一有 bug 。 通过静态套件把它暴露出来，而不是等到下一次批次运行时
    第一次 mismatch 才发现。
    """
    assert set(gen._discover_cases()) == set(ALL_CASE_IDS)


def test_nested_case_ids_discovered() -> None:
    """例如 ``online_shopping/scenario_add_cart`` 位于第 2 层。"""
    ids = gen._discover_cases()
    nested = [c for c in ids if '/' in c]
    assert nested, f'no nested case_ids found in {gen.CASES_DIR} — did the walker break? sample: {ids[:5]}'


# ---------------------------------------------------------------------------
# 环境变量驱动的过滤
# ---------------------------------------------------------------------------


def test_explicit_case_ids_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AIDATA_CASE_IDS', 'diabetes_followup')
    items = list(gen.generate_configs())
    assert [it.run_id for it in items] == ['diabetes_followup']


def test_explicit_case_ids_filter_supports_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    """过滤器必须接受与 ``aidata.py`` 在 CLI 上使用的相同 ``/`` 编码
    形式 —— 否则 ``AIDATA_CASE_IDS`` 与旧的
    ``aidata.py run online_shopping/scenario_add_cart`` 就会分道扬镳。
    """
    target = 'online_shopping/scenario_add_cart'
    if target not in ALL_CASE_IDS:
        pytest.skip(f'{target} not present in cases/ — update fixture if it moved')
    monkeypatch.setenv('AIDATA_CASE_IDS', target)
    items = list(gen.generate_configs())
    assert [it.run_id for it in items] == [target]


def test_explicit_case_ids_unknown_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """未知的 case_id 会被告警但不会终止生成 —— 静默丢弃太容易被忽略；
    在有人重命名过程中手动编辑一份陈旧列表时，直接抛异常又太脆弱。
    """
    monkeypatch.setenv('AIDATA_CASE_IDS', 'diabetes_followup,does_not_exist')
    with caplog.at_level('WARNING', logger=gen.__name__):
        items = list(gen.generate_configs())
    assert [it.run_id for it in items] == ['diabetes_followup']
    assert any('does_not_exist' in record.message for record in caplog.records), (
        f'expected a WARNING naming the missing case; got {[r.message for r in caplog.records]!r}'
    )


def test_default_skips_pending_upstream() -> None:
    """带有非空 ``pending_upstream:`` 的 case 默认会被排除 ——
    与动态测试层的 ``pytest.skip`` 语义一致。
    """
    items = list(gen.generate_configs())
    for it in items:
        assert not it.metadata.get('pending_upstream'), (
            f'{it.run_id}: pending_upstream={it.metadata.get("pending_upstream")!r} leaked into the default selection'
        )


def test_include_pending_upstream_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AIDATA_SKIP_PENDING_UPSTREAM=0`` 会把它们放回来。

    如果当前打包的 case 都不声明 ``pending_upstream:`` ，则自动跳过，
    这样测试即便在 case 演变的过程中也保持有意义。
    """
    monkeypatch.setenv('AIDATA_SKIP_PENDING_UPSTREAM', '0')
    relaxed = list(gen.generate_configs())
    pending_ids = [it.run_id for it in relaxed if it.metadata.get('pending_upstream')]
    if not pending_ids:
        pytest.skip('no cases currently declare pending_upstream — nothing to assert')

    monkeypatch.setenv('AIDATA_SKIP_PENDING_UPSTREAM', '1')
    default_run_ids = {it.run_id for it in gen.generate_configs()}
    for cid in pending_ids:
        assert cid not in default_run_ids, f'{cid}: still present in default selection despite pending_upstream marker'


def test_conflict_groups_stamped_into_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('AIDATA_CONFLICT_GROUPS', 'gongfeng,llm-quota')
    items = list(gen.generate_configs())
    assert items, 'generator yielded nothing — bug in test setup'
    for it in items:
        assert it.metadata.get('conflict_groups') == ['gongfeng', 'llm-quota'], (
            f'{it.run_id}: conflict_groups not stamped, got metadata={it.metadata!r}'
        )


def test_conflict_groups_absent_by_default() -> None:
    """当环境变量未设置时， metadata 中不应出现该键。

    ``BatchConfig.enable_conflict_groups`` 默认为 False ；此时输出一个
    空列表只会成为噪音，而残留一个陈旧的 ``['gongfeng']`` 又会成为
    footgun ：如果之后有人把 BatchConfig 标志翻上以为它是空操作默认值
    就麻烦了。
    """
    items = list(gen.generate_configs())
    for it in items:
        assert 'conflict_groups' not in it.metadata, f'{it.run_id}: conflict_groups leaked: {it.metadata!r}'


# ---------------------------------------------------------------------------
# GeneratorLoader （真正的 ``nexus run-batch`` 入口路径）
# ---------------------------------------------------------------------------


def test_loader_path_consistent_with_module() -> None:
    """通过 ``run-batch`` 使用的同一个 loader 走一遍 round-trip ，
    以确保动态导入路径与静态导入视图一致。

    可捕获：（ 1 ）公开协议与 ``importlib.util.spec_from_file_location``
    发现结果之间的意外漂移；（ 2 ）generator.py 中的相对导入 ——
    这些导入在 :class:`GeneratorLoader` 中的
    ``sys.modules['config_generator'] = ...`` 重赋值之后可能无法存活。
    """
    loader = GeneratorLoader(Path(gen.__file__))
    direct = list(gen.generate_configs())
    via_loader = list(loader.generate())
    assert [it.run_id for it in direct] == [it.run_id for it in via_loader]
    assert loader.get_total_count() == gen.get_total_count()
    assert loader.get_batch_metadata()['task_type'] == 'aidata'
