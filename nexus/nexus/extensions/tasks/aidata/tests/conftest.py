"""aidata 测试共享 fixture。

以下两层共用本文件：

* ``test_cases_resolve.py`` —— 纯静态， 无网络、 无运行时。 只需要
  ``all_case_ids`` 来对每个 case 目录参数化。
* ``test_cases_startup.py`` —— 通过 gongfeng 动态运行。 每个测试用例拥有
  独立的 ``ExecutionContext`` 并拉起一个全新的 ``main`` 沙箱； 这样以
  牺牲 warm-start 的时间换取在 ``pytest-xdist`` 下的正确性 （worker 是
  独立进程， 无法共享 Runtime 对象） 。

要并行运行动态层::

    uv run pytest nexus/extensions/tasks/aidata/tests \
        -n 8 --dist loadgroup

12 个用例 × 每个约 3 分钟， 在 ``-n 8`` 下的实际耗时约为 3-5 分钟。
"""

from __future__ import annotations

import os

from pathlib import Path

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# 仓库布局 + .env 传播
# ---------------------------------------------------------------------------


AIDATA_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = AIDATA_ROOT.parents[3]  # <repo>/nexus/extensions/tasks/aidata → <repo>


def _resolve_cases_root() -> Path:
    """与 ``engine.paths.CASES_DIR`` 使用相同的查找逻辑 —— 环境变量优先，
    否则回退到仓库内的 ``cases/`` 目录。

    尊重 ``AIDATA_CASES_ROOT`` ， 使得上游合并版本的 aidata
    （出于避免引入专有 fixture 的考虑， 有意**不**内含 case 数据） 仍可
    通过将该环境变量指向下游 Tasks 仓库 / 子模块来测试。 而遗留行为 ——
    case 数据与 engine 放在一起 —— 依然开箱即用。
    """
    override = os.getenv('AIDATA_CASES_ROOT')
    if override:
        return Path(override)
    return AIDATA_ROOT / 'cases'


CASES_ROOT = _resolve_cases_root()


def _load_env_files_once() -> None:
    """将 ``.env`` 文件合并进 ``os.environ`` 。

    按顺序尝试两个位置：

    1. 下游 cases 目录下的 ``.env`` （ ``<cases-root>/../.env`` ） —— 与
       ``aidata.py`` CLI 在带有 cases 的 checkout 中运行时使用的约定
       一致。 优先级高于上游 nexus 仓库的 ``.env`` ， 使下游 operator
       能够按目录树覆盖凭据。
    2. 上游 nexus 仓库中的 ``<repo>/.env`` —— 在直接对上游 checkout
       运行测试时有用。

    所有 key 都使用 ``setdefault`` 语义： 现有环境变量优先于文件。 若两个
    文件都不存在则为 no-op。
    """
    candidates: list[Path] = []
    cases_env = CASES_ROOT.parent / '.env'
    if cases_env.is_file():
        candidates.append(cases_env)
    repo_env = REPO_ROOT / '.env'
    if repo_env.is_file() and repo_env not in candidates:
        candidates.append(repo_env)

    for env_file in candidates:
        for raw_line in env_file.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export ') :].lstrip()
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)


_load_env_files_once()


# ---------------------------------------------------------------------------
# 用例发现
# ---------------------------------------------------------------------------


def _discover_case_ids() -> list[str]:
    """返回 ``cases/`` 下所有可发现的 case id。

    只要目录中带有 ``task.yaml`` 或 ``case.yaml`` 就算一个 case。
    会遍历完整的目录树， 因此支持任意深度的分组 —— 例如
    ``work/channel_inventory_ppt`` （ 1 层） 和
    ``work/poc/email_http_poc`` （ 2 层） 都通过 ``case_spec.resolve_case``
    以完全相同的方式流转， 因为 case id 都是相对于 ``cases/`` 的相对路径。
    """
    if not CASES_ROOT.is_dir():
        return []
    ids: list[str] = []
    for manifest in CASES_ROOT.rglob('task.yaml'):
        ids.append(str(manifest.parent.relative_to(CASES_ROOT)))
    for manifest in CASES_ROOT.rglob('case.yaml'):
        case_id = str(manifest.parent.relative_to(CASES_ROOT))
        if case_id not in ids:
            ids.append(case_id)
    return sorted(ids)


ALL_CASE_IDS = _discover_case_ids()


@pytest.fixture(scope='session')
def all_case_ids() -> list[str]:
    """以 fixture 形式暴露， 便于测试函数体内发现。"""
    return list(ALL_CASE_IDS)


# ---------------------------------------------------------------------------
# Gongfeng 运行时 —— 每个测试独立 fixture， 在 pytest-xdist 下并行安全
# ---------------------------------------------------------------------------


def _require_env(name: str) -> str:
    value = os.environ.get(name) or ''
    if not value:
        pytest.fail(
            f'aidata tests require {name} in the environment '
            f'(copy harness .env or export directly). If you really want '
            f'to skip the gongfeng layer use `pytest -k resolve`.'
        )
    return value


@pytest.fixture(scope='session')
def gongfeng_env() -> dict[str, str]:
    """当 gongfeng 凭据缺失时快速失败。

    我们特意**不**将这些测试标记为 ``skipif`` —— aidata 测试套件仅在
    gongfeng 可达时才有意义， 而静默跳过是 CI 里最糟糕的失败模式。
    """
    return {
        'api_base_url': _require_env('GONGFENG_API_BASE_URL'),
        'token': _require_env('GONGFENG_TOKEN'),
    }


@pytest.fixture(scope='session')
def runtime_template() -> dict:
    """一次性解析 ``runtime.template.yaml`` 以获取 ``main`` 镜像 tag。"""
    import yaml

    template_file = AIDATA_ROOT / 'runtime.template.yaml'
    return yaml.safe_load(template_file.read_text(encoding='utf-8'))


@pytest_asyncio.fixture
async def per_case_ctx(gongfeng_env, runtime_template):
    """为每个测试用例产出全新的 ExecutionContext + main 运行时。

    权衡： 冷启动每个用例需付出约 40s 的代价， 但一旦跑在 ``pytest-xdist``
    下， 每测试隔离便是硬需求 （worker 是独立进程， 无法共享 Runtime
    对象） 。 ``__aexit__`` 上的 ``release_all`` 会拆掉 main 以及测试期间
    获取的任何 ``service_*`` 运行时。
    """
    from nexus.core.context import ExecutionContext
    from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider

    provider = GongfengRuntimeProvider(
        api_base_url=gongfeng_env['api_base_url'],
        token=gongfeng_env['token'],
        resource_spec={'cpu': 4, 'memory': 8},
        request_timeout=1800.0,
    )
    main_image = runtime_template['runtime_configs']['main']['image']
    provider.configure('main', image=main_image)

    ctx = ExecutionContext(
        runtime_provider=provider,
        runtime_configs=runtime_template.get('runtime_configs', {}),
        openai_clients={},  # 测试不会实际运行 agent
        task_name='aidata',
        task_config={},
        tool_configs={},
        run_id='pytest-aidata',
    )
    async with ctx:
        yield ctx
    # ctx.__aexit__ ⇒ release_all （main + 每个用例创建的所有 service_* 运行时）
