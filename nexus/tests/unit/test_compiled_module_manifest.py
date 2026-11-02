import zipfile

from pathlib import Path

from nexus.core.module_meta import (
    ModuleMeta,
    discover_all_modules,
    discover_modules_from_whl,
    modules_from_manifest_records,
    parse_directory_module,
    parse_pep723_metadata,
    register_objects_from_meta,
)
from nexus.dataset.source import dataset_source_registry


def test_modules_from_manifest_records_builds_module_meta_objects():
    records = [
        {
            'name': 'file_editor',
            'module_path': 'nexus.extensions.tools.file_editor',
            'description': 'File editor',
            'version': '0.1.0',
            'authors': [{'name': 'Alice', 'email': 'alice@example.com'}],
            'maintainers': [{'name': 'Bob', 'email': 'bob@example.com'}],
            'base_deps': [],
            'host_extras': [],
            'runtime_extras': ['binaryornot>=0.4.4'],
            'objects': {'tools': {'str_replace_editor': 'nexus.extensions.tools.file_editor.tool'}},
            'kind': 'directory',
        }
    ]

    modules = modules_from_manifest_records(records)

    assert modules == [
        ModuleMeta(
            name='file_editor',
            module_path='nexus.extensions.tools.file_editor',
            fs_path=Path('<compiled:nexus.extensions.tools.file_editor>'),
            description='File editor',
            version='0.1.0',
            authors=({'name': 'Alice', 'email': 'alice@example.com'},),
            maintainers=({'name': 'Bob', 'email': 'bob@example.com'},),
            base_deps=(),
            host_extras=(),
            runtime_extras=('binaryornot>=0.4.4',),
            objects={'tools': {'str_replace_editor': 'nexus.extensions.tools.file_editor.tool'}},
            kind='directory',
        )
    ]


def test_module_metadata_registers_dataset_sources():
    source_name = 'test-t3-harbor-meta'
    meta = ModuleMeta(
        name='test-source',
        module_path='nexus.extensions.harbor_bridge',
        fs_path=Path('/repo/nexus/extensions/harbor_bridge'),
        objects={
            'dataset_sources': {
                source_name: 'nexus.extensions.harbor_bridge.dataset_source',
            }
        },
    )

    register_objects_from_meta(meta)

    assert source_name in dataset_source_registry


def test_render_manifest_source_is_importable_python():
    from deployment.pypi.generate_nuitka_module_manifest import render_manifest_source

    meta = ModuleMeta(
        name='main',
        module_path='nexus.extensions.agents.main',
        fs_path=Path('/repo/nexus/extensions/agents/main'),
        authors=({'name': 'Alice'},),
        maintainers=({'name': 'Bob', 'email': 'bob@example.com'},),
        objects={'agents': {'main_agent': 'nexus.extensions.agents.main.agent'}},
    )

    source = render_manifest_source([meta])

    assert 'COMPILED_MODULES' in source
    assert "'module_path': 'nexus.extensions.agents.main'" in source
    assert "'authors': [{'name': 'Alice'}]" in source
    assert "'maintainers': [{'email': 'bob@example.com', 'name': 'Bob'}]" in source
    assert "'main_agent': 'nexus.extensions.agents.main.agent'" in source


def test_parse_directory_module_reads_project_people(tmp_path):
    module_dir = tmp_path / 'nexus' / 'extensions' / 'tasks' / 'demo'
    module_dir.mkdir(parents=True)
    (module_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "demo"\n'
        'authors = [{ name = "Alice", email = "alice@example.com" }]\n'
        'maintainers = [{ name = "Bob" }]\n'
        'dependencies = ["requests>=2"]\n'
        '\n'
        '[tool.nexus]\n'
        'tasks = { demo = "nexus.extensions.tasks.demo" }\n'
    )

    meta = parse_directory_module(module_dir, tmp_path)

    assert meta is not None
    assert meta.authors == ({'name': 'Alice', 'email': 'alice@example.com'},)
    assert meta.maintainers == ({'name': 'Bob'},)
    assert meta.base_deps == ('requests>=2',)


def test_parse_directory_module_reads_run_parts(tmp_path):
    module_dir = tmp_path / 'nexus' / 'extensions' / 'agents' / 'demo'
    module_dir.mkdir(parents=True)
    (module_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "demo"\n'
        '\n'
        '[tool.nexus]\n'
        'agents = { demo = "nexus.extensions.agents.demo.agent" }\n'
        'run_parts = { demo = "nexus.extensions.agents.demo.run_parts" }\n'
        'dataset_sources = { demo-source = "nexus.extensions.agents.demo.dataset_source" }\n'
    )

    meta = parse_directory_module(module_dir, tmp_path)

    assert meta is not None
    assert meta.objects['run_parts'] == {
        'demo': 'nexus.extensions.agents.demo.run_parts',
    }
    assert meta.objects['dataset_sources'] == {
        'demo-source': 'nexus.extensions.agents.demo.dataset_source',
    }


def test_discovery_reads_top_level_extension_modules(tmp_path):
    module_dir = tmp_path / 'nexus' / 'extensions' / 'harbor_bridge'
    module_dir.mkdir(parents=True)
    (module_dir / 'pyproject.toml').write_text(
        '[project]\n'
        'name = "harbor_bridge"\n'
        'version = "1.0.0"\n'
        '\n'
        '[tool.nexus]\n'
        'dataset_sources = { t3-harbor = "nexus.extensions.harbor_bridge.dataset_source" }\n'
    )

    modules = discover_all_modules(
        tmp_path,
        include_compiled_manifest=False,
        include_source=True,
    )

    assert [(module.module_path, module.objects) for module in modules] == [
        (
            'nexus.extensions.harbor_bridge',
            {
                'dataset_sources': {
                    't3-harbor': 'nexus.extensions.harbor_bridge.dataset_source',
                }
            },
        )
    ]


def test_wheel_discovery_reads_only_exact_top_level_module_path(tmp_path):
    wheel = tmp_path / 'nexus.whl'
    module_toml = (
        '[project]\n'
        'name = "harbor_bridge"\n'
        'version = "1.0.0"\n'
        '\n'
        '[tool.nexus]\n'
        'dataset_sources = { t3-harbor = "nexus.extensions.harbor_bridge.dataset_source" }\n'
    )
    nested_toml = (
        '[project]\n'
        'name = "internal"\n'
        'version = "1.0.0"\n'
        '\n'
        '[tool.nexus]\n'
        'dataset_sources = { invalid = "nexus.extensions.harbor_bridge.internal" }\n'
    )
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('nexus/extensions/harbor_bridge/pyproject.toml', module_toml)
        archive.writestr('nexus/extensions/harbor_bridge/internal/pyproject.toml', nested_toml)

    modules = discover_modules_from_whl(wheel)

    assert [(module.module_path, module.name) for module in modules] == [
        ('nexus.extensions.harbor_bridge', 'harbor_bridge')
    ]


def test_parse_pep723_metadata_reads_project_people(tmp_path):
    module_file = tmp_path / 'nexus' / 'extensions' / 'tasks' / 'demo.py'
    module_file.parent.mkdir(parents=True)
    module_file.write_text(
        '# /// script\n'
        '# dependencies = ["requests>=2"]\n'
        '#\n'
        '# [project]\n'
        '# name = "demo"\n'
        '# authors = [{ name = "Alice", email = "alice@example.com" }]\n'
        '# maintainers = [{ name = "Bob" }]\n'
        '#\n'
        '# [tool.nexus]\n'
        '# tasks = { demo = "nexus.extensions.tasks.demo" }\n'
        '# run_parts = { demo = "nexus.extensions.tasks.demo" }\n'
        '# ///\n'
        '\n'
        'def demo():\n'
        '    pass\n'
    )

    meta = parse_pep723_metadata(module_file, tmp_path)

    assert meta is not None
    assert meta.name == 'demo'
    assert meta.authors == ({'name': 'Alice', 'email': 'alice@example.com'},)
    assert meta.maintainers == ({'name': 'Bob'},)
    assert meta.base_deps == ('requests>=2',)
    assert meta.objects['run_parts'] == {
        'demo': 'nexus.extensions.tasks.demo',
    }


def test_parse_pep723_metadata_requires_tool_nexus(tmp_path):
    module_file = tmp_path / 'nexus' / 'extensions' / 'tasks' / 'demo.py'
    module_file.parent.mkdir(parents=True)
    module_file.write_text(
        '# /// script\n'
        '# dependencies = ["requests>=2"]\n'
        '#\n'
        '# [project]\n'
        '# name = "demo"\n'
        '# ///\n'
    )

    assert parse_pep723_metadata(module_file, tmp_path) is None


def test_discover_modules_from_whl_reads_pep723_single_file_metadata(tmp_path):
    import zipfile

    wheel = tmp_path / 'nexus_agent-1.0.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr(
            'nexus/extensions/openai_clients/litellm.py',
            '# /// script\n'
            '# dependencies = []\n'
            '#\n'
            '# [project]\n'
            '# name = "litellm"\n'
            '#\n'
            '# [optional-dependencies]\n'
            '# host = ["boto3", "litellm>=1.91.2,<1.92.0"]\n'
            '#\n'
            '# [tool.nexus]\n'
            '# openai_clients = { litellm = "nexus.extensions.openai_clients.litellm" }\n'
            '# ///\n',
        )

    modules = discover_modules_from_whl(wheel)

    assert [(module.name, module.kind, module.host_deps) for module in modules] == [
        ('litellm', 'single_file', ['boto3', 'litellm>=1.91.2,<1.92.0'])
    ]


def test_discover_all_modules_can_use_compiled_manifest_only(monkeypatch):
    import sys
    import types

    module = types.ModuleType('nexus._compiled_module_manifest')
    module.COMPILED_MODULES = [
        {
            'name': 'main',
            'module_path': 'nexus.extensions.agents.main',
            'objects': {'agents': {'main_agent': 'nexus.extensions.agents.main.agent'}},
            'kind': 'directory',
        }
    ]
    monkeypatch.setitem(sys.modules, 'nexus._compiled_module_manifest', module)

    modules = discover_all_modules(
        include_source=False,
        include_compiled_manifest=True,
    )

    assert [(m.name, m.module_path) for m in modules] == [('main', 'nexus.extensions.agents.main')]


def test_discover_all_modules_does_not_scan_current_directory_extensions(tmp_path, monkeypatch):
    pkg = tmp_path / 'third_party_tool'
    pkg.mkdir()
    (pkg / 'pyproject.toml').write_text(
        '[project]\nname = "third-party-tool"\n\n[tool.nexus]\ntools = { third_party_tool = "third_party_tool.tool" }\n'
    )

    monkeypatch.chdir(tmp_path)

    modules = discover_all_modules(
        include_source=False,
        include_compiled_manifest=False,
    )

    assert modules == []
