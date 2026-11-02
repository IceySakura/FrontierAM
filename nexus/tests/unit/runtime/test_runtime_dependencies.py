from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


def test_runtime_extra_declares_pillow_for_pyautogui_screenshots() -> None:
    pyproject_path = Path(__file__).parents[3] / 'pyproject.toml'
    pyproject = tomllib.loads(pyproject_path.read_text())

    runtime_deps = pyproject['project']['optional-dependencies']['runtime']

    assert any(dep.split('>=', 1)[0].lower() == 'pillow' for dep in runtime_deps)


def test_runtime_extra_uses_snapshot_uvloop_build() -> None:
    pyproject_path = Path(__file__).parents[3] / 'pyproject.toml'
    pyproject = tomllib.loads(pyproject_path.read_text())

    runtime_deps = pyproject['project']['optional-dependencies']['runtime']

    assert "uvloop-snapshot==0.22.1.post1; sys_platform != 'win32'" in runtime_deps
    assert not any(dep.startswith('uvloop>=') for dep in runtime_deps)
    distributed_deps = pyproject['project']['optional-dependencies']['distributed']
    assert not any(Requirement(dep).name.lower().startswith('uvloop') for dep in distributed_deps)


def test_runtime_extra_keeps_fastapi_standard_server_dependencies() -> None:
    pyproject_path = Path(__file__).parents[3] / 'pyproject.toml'
    pyproject = tomllib.loads(pyproject_path.read_text())

    project_deps = pyproject['project']['dependencies']
    runtime_deps = pyproject['project']['optional-dependencies']['runtime']
    dependency_names = {Requirement(dep).name.lower() for dep in [*project_deps, *runtime_deps]}

    required_dependency_names = {
        'colorama',
        'email-validator',
        'fastapi',
        'httptools',
        'httpx',
        'jinja2',
        'pydantic-extra-types',
        'pydantic-settings',
        'python-dotenv',
        'python-multipart',
        'pyyaml',
        'uvicorn',
        'uvloop-snapshot',
        'watchfiles',
        'websockets',
    }

    assert required_dependency_names - dependency_names == set()


def test_diagnostic_tools_are_not_standard_dependencies() -> None:
    pyproject_path = Path(__file__).parents[3] / 'pyproject.toml'
    pyproject = tomllib.loads(pyproject_path.read_text())

    dependencies = pyproject['project']['dependencies']

    assert not any(dep.split('>=', 1)[0].lower() == 'aiomonitor' for dep in dependencies)
    assert not any(dep.split('>=', 1)[0].lower() == 'py-spy' for dep in dependencies)


def test_diagnostic_tools_are_scoped_to_runtime_and_distributed_extras() -> None:
    pyproject_path = Path(__file__).parents[3] / 'pyproject.toml'
    pyproject = tomllib.loads(pyproject_path.read_text())

    optional_dependencies = pyproject['project']['optional-dependencies']
    runtime_deps = optional_dependencies['runtime']
    distributed_deps = optional_dependencies['distributed']

    assert any(dep.split('>=', 1)[0].lower() == 'aiomonitor' for dep in runtime_deps)
    assert any(dep.split('>=', 1)[0].lower() == 'aiomonitor' for dep in distributed_deps)
    assert any(dep.split('>=', 1)[0].lower() == 'py-spy' for dep in distributed_deps)


def test_runtime_aiomonitor_debug_port_mapping() -> None:
    from nexus.runtime.server.debug import aiomonitor_webui_port

    assert aiomonitor_webui_port('main', base_port=50200) == 50200
