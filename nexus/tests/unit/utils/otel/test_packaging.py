from pathlib import Path

import tomllib


def test_nuitka_wheel_excludes_optional_otel_support() -> None:
    project_root = Path(__file__).parents[4]
    script = (project_root / 'deployment/pypi/build_nuitka_binary_wheel.sh').read_text()
    assert 'pip_install -q -e ".[dev,runtime]" nuitka' in script
    assert '.[dev,runtime,otel]' not in script
    assert 'project.get("entry-points", {})' not in script
    assert '(dist_info / "entry_points.txt").write_text("[console_scripts]\\nnexus = nexus.cli.main:main\\n")' in script


def test_dev_extra_can_collect_otel_unit_tests() -> None:
    project_root = Path(__file__).parents[4]
    project = tomllib.loads((project_root / 'pyproject.toml').read_text())['project']

    dependencies = project['optional-dependencies']['dev']
    assert 'opentelemetry-distro>=0.51b0' in dependencies
    assert 'opentelemetry-exporter-otlp-proto-http>=1.30.0' in dependencies
    assert not any(dependency.startswith('opentelemetry-distro[otlp]') for dependency in dependencies)


def test_otel_extra_uses_official_aiohttp_server_instrumentation() -> None:
    project_root = Path(__file__).parents[4]
    project = tomllib.loads((project_root / 'pyproject.toml').read_text())['project']

    dependencies = project['optional-dependencies']['otel']
    assert 'opentelemetry-instrumentation-aiohttp-server>=0.51b0' in dependencies
