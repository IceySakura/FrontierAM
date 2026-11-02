import re

from pathlib import Path

import tomllib
import yaml

from nexus.distributed.config import DistributedConfig
from nexus.distributed.environment.rendering import sys_executable_for_workload


ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT_ROOT = ROOT / 'deployment' / 'distributed'
CHART_ROOT = DEPLOYMENT_ROOT / 'helm' / 'nexus-distributed'


def test_universal_image_contains_only_python313_distributed_runtime():
    dockerfile = (DEPLOYMENT_ROOT / 'Dockerfile.universal').read_text()

    assert (
        'FROM mirrors.tencent.com/tlinux/tlinux4.0-minimal'
        '@sha256:9314bb5aa4696df4dfef9331c334b317dc944cfa345366f60e40442d820494ef' in dockerfile
    )
    assert 'NEXUS_PYTHON_VERSION=3.13' in dockerfile
    assert '/opt/nexus-venv/bin' in dockerfile
    assert '--extra distributed' in dockerfile
    assert '--extra runtime' not in dockerfile
    assert '3.14t' not in dockerfile
    assert '/nix/runtime' not in dockerfile


def test_universal_image_installs_otel_dependencies_used_by_control_components():
    dockerfile = (DEPLOYMENT_ROOT / 'Dockerfile.universal').read_text()

    assert '--extra distributed --extra otel' in dockerfile


def test_control_image_is_only_a_universal_compatibility_alias():
    dockerfile = (DEPLOYMENT_ROOT / 'Dockerfile.control').read_text()

    assert 'ARG NEXUS_UNIVERSAL_BASE_IMAGE=nexus-distributed-universal:local' in dockerfile
    assert 'FROM ${NEXUS_UNIVERSAL_BASE_IMAGE}' in dockerfile
    assert 'uv pip install' not in dockerfile


def test_workload_image_extends_universal_with_runtime_dependencies():
    dockerfile = (DEPLOYMENT_ROOT / 'Dockerfile.workload').read_text()

    assert 'ARG NEXUS_UNIVERSAL_BASE_IMAGE=nexus-distributed-universal:local' in dockerfile
    assert 'FROM ${NEXUS_UNIVERSAL_BASE_IMAGE}' in dockerfile
    assert '--extra runtime' in dockerfile
    assert '--extra workload' in dockerfile
    assert '/opt/nexus-venv/bin/python' in dockerfile
    assert '/nix/runtime' not in dockerfile
    assert '3.14t' not in dockerfile
    assert 'NEXUS_AUTO_INSTALL=1' not in dockerfile


def test_workload_extra_has_no_preinstalled_dependencies():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    workload = project['optional-dependencies']['workload']

    assert workload == []


def test_distributed_extra_contains_metrics_runtime_dependency():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    distributed = project['optional-dependencies']['distributed']

    assert any(requirement.startswith('opentelemetry-api') for requirement in distributed)


def test_builder_image_extends_workload_with_python_build_toolchain():
    dockerfile = (DEPLOYMENT_ROOT / 'Dockerfile.builder').read_text()
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    builder_dependencies = project['optional-dependencies']['builder']
    builder_rpms = (DEPLOYMENT_ROOT / 'rpm' / 'builder.txt').read_text()

    assert 'ARG NEXUS_WORKLOAD_BASE_IMAGE=nexus-distributed-workload:local' in dockerfile
    assert 'FROM ${NEXUS_WORKLOAD_BASE_IMAGE}' in dockerfile
    for package in ('gcc', 'gcc-c++', 'make', 'cmake', 'ninja-build', 'patchelf', 'rust', 'cargo'):
        assert f'{package}-' in builder_rpms
    for package in ('build', 'wheel', 'auditwheel', 'setuptools'):
        assert any(requirement.startswith(package) for requirement in builder_dependencies)
    assert '--extra builder' in dockerfile
    assert 'nexus.distributed.dependency_build.worker' in dockerfile
    assert 'privileged' not in dockerfile.lower()


def test_distributed_images_use_one_mirror_and_frozen_python_lock():
    dockerfiles = [
        (DEPLOYMENT_ROOT / name).read_text()
        for name in (
            'Dockerfile.universal',
            'Dockerfile.universal.threading-free',
            'Dockerfile.workload',
            'Dockerfile.builder',
        )
    ]

    for dockerfile in dockerfiles:
        assert 'NEXUS_PYPI_INDEX_URL=https://mirrors.tencent.com/pypi/simple/' in dockerfile
        assert 'PIP_INDEX_URL=${NEXUS_PYPI_INDEX_URL}' in dockerfile
        assert 'UV_INDEX_URL=${NEXUS_PYPI_INDEX_URL}' in dockerfile
        assert 'uv export --quiet --frozen' in dockerfile
        assert '--no-config --no-deps --require-hashes' in dockerfile
        assert "'.[" not in dockerfile


def test_uv_lock_is_included_in_distributed_image_build_context():
    ignored = (ROOT / '.dockerignore').read_text().splitlines()

    assert 'uv.lock' not in ignored


def test_distributed_images_record_release_build_inputs():
    for name in (
        'Dockerfile.universal',
        'Dockerfile.universal.threading-free',
        'Dockerfile.workload',
        'Dockerfile.builder',
    ):
        dockerfile = (DEPLOYMENT_ROOT / name).read_text()
        assert 'ARG NEXUS_BUILD_REVISION=unknown' in dockerfile
        assert 'org.opencontainers.image.revision="${NEXUS_BUILD_REVISION}"' in dockerfile


def test_distributed_images_install_versioned_rpm_manifests():
    manifests = {
        'Dockerfile.universal': DEPLOYMENT_ROOT / 'rpm' / 'universal.txt',
        'Dockerfile.builder': DEPLOYMENT_ROOT / 'rpm' / 'builder.txt',
    }

    for dockerfile_name, manifest_path in manifests.items():
        dockerfile = (DEPLOYMENT_ROOT / dockerfile_name).read_text()
        assert f'COPY deployment/distributed/rpm/{manifest_path.name}' in dockerfile
        assert f'dnf -y install $(cat /tmp/{manifest_path.name})' in dockerfile
        packages = [line for line in manifest_path.read_text().splitlines() if line]
        assert packages
        assert all(
            re.fullmatch(r'[A-Za-z0-9+_.-]+-[0-9]+:[A-Za-z0-9+_.~-]+\.(?:x86_64|noarch)', item) for item in packages
        )


def test_workload_image_has_no_preinstalled_model_client_setup():
    dockerfile = (DEPLOYMENT_ROOT / 'Dockerfile.workload').read_text()
    readme = (DEPLOYMENT_ROOT / 'README.md').read_text()

    assert 'litellm' not in dockerfile.lower()
    assert 'boto3' not in dockerfile.lower()
    assert 'litellm' not in readme.lower()
    assert 'boto3' not in readme.lower()


def test_helm_keeps_role_images_separate_without_injecting_dynamic_image_config():
    values = yaml.safe_load((CHART_ROOT / 'values.yaml').read_text())
    helpers = (CHART_ROOT / 'templates' / '_helpers.tpl').read_text()

    assert values['controlImage']['repository'] == 'nexus-distributed-universal'
    assert values['workloadImage']['repository'] == 'nexus-distributed-workload'
    assert values['builderImage']['repository'] == 'nexus-distributed-builder'
    assert values['controlImage']['tag'] == values['workloadImage']['tag'] == values['builderImage']['tag']
    assert (
        values['controlImage']['digest'] == values['workloadImage']['digest'] == values['builderImage']['digest'] == ''
    )
    for value_name in ('controlImage', 'workloadImage', 'builderImage'):
        assert f'printf "%s@%s" .Values.{value_name}.repository .Values.{value_name}.digest' in helpers
    assert 'nexus-distributed.builderImage' in helpers
    assert 'NEXUS_DIST_ENVIRONMENT_BUILDER_IMAGE' not in helpers


def test_distributed_config_defaults_to_shared_64_shard_topology():
    assert DistributedConfig().shard_count == 64


def test_distributed_config_configures_dependency_builder_and_helm_grants_job_access():
    config = DistributedConfig()
    helpers = (CHART_ROOT / 'templates' / '_helpers.tpl').read_text()
    rbac = (CHART_ROOT / 'templates' / 'environment-controller-rbac.yaml').read_text()

    assert config.dependency_builder_cpu_millicores == 2000
    assert config.dependency_builder_memory_bytes == 4 * 1024**3
    assert config.dependency_builder_workspace_bytes == 8 * 1024**3
    assert config.dependency_builder_index_url == 'https://mirrors.tencent.com/pypi/simple/'
    for env_name in (
        'NEXUS_DIST_DEPENDENCY_BUILD_BATCH_SIZE',
        'NEXUS_DIST_DEPENDENCY_BUILD_MAX_ATTEMPTS',
        'NEXUS_DIST_DEPENDENCY_BUILDER_CPU_MILLICORES',
        'NEXUS_DIST_DEPENDENCY_BUILDER_MEMORY_BYTES',
        'NEXUS_DIST_DEPENDENCY_BUILDER_WORKSPACE_BYTES',
        'NEXUS_DIST_DEPENDENCY_BUILDER_INDEX_URL',
    ):
        assert env_name not in helpers
    assert 'apiGroups: ["batch"]' in rbac
    assert 'resources: ["jobs"]' in rbac
    assert 'verbs: ["get", "create", "delete"]' in rbac


def test_helm_examples_use_the_layered_images():
    for name in ('values-local-predefined.yaml', 'values-local-dynamic.yaml'):
        values = yaml.safe_load((CHART_ROOT / 'examples' / name).read_text())
        assert values['controlImage']['repository'] == 'nexus-distributed-universal'
        assert values['workloadImage']['repository'] == 'nexus-distributed-workload'
        assert values['builderImage']['repository'] == 'nexus-distributed-builder'


def test_dynamic_workload_uses_the_shared_python313_venv():
    assert sys_executable_for_workload() == '/opt/nexus-venv/bin/python'
