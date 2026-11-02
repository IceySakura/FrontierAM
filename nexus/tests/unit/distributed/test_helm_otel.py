from __future__ import annotations

import shutil
import subprocess

from pathlib import Path

import pytest
import yaml


CHART = Path(__file__).parents[3] / 'deployment/distributed/helm/nexus-distributed'


def _render_otel_chart() -> list[dict]:
    helm = shutil.which('helm')
    if helm is None:
        pytest.skip('helm is not installed')
    completed = subprocess.run(
        [
            helm,
            'template',
            'nexus-dist',
            str(CHART),
            '--set',
            'otel.enabled=true',
            '--set',
            'otel.endpoint=http://collector.internal:4318',
            '--set',
            'otel.existingSecret=nexus-otel',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [document for document in yaml.safe_load_all(completed.stdout) if isinstance(document, dict)]


def test_otel_env_is_rendered_for_every_long_running_distributed_service() -> None:
    documents = _render_otel_chart()
    deployments = {
        document['metadata']['labels']['app.kubernetes.io/component']: document
        for document in documents
        if document.get('kind') in {'Deployment', 'StatefulSet'}
        and document.get('metadata', {}).get('labels', {}).get('app.kubernetes.io/component')
    }
    expected_service_names = {
        'application-gateway': 'nexus-distributed-application-gateway',
        'control-gateway': 'nexus-distributed-control-gateway',
        'scheduler': 'nexus-distributed-scheduler',
        'reconciler': 'nexus-distributed-reconciler',
        'instance-manager': 'nexus-distributed-instance-manager',
    }

    for component, service_name in expected_service_names.items():
        container = deployments[component]['spec']['template']['spec']['containers'][0]
        env = {item['name']: item for item in container['env']}
        assert env['NEXUS_OTEL_ENABLED']['value'] == 'true'
        assert env['OTEL_SERVICE_NAME']['value'] == service_name
        assert env['OTEL_EXPORTER_OTLP_ENDPOINT']['value'] == 'http://collector.internal:4318'
        assert env['OTEL_EXPORTER_OTLP_PROTOCOL']['value'] == 'http/protobuf'
        assert env['OTEL_TRACES_EXPORTER']['value'] == 'otlp_proto_http'
        assert env['OTEL_RESOURCE_ATTRIBUTES']['valueFrom']['secretKeyRef']['key'] == 'otel-resource-attributes'


def test_otel_env_is_not_rendered_when_disabled() -> None:
    helm = shutil.which('helm')
    if helm is None:
        pytest.skip('helm is not installed')
    completed = subprocess.run(
        [helm, 'template', 'nexus-dist', str(CHART)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert 'NEXUS_OTEL_ENABLED' not in completed.stdout


def test_db_init_is_a_regular_install_job_and_a_pre_upgrade_hook() -> None:
    helm = shutil.which('helm')
    if helm is None:
        pytest.skip('helm is not installed')

    def render(*extra_args: str) -> dict:
        completed = subprocess.run(
            [helm, 'template', 'nexus-dist', str(CHART), *extra_args],
            check=True,
            capture_output=True,
            text=True,
        )
        return next(
            document
            for document in yaml.safe_load_all(completed.stdout)
            if isinstance(document, dict)
            and document.get('kind') == 'Job'
            and document.get('metadata', {}).get('labels', {}).get('app.kubernetes.io/component') == 'db-init'
        )

    install_job = render()
    upgrade_job = render('--is-upgrade')

    assert 'helm.sh/hook' not in install_job['metadata'].get('annotations', {})
    assert upgrade_job['metadata']['annotations'] == {
        'helm.sh/hook': 'pre-upgrade',
        'helm.sh/hook-delete-policy': 'before-hook-creation',
        'helm.sh/hook-weight': '-5',
    }
