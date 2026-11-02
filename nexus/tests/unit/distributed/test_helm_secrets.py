from __future__ import annotations

import shutil
import subprocess

from pathlib import Path

import pytest
import yaml


CHART = Path(__file__).parents[3] / 'deployment/distributed/helm/nexus-distributed'
SECRET_ROOT = '/var/run/secrets/nexus'
EXAMPLE_SECRET = CHART / 'examples/secrets.example.yaml'
README = CHART / 'README.md'
LOCAL_DYNAMIC_VALUES = CHART / 'examples/values-local-dynamic.yaml'


def _render_chart(*extra_args: str) -> list[dict]:
    helm = shutil.which('helm')
    if helm is None:
        pytest.skip('helm is not installed')
    completed = subprocess.run(
        [
            helm,
            'template',
            'nexus-dist',
            str(CHART),
            *extra_args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [document for document in yaml.safe_load_all(completed.stdout) if isinstance(document, dict)]


def _pod_documents(documents: list[dict]) -> dict[str, dict]:
    return {
        document['metadata']['labels']['app.kubernetes.io/component']: document
        for document in documents
        if document.get('kind') in {'Deployment', 'StatefulSet', 'Job'}
        and document.get('metadata', {}).get('labels', {}).get('app.kubernetes.io/component')
    }


def _service_documents(documents: list[dict]) -> dict[str, dict]:
    return {
        document['metadata']['labels']['app.kubernetes.io/component']: document
        for document in documents
        if document.get('kind') == 'Service'
        and document.get('metadata', {}).get('labels', {}).get('app.kubernetes.io/component')
    }


def _projected_paths(document: dict) -> set[str]:
    pod_spec = document['spec']['template']['spec']
    volume = next(volume for volume in pod_spec['volumes'] if volume['name'] == 'nexus-secrets')
    return {item['path'] for source in volume['projected']['sources'] for item in source['secret'].get('items', [])}


def test_each_distributed_role_receives_only_the_database_url_secret() -> None:
    pods = _pod_documents(_render_chart())

    for component in (
        'control-gateway',
        'application-gateway',
        'scheduler',
        'reconciler',
        'instance-manager',
        'db-init',
    ):
        document = pods[component]
        container = document['spec']['template']['spec']['containers'][0]
        assert {'name': 'nexus-secrets', 'mountPath': SECRET_ROOT, 'readOnly': True} in container['volumeMounts']
        assert _projected_paths(document) == {'database/url'}


def test_rendered_containers_do_not_define_duplicate_environment_variables() -> None:
    for component, document in _pod_documents(_render_chart()).items():
        for container in document['spec']['template']['spec']['containers']:
            names = [item['name'] for item in container.get('env', [])]
            assert len(names) == len(set(names)), component


def test_reconciler_uses_the_standard_kubernetes_service_account_projection() -> None:
    reconciler = _pod_documents(
        _render_chart(
            '--set',
            'dynamicEnvironment.enabledProviders[0]=kubernetes',
        )
    )['reconciler']
    pod_spec = reconciler['spec']['template']['spec']

    assert pod_spec['automountServiceAccountToken'] is True
    assert pod_spec['serviceAccountName'] == 'nexus-dist-environment-controller'
    assert _projected_paths(reconciler) == {'database/url'}


def test_provider_capability_changes_rbac_without_mounting_platform_credentials() -> None:
    pods = _pod_documents(
        _render_chart(
            '-f',
            str(LOCAL_DYNAMIC_VALUES),
        )
    )

    configured_components = ('control-gateway', 'application-gateway', 'scheduler', 'reconciler', 'db-init')
    for component in configured_components:
        document = pods[component]
        env = {
            item['name']: item['value']
            for item in document['spec']['template']['spec']['containers'][0].get('env', [])
            if 'value' in item
        }
        assert 'NEXUS_DIST_ENVIRONMENT_ENABLED_PROVIDERS' not in env
        assert 'NEXUS_DIST_ENVIRONMENT_PROVIDER_CONFIGS' not in env

    reconciler = pods['reconciler']['spec']['template']['spec']
    container = reconciler['containers'][0]
    assert all(mount['name'] != 'provider-credentials' for mount in container.get('volumeMounts', []))
    assert all(volume['name'] != 'provider-credentials' for volume in reconciler.get('volumes', []))


def test_platform_and_user_credentials_are_not_rendered_into_workloads() -> None:
    forbidden = {
        'GONGFENG_TOKEN',
        'MODEL_SERVICE_API_APP_ID',
        'MODEL_SERVICE_API_APP_KEY',
        'MODEL_SERVICE_API_BASE_URL',
        'WEDATA_KEY',
        'WEDATA_KEY_FILE',
        'NEXUS_DIST_INTERNAL_TOKEN_FILE',
        'NEXUS_DIST_TAI_APPLICATION_TOKEN_FILE',
        'NEXUS_DIST_SUBMIT_TOKEN_HASH_KEYS_FILE',
        'NEXUS_DIST_ARTIFACT_UPLOAD_TICKET_KEYS_FILE',
        'NEXUS_DIST_ARTIFACT_WRITER_USERNAME_FILE',
        'NEXUS_DIST_ARTIFACT_WRITER_TOKEN_FILE',
        'NEXUS_DIST_ARTIFACT_MAINTENANCE_USERNAME_FILE',
        'NEXUS_DIST_ARTIFACT_MAINTENANCE_TOKEN_FILE',
        'NEXUS_DIST_ENVIRONMENT_PROVIDER_CREDENTIALS_ROOT',
        'NEXUS_DIST_ENVIRONMENT_WEDATA_KEY_SECRET_NAME',
        'NEXUS_DIST_ENVIRONMENT_WEDATA_KEY_SECRET_KEY',
    }

    for document in _pod_documents(_render_chart()).values():
        for container in document['spec']['template']['spec']['containers']:
            assert forbidden.isdisjoint(item['name'] for item in container.get('env', []))


def test_database_bootstrap_uses_one_canonical_file_and_no_plain_database_url() -> None:
    pods = _pod_documents(_render_chart())

    for component in (
        'control-gateway',
        'application-gateway',
        'scheduler',
        'reconciler',
        'instance-manager',
        'db-init',
    ):
        container = pods[component]['spec']['template']['spec']['containers'][0]
        env = {item['name']: item for item in container['env']}
        assert env['NEXUS_DIST_DATABASE_URL_FILE']['value'] == f'{SECRET_ROOT}/database/url'
        assert 'NEXUS_DIST_DATABASE_URL' not in env

        assert _projected_paths(pods[component]) == {'database/url'}


def test_helm_services_require_a_stored_production_config_by_default() -> None:
    pods = _pod_documents(_render_chart())

    for component in ('control-gateway', 'application-gateway', 'scheduler', 'reconciler', 'instance-manager'):
        env = {
            item['name']: item.get('value')
            for item in pods[component]['spec']['template']['spec']['containers'][0]['env']
        }
        assert env['NEXUS_DIST_CONFIG_STORE_REQUIRED'] == '1'

    local_pods = _pod_documents(_render_chart('-f', str(LOCAL_DYNAMIC_VALUES)))
    for component in ('control-gateway', 'application-gateway', 'scheduler', 'reconciler'):
        env = {
            item['name']: item.get('value')
            for item in local_pods[component]['spec']['template']['spec']['containers'][0]['env']
        }
        assert env['NEXUS_DIST_CONFIG_STORE_REQUIRED'] == '0'


def test_live_compatibility_is_explicit_and_uses_standard_health_probe() -> None:
    disabled = _pod_documents(_render_chart())['application-gateway']
    disabled_container = disabled['spec']['template']['spec']['containers'][0]
    assert '--enable-live-compat' not in disabled_container['args']
    assert disabled_container['readinessProbe']['httpGet']['path'] == '/v1/health'

    enabled = _pod_documents(_render_chart('--set', 'liveCompatibility.enabled=true'))['application-gateway']
    enabled_container = enabled['spec']['template']['spec']['containers'][0]
    assert '--enable-live-compat' in enabled_container['args']
    assert enabled_container['readinessProbe']['httpGet']['path'] == '/v1/health'
    assert _projected_paths(enabled) == {'database/url'}


def test_predefined_manager_reports_canonical_workload_label() -> None:
    manager = _pod_documents(
        _render_chart(
            '--set-string',
            'instanceManager.workload=execution',
            '--set-string',
            'instanceManager.labels.workload=ignored',
        )
    )['instance-manager']
    command = manager['spec']['template']['spec']['containers'][0]['args'][0]

    assert command.count('--label workload=execution') == 1
    assert '--label workload=ignored' not in command


def test_business_configuration_is_not_rendered_as_process_environment() -> None:
    pods = _pod_documents(_render_chart())
    forbidden = {
        'NEXUS_DIST_AUTH_MODE',
        'NEXUS_DIST_CORS_ORIGINS',
        'NEXUS_DIST_APPLICATION_PUBLIC_BASE_URL',
        'NEXUS_DIST_APPLICATION_UPLOAD_BASE_URL',
        'NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL',
        'NEXUS_DIST_ARTIFACT_STORE_PROVIDER',
        'NEXUS_DIST_ARTIFACT_TRANSFER_SPOOL_CAPACITY_BYTES',
        'NEXUS_DIST_ARTIFACT_UPLOAD_TOTAL_CONCURRENCY',
        'NEXUS_DIST_ARTIFACT_DOWNLOAD_TOTAL_CONCURRENCY',
        'NEXUS_DIST_ARTIFACT_MAINTENANCE_ENABLED',
        'NEXUS_DIST_DEPENDENCY_BUILD_BATCH_SIZE',
        'NEXUS_DIST_SCHEDULER_POLL_INTERVAL',
        'NEXUS_DIST_RECONCILER_CORE_TIMEOUT',
        'NEXUS_DIST_DB_POOL_SIZE',
        'NEXUS_DIST_DB_MAX_OVERFLOW',
    }
    for document in pods.values():
        for container in document['spec']['template']['spec']['containers']:
            assert forbidden.isdisjoint(item['name'] for item in container.get('env', []))

    application = pods['application-gateway']['spec']['template']['spec']['containers'][0]
    application_env = {item['name']: item.get('value') for item in application['env']}
    assert application_env['NEXUS_DIST_ARTIFACT_UPLOAD_SPOOL_DIR'] == '/data/nexus-upload-spool'
    upload_spool = next(
        volume
        for volume in pods['application-gateway']['spec']['template']['spec']['volumes']
        if volume['name'] == 'upload-spool'
    )
    assert upload_spool['emptyDir']['sizeLimit'] == '4Gi'


def test_application_gateway_service_topology_is_deployment_configurable() -> None:
    documents = _render_chart(
        '--set',
        'applicationGateway.service.type=LoadBalancer',
        '--set',
        'applicationGateway.service.externalTrafficPolicy=Local',
        '--set-string',
        'applicationGateway.service.annotations.service\\.cloud\\.tencent\\.com/loadbalance-id=lb-existing',
    )
    services = _service_documents(documents)

    service = services['application-gateway']
    assert service['metadata']['annotations'] == {'service.cloud.tencent.com/loadbalance-id': 'lb-existing'}
    assert service['spec']['type'] == 'LoadBalancer'
    assert service['spec']['externalTrafficPolicy'] == 'Local'


def test_predefined_manager_uses_retained_data_pvc_and_separate_tmp_volume() -> None:
    manager = _pod_documents(_render_chart())['instance-manager']
    pod_spec = manager['spec']['template']['spec']
    container = pod_spec['containers'][0]

    assert {'name': 'data', 'mountPath': '/data'} in container['volumeMounts']
    assert {'name': 'tmp', 'mountPath': '/tmp'} in container['volumeMounts']
    assert not any(mount['mountPath'] == '/data/nexus-distributed' for mount in container['volumeMounts'])
    assert not any(mount['mountPath'] == '/data/nexus-environment' for mount in container['volumeMounts'])
    assert next(volume for volume in pod_spec['volumes'] if volume['name'] == 'tmp')['emptyDir'] == {'sizeLimit': '2Gi'}
    assert not any(volume['name'] == 'data' for volume in pod_spec['volumes'])
    assert manager['spec']['persistentVolumeClaimRetentionPolicy'] == {
        'whenDeleted': 'Retain',
        'whenScaled': 'Retain',
    }
    claim = manager['spec']['volumeClaimTemplates'][0]
    assert claim['metadata']['name'] == 'data'
    assert claim['spec']['accessModes'] == ['ReadWriteOnce']
    assert claim['spec']['resources']['requests']['storage'] == '20Gi'
    command = container['args'][0]
    assert '--environment-id predefined-local' in command
    assert '--pool-id interactive' in command
    assert '--workload execution' in command
    assert '--environment-name' not in command
    assert '--environment-key' not in command
    assert '--workload-class' not in command


def test_predefined_manager_can_explicitly_use_bounded_ephemeral_data_for_kind() -> None:
    manager = _pod_documents(_render_chart('--set', 'instanceManager.persistence.enabled=false'))['instance-manager']
    pod_spec = manager['spec']['template']['spec']

    assert 'volumeClaimTemplates' not in manager['spec']
    assert next(volume for volume in pod_spec['volumes'] if volume['name'] == 'data')['emptyDir'] == {
        'sizeLimit': '4Gi'
    }


def test_predefined_batch_manager_uses_orchestration_pool_and_shared_manager_template() -> None:
    batch_manager = _pod_documents(_render_chart('--set', 'batchManager.enabled=true'))['batch-manager']
    pod_spec = batch_manager['spec']['template']['spec']
    container = pod_spec['containers'][0]
    command = container['args'][0]
    env = {item['name']: item.get('value') for item in container['env']}

    assert '--pool-id orchestration' in command
    assert '--workload orchestration' in command
    assert '--label workload=orchestration' in command
    assert 'NEXUS_DIST_DB_POOL_SIZE' not in env
    assert 'NEXUS_DIST_DB_MAX_OVERFLOW' not in env
    assert _projected_paths(batch_manager) == {'database/url'}
    assert {'name': 'data', 'mountPath': '/data'} in container['volumeMounts']
    assert {'name': 'tmp', 'mountPath': '/tmp'} in container['volumeMounts']
    claim = batch_manager['spec']['volumeClaimTemplates'][0]
    assert claim['metadata']['labels']['app.kubernetes.io/component'] == 'batch-manager'


def test_application_secret_example_and_readme_only_cover_database_url() -> None:
    example = yaml.safe_load(EXAMPLE_SECRET.read_text())
    example_keys = set(example['stringData'])
    assert example_keys == {'database-url'}

    readme = README.read_text()
    assert '--from-literal=database-url=' in readme
    for key in (
        'internal-token',
        'tai-application-token',
        'upload-ticket-keys',
        'submit-token-hash-keys',
        'gongfeng-token',
        'model-service-api-app-key',
    ):
        assert f'--from-literal={key}=' not in readme
    assert 'identity-proxy-token' not in readme
    assert 'X-Nexus-Identity-Proxy-Token' not in readme


def test_control_gateway_reuses_environment_controller_rbac():
    documents = _render_chart(
        '--set', 'dynamicEnvironment.enabledProviders={kubernetes}', '--set', 'dynamicEnvironment.namespace=workers'
    )
    gateway = _pod_documents(documents)['control-gateway']['spec']['template']['spec']
    account = gateway['serviceAccountName']
    binding = next(
        doc
        for doc in documents
        if doc.get('kind') == 'RoleBinding' and any(subject['name'] == account for subject in doc['subjects'])
    )
    assert binding['metadata']['namespace'] == 'workers'
    role = next(
        doc for doc in documents if doc.get('kind') == 'Role' and doc['metadata']['name'] == binding['roleRef']['name']
    )
    permissions = {
        (resource, verb) for rule in role['rules'] for resource in rule['resources'] for verb in rule['verbs']
    }
    assert ('pods', 'list') in permissions
    reconciler = _pod_documents(documents)['reconciler']['spec']['template']['spec']
    assert account == reconciler['serviceAccountName']
    assert not any('build-log-reader' in doc.get('metadata', {}).get('name', '') for doc in documents)
