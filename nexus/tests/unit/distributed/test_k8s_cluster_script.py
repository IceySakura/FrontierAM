import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

from nexus.dataset.zip_index import read_zip_index_from_bytes
from nexus.distributed.batch.materialize import materialize_generator
from nexus.distributed.config import DistributedConfig
from nexus.distributed.submission.specs import (
    BatchSubmission,
    ConfigOnlyCode,
    ImageCode,
    LiveSubmission,
    SingleSubmission,
    SourcePackageCode,
)
from tests.integration.distributed import cluster_config
from tests.integration.distributed import fixtures as distributed_fixtures
from tests.integration.distributed import run_k8s_dynamic_cluster as dynamic_cluster
from tests.integration.distributed import run_k8s_predefined_cluster as k8s_cluster


def test_workload_readiness_counts_only_scheduler_shard_group():
    source = inspect.getsource(k8s_cluster.assert_workload_ready).lower()

    assert "where shard_group = 'execution-scheduler'" in source


def test_workload_readiness_uses_distributed_default_shard_count():
    source = inspect.getsource(k8s_cluster.assert_workload_ready)

    assert DistributedConfig().shard_count == k8s_cluster.EXPECTED_SCHEDULER_SHARD_COUNT
    assert 'EXPECTED_SCHEDULER_SHARD_COUNT' in source


def test_repeated_cluster_cases_cover_three_tasks_three_times_and_one_kill():
    cases = k8s_cluster.build_repeated_cluster_cases('unit')

    assert k8s_cluster.EXPECTED_CLUSTER_SLOT_COUNT == 2
    assert len(cases) == 9
    assert Counter(case.task_key for case in cases) == {
        'live_alpha': 3,
        'live_beta': 3,
        'desktop_single': 3,
    }

    kill_cases = [case for case in cases if case.kill]
    assert [case.task_key for case in kill_cases] == ['live_beta']
    assert kill_cases[0].iteration == 2


def test_predefined_environment_enables_batch_up_to_execution_capacity():
    definition = k8s_cluster.predefined_environment_definition()

    assert definition['allocation_type'] == 'predefined'
    assert definition['provider_type'] == 'predefined'
    assert definition['resource_limits'] == {
        'batch_max_inflight': k8s_cluster.EXPECTED_CLUSTER_SLOT_COUNT,
    }


def test_dynamic_dataset_submission_uses_execution_scoped_wedata_credential():
    payload = dynamic_cluster.dataset_batch_submission(idempotency_key='dataset-credential-test')

    submission = BatchSubmission.model_validate(payload)

    assert submission.data_credentials is not None
    assert submission.data_credentials.kinds() == ('wedata',)
    assert 'data_credentials' not in submission.model_dump(mode='json')


def test_dynamic_remote_dataset_submission_explicitly_pins_task_list_v2():
    payload = dynamic_cluster.dataset_batch_submission(idempotency_key='dataset-task-list-v2')

    assert payload['generation']['source']['task_list_version'] == 2


def test_desktop_single_payload_uses_opus48_model_service_and_gongfeng_runtime():
    payload = k8s_cluster.desktop_single_payload('unit-desktop', iteration=1)

    assert payload['mode'] == 'job'
    assert payload['spec_type'] == 'nexus_single'
    spec = payload['spec']
    assert spec['runtime_provider_name'] == 'gongfeng'
    assert spec['runtime_provider_config']['token'] == '${oc.env:GONGFENG_TOKEN}'
    assert spec['openai_clients']['main']['name'] == 'litellm'
    client_args = spec['openai_clients']['main']['config']['client_args']
    assert 'provider=aws_third' in client_args['api_key']
    assert 'anthropic.claude-opus-4-8' in client_args['api_key']
    assert client_args['base_url'] == '${oc.env:MODEL_SERVICE_API_BASE_URL}'
    assert (
        spec['openai_clients']['main']['config']['request_args']['model']
        == '${oc.env:NEXUS_MODEL,"anthropic.claude-opus-4-8"}'
    )
    assert {'desktop/start', 'desktop/screen_size', 'desktop/bash'} <= set(spec['agent_config']['tools'])


def test_opus48_payload_allows_runtime_image_override():
    runtime_image = 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:aiomonitor-test'

    payload = k8s_cluster.opus48_live_payload(
        'unit-live',
        task_key='live',
        iteration=1,
        runtime_image=runtime_image,
    )
    desktop = k8s_cluster.desktop_single_payload(
        'unit-desktop',
        iteration=1,
        runtime_image=runtime_image,
    )

    assert payload['spec']['runtime_provider_config']['runtime_image'] == runtime_image
    assert desktop['spec']['runtime_provider_config']['runtime_image'] == runtime_image


def test_predefined_run_input_resolves_task_credentials_before_submission(monkeypatch):
    monkeypatch.setenv('GONGFENG_TOKEN', 'outer-gongfeng')
    environment = {
        'GONGFENG_TOKEN': 'submitted-gongfeng',
        'MODEL_SERVICE_API_APP_ID': 'submitted-app-id',
        'MODEL_SERVICE_API_APP_KEY': 'submitted-app-key',
        'MODEL_SERVICE_API_BASE_URL': 'https://model.example',
    }

    payload = k8s_cluster.desktop_single_payload(
        'unit-desktop-resolved',
        iteration=1,
        environment_values=environment,
    )

    spec = payload['spec']
    assert spec['runtime_provider_config']['token'] == 'submitted-gongfeng'
    client_args = spec['openai_clients']['main']['config']['client_args']
    assert client_args['base_url'] == 'https://model.example'
    assert client_args['api_key'].startswith('submitted-app-id:submitted-app-key?')
    assert os.environ['GONGFENG_TOKEN'] == 'outer-gongfeng'


def test_k8s_instance_manager_capacity_matches_expected_slot_count():
    values = yaml.safe_load(k8s_cluster.LOCAL_VALUES.read_text())
    replicas = values['instanceManager']['replicas']
    capacity = values['instanceManager']['capacity']

    assert replicas == 2
    assert capacity == 1
    assert replicas * capacity == k8s_cluster.EXPECTED_CLUSTER_SLOT_COUNT


def test_local_predefined_values_include_dedicated_batch_manager():
    values = yaml.safe_load(k8s_cluster.LOCAL_VALUES.read_text())

    assert values['batchManager'] == {
        'enabled': True,
        'replicas': 1,
        'port': 8091,
        'capacity': 1,
        'executor': 'asyncio',
        'forceReplace': True,
        'autoInstall': True,
        'environmentId': 'predefined-local',
        'poolId': 'orchestration',
        'workload': 'orchestration',
        'labels': {'region': 'local'},
    }


def test_typed_predefined_payloads_are_config_only_and_task_list_is_canonical(tmp_path: Path):
    run_tag = 'unit-predefined'
    fake_runtime_base_url = 'http://nexus-fake-runtime:8080'
    single = SingleSubmission.model_validate(k8s_cluster.typed_predefined_single_submission(run_tag))
    live = LiveSubmission.model_validate(
        k8s_cluster.typed_predefined_live_submission(
            run_tag,
            fake_runtime_base_url=fake_runtime_base_url,
        )
    )
    task_list = tmp_path / 'tasks.jsonl'
    manifest = k8s_cluster.write_predefined_task_list(task_list, run_tag)
    task_list_ref = {
        'artifact_id': 'art-unit-task-list',
        'kind': 'task_list',
        'sha256': manifest['sha256'],
        'size_bytes': manifest['size_bytes'],
    }
    batch = BatchSubmission.model_validate(
        k8s_cluster.typed_predefined_batch_submission(
            task_list_ref=task_list_ref,
            manifest=manifest,
            run_tag=run_tag,
        )
    )

    assert isinstance(single.code, ConfigOnlyCode)
    assert isinstance(live.code, ConfigOnlyCode)
    assert isinstance(batch.code, ConfigOnlyCode)
    assert live.run_input['runtime_provider_config'] == {'base_url': fake_runtime_base_url}
    assert single.pool == live.pool == batch.pool == 'interactive'
    assert batch.batch.max_inflight == k8s_cluster.EXPECTED_CLUSTER_SLOT_COUNT
    records = [json.loads(line) for line in task_list.read_text().splitlines()]
    assert manifest['item_count'] == len(records) == k8s_cluster.PREDEFINED_BATCH_ITEM_COUNT
    assert [item['item_id'] for item in records] == [f'item-{index:04d}' for index in range(len(records))]
    assert hashlib.sha256(task_list.read_bytes()).hexdigest() == manifest['sha256']
    assert task_list.stat().st_size == manifest['size_bytes']


def test_legacy_live_request_scopes_fake_runtime_to_release_service():
    fake_runtime_base_url = 'http://nexus-fake-runtime:8080'

    request = k8s_cluster.fake_live_run_request(
        'unit-live',
        fake_runtime_base_url=fake_runtime_base_url,
    )

    assert request['runtime_provider_config'] == {'base_url': fake_runtime_base_url}


@pytest.mark.asyncio
async def test_distributed_fake_runtime_provider_uses_provider_level_base_url():
    base_url = 'http://nexus-fake-runtime:8080'
    provider = distributed_fixtures.DistributedFakeRuntimeProvider(base_url=base_url)

    runtime = await provider.acquire('main')

    assert runtime.base_url == base_url


def test_local_values_use_separate_universal_workload_and_builder_images():
    values = yaml.safe_load(k8s_cluster.LOCAL_VALUES.read_text())

    assert values['controlImage'] == {
        'repository': 'nexus-distributed-universal',
        'tag': 'local',
        'pullPolicy': 'IfNotPresent',
    }
    assert values['workloadImage'] == {
        'repository': 'nexus-distributed-workload',
        'tag': 'local',
        'pullPolicy': 'IfNotPresent',
    }
    assert values['builderImage'] == {
        'repository': 'nexus-distributed-builder',
        'tag': 'local',
        'pullPolicy': 'IfNotPresent',
    }


def test_dynamic_environment_image_pull_secrets_default_empty():
    values = yaml.safe_load((k8s_cluster.CHART_DIR / 'values.yaml').read_text())

    assert DistributedConfig().environment_image_pull_secrets == []
    assert 'imagePullSecrets' not in values['dynamicEnvironment']


def test_local_values_enable_aiomonitor_by_default():
    values = yaml.safe_load(k8s_cluster.LOCAL_VALUES.read_text())

    config = DistributedConfig()
    assert config.aiomonitor_enabled is True
    assert config.aiomonitor_host == '127.0.0.1'
    assert config.aiomonitor_base_port == 50100
    assert config.aiomonitor_webui_base_port == 50200
    assert config.aiomonitor_console_base_port == 50300
    assert 'distributed' not in values


def test_default_workload_base_image_is_the_universal_image():
    dockerfile = (k8s_cluster.ROOT / 'deployment/distributed/Dockerfile.workload').read_text()

    assert 'NEXUS_UNIVERSAL_BASE_IMAGE=nexus-distributed-universal:local' in dockerfile
    assert 'FROM ${NEXUS_UNIVERSAL_BASE_IMAGE}' in dockerfile


def test_predefined_devcloud_uses_all_three_layered_images():
    args = k8s_cluster.parse_args([])

    assert args.kubeconfig == k8s_cluster.DEFAULT_DEVCLOUD_KUBECONFIG
    assert args.namespace == k8s_cluster.DEFAULT_DEVCLOUD_NAMESPACE
    assert args.release_name == 'nexus-phase2-devcloud-predefined'
    assert args.control_image == 'mirrors.tencent.com/nexus_utopia/nexus-distributed-universal:phase2-devcloud'
    assert args.workload_image == 'mirrors.tencent.com/nexus_utopia/nexus-distributed-workload:phase2-devcloud'
    assert args.builder_image == 'mirrors.tencent.com/nexus_utopia/nexus-distributed-builder:phase2-devcloud'
    assert not hasattr(args, 'kind_bin')


def test_control_image_is_a_universal_compatibility_alias():
    dockerfile = (k8s_cluster.ROOT / 'deployment/distributed/Dockerfile.control').read_text()

    assert dockerfile.splitlines() == [
        'ARG NEXUS_UNIVERSAL_BASE_IMAGE=nexus-distributed-universal:local',
        'FROM ${NEXUS_UNIVERSAL_BASE_IMAGE}',
    ]


def test_universal_image_prepares_only_python313():
    dockerfile = (k8s_cluster.ROOT / 'deployment/distributed/Dockerfile.universal').read_text()

    assert 'ARG NEXUS_PYTHON_VERSION=3.13' in dockerfile
    assert '/opt/nexus-venv' in dockerfile
    assert '3.14t' not in dockerfile
    assert '/nix/runtime' not in dockerfile


def test_universal_image_makes_runtime_sources_readable_by_dynamic_uid():
    dockerfile = (k8s_cluster.ROOT / 'deployment/distributed/Dockerfile.universal').read_text()

    assert 'chmod -R a+rX /app /opt/nexus-venv /opt/uv-bootstrap /opt/uv-python' in dockerfile


def test_workload_image_does_not_preinstall_litellm_model_cost_map():
    dockerfile = (k8s_cluster.ROOT / 'deployment/distributed/Dockerfile.workload').read_text()

    assert 'NEXUS_LITELLM_MODEL_MAP_COMMIT=' not in dockerfile
    assert 'NEXUS_LITELLM_MODEL_MAP_SHA256=' not in dockerfile
    assert 'model_prices_and_context_window_backup.json' not in dockerfile


def test_db_init_job_delegates_database_creation_to_distributed_cli():
    template = (k8s_cluster.ROOT / 'deployment/distributed/helm/nexus-distributed/templates/db-init.yaml').read_text()

    assert 'distributed' in template
    assert 'db' in template
    assert 'init' in template
    assert 'create_async_engine' not in template
    assert 'distributed db upgrade' not in template


def test_scheduler_business_limits_live_in_distributed_config_not_helm():
    chart_root = k8s_cluster.ROOT / 'deployment/distributed/helm/nexus-distributed'
    template = (chart_root / 'templates/schedulers.yaml').read_text()
    helpers = (chart_root / 'templates/_helpers.tpl').read_text()
    values = yaml.safe_load((chart_root / 'values.yaml').read_text())
    config = DistributedConfig()

    assert config.scheduler_scan_limit == 250
    assert config.scheduler_max_assignments == 250
    assert config.scheduler_manager_lock_limit == 64
    assert config.scheduler_manager_lock_contention_backoff.total_seconds() == 0.25
    assert config.scheduler_poll_interval == 0.1
    assert config.batch_reconcile_poll_interval == 0.1
    assert config.batch_submit_batch_size == 2000
    assert config.batch_lifecycle_poll_interval == 0.5
    assert config.batch_lifecycle_page_size == 500
    assert set(values['scheduler']) == {'replicas', 'resources'}
    assert set(values['reconciler']) == {'replicas', 'extraVolumeMounts', 'extraVolumes', 'resources'}
    for source in (template, helpers):
        assert 'NEXUS_DIST_SCHEDULER_SCAN_LIMIT' not in source
        assert 'NEXUS_DIST_SCHEDULER_MAX_ASSIGNMENTS' not in source
        assert 'NEXUS_DIST_BATCH_LIFECYCLE_POLL_INTERVAL' not in source
        assert 'NEXUS_DIST_BATCH_SUBMIT_BATCH_SIZE' not in source


def test_cluster_bootstrap_separates_database_and_mysql_secrets() -> None:
    database = cluster_config.database_secret_values(
        release_name='phase2-e2e',
        namespace='namespace',
        password='nexus_dist_local',
    )

    assert set(database) == {'database-url'}
    assert set(cluster_config.mysql_secret_values(password='nexus_dist_local')) == {'mysql-root-password'}
    assert (
        database['database-url']
        == 'mysql+asyncmy://root:nexus_dist_local@phase2-e2e-mysql.namespace.svc.cluster.local:3306/'
        'nexus_dist?charset=utf8mb4'
    )


def test_cluster_platform_config_and_secrets_validate_as_typed_startup_snapshot() -> None:
    ordinary = cluster_config.platform_config_values(
        application_public_base_url='http://127.0.0.1:8082',
        application_internal_base_url='http://nexus-application-gateway:8082',
        workload_image='registry/workload:test',
        builder_image='registry/builder:test',
        namespace='workloads',
        release_name='nexus',
        database_secret_name='nexus-database',
        enabled_providers=['kubernetes'],
        fake_runtime_base_url='http://nexus-fake-runtime:8080',
    )
    secrets = cluster_config.platform_secret_values({})
    config = DistributedConfig.model_validate({**ordinary, 'secrets': secrets})

    assert config.application_public_base_url == 'http://127.0.0.1:8082'
    assert config.environment_enabled_providers == ['kubernetes']
    assert config.environment_database_url_secret_name == 'nexus-database'
    assert config.read_internal_token() == 'local-dynamic-internal-token'
    assert config.read_tai_application_token() == '0' * 32


def test_cluster_platform_config_can_enable_deterministic_artifact_routing() -> None:
    ordinary = cluster_config.platform_config_values(
        application_public_base_url='http://127.0.0.1:8082',
        application_internal_base_url='http://nexus-application-gateway:8082',
        workload_image='registry/workload:test',
        builder_image='registry/builder:test',
        namespace='workloads',
        release_name='nexus',
        database_secret_name='nexus-database',
        artifact_routing_enabled=True,
        artifact_routing_seed='artifact-store-test',
        artifact_routing_version=3,
        artifact_service_base_url='http://test.artifact.example',
        artifact_service_namespace_prefix='/_nexus_dist_dev',
        artifact_routes=[
            {'target': 'local', 'weight_bps': 0},
            {'target': 'artifact_service', 'weight_bps': 10000},
        ],
    )

    config = DistributedConfig.model_validate({**ordinary, 'secrets': cluster_config.platform_secret_values({})})

    assert config.artifact_store_routing_enabled is True
    assert config.artifact_store_routing_seed == 'artifact-store-test'
    assert config.artifact_store_routing_version == 3
    assert config.artifact_service_base_url == 'http://test.artifact.example'
    assert config.artifact_service_namespace_prefix == '/_nexus_dist_dev'
    assert [route.model_dump() for route in config.artifact_store_routes] == [
        {'target': 'local', 'weight_bps': 0},
        {'target': 'artifact_service', 'weight_bps': 10000},
    ]


def test_dynamic_local_dataset_reserves_tdsql_for_batch_task_store_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePortForward:
        local_port = 3306

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def capture_seed(_run, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(dynamic_cluster, 'PortForward', FakePortForward)
    monkeypatch.setattr(dynamic_cluster, 'wait_tcp_ok', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dynamic_cluster, 'seed_platform_config', capture_seed)
    trace = cluster_config.TDSQLTraceConfig(
        host='tdsql.example',
        port=3306,
        user='nexus-e2e',
        password='task-store-password',
        db='nexus_runner',
    )

    dynamic_cluster.seed_dynamic_platform_config(
        'kubectl',
        namespace='workloads',
        release_name='nexus-dataset-e2e',
        mysql_password='mysql-password',
        environment={},
        application_public_base_url='http://127.0.0.1:8082',
        workload_image='registry/workload:test',
        builder_image='registry/builder:test',
        database_secret_name='nexus-database',
        artifact_provider='local',
        artifact_base_url='',
        artifact_repository='',
        artifact_credentials=None,
        include_dataset=True,
        tdsql_trace=trace,
    )

    config_values = captured['config_values']
    secret_values = captured['secret_values']
    assert isinstance(config_values, dict)
    assert isinstance(secret_values, dict)
    config = DistributedConfig.model_validate({**config_values, 'secrets': secret_values})
    assert config.default_run_trace_sinks == []
    assert set(config.run_trace_sinks) == {'tdsql'}
    assert config.batch_task_store_database == 'nexus_runner'
    assert config.read_run_trace_credential('tdsql-password') == 'task-store-password'


def test_dynamic_cluster_scopes_dcdb_maintenance_credentials_to_control_gateway(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePortForward:
        local_port = 3306

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(dynamic_cluster, 'PortForward', FakePortForward)
    monkeypatch.setattr(dynamic_cluster, 'wait_tcp_ok', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dynamic_cluster, 'seed_platform_config', lambda _run, **kwargs: captured.update(kwargs))
    trace = cluster_config.TDSQLTraceConfig(
        host='tdsql-write.example',
        port=3306,
        user='nexus-e2e',
        password='task-store-password',
        db='nexus_runner',
        read_host='tdsql-read.example',
        read_port=3306,
        account_provider_type='tencent_cloud_dcdb',
        dcdb_region='ap-nanjing',
        dcdb_instance_id='tdsqlshard-test',
        dcdb_api_endpoint='dcdb.internal.tencentcloudapi.com',
        dcdb_secret_id='test-secret-id',
        dcdb_secret_key='test-secret-key',
    )

    dynamic_cluster.seed_dynamic_platform_config(
        'kubectl',
        namespace='workloads',
        release_name='nexus-dataset-e2e',
        mysql_password='mysql-password',
        environment={},
        application_public_base_url='http://127.0.0.1:8082',
        workload_image='registry/workload:test',
        builder_image='registry/builder:test',
        database_secret_name='nexus-database',
        artifact_provider='local',
        artifact_base_url='',
        artifact_repository='',
        artifact_credentials=None,
        include_dataset=True,
        tdsql_trace=trace,
    )

    assert captured['role_secret_values'] == {
        ('control-gateway', '*'): {
            'trace_database_credentials': {
                'tdsql-maintenance': {
                    'provider_type': 'tencent_cloud_dcdb',
                    'secret_id': 'test-secret-id',
                    'secret_key': 'test-secret-key',
                }
            }
        }
    }


def test_cluster_platform_config_adds_tdsql_trace_without_exposing_password() -> None:
    trace = cluster_config.TDSQLTraceConfig(
        host='tdsql.example',
        port=3307,
        user='trajectory',
        password='trajectory-password',
        db='trajectory_test',
        read_host='tdsql-read.example',
        read_port=3308,
        maintenance_user='trajectory-admin',
        maintenance_password='maintenance-password',
    )
    ordinary = cluster_config.platform_config_values(
        application_public_base_url='http://127.0.0.1:8082',
        application_internal_base_url='http://application:8082',
        workload_image='registry/workload:test',
        builder_image='registry/builder:test',
        namespace='workloads',
        release_name='nexus',
        database_secret_name='nexus-database',
        tdsql_trace=trace,
    )
    secrets = cluster_config.platform_secret_values({}, tdsql_trace=trace)
    control_secrets = cluster_config.platform_control_gateway_secret_values(tdsql_trace=trace)
    config = DistributedConfig.model_validate({**ordinary, 'secrets': {**secrets, **control_secrets}})

    assert ordinary['default_run_trace_sinks'] == ['tdsql']
    assert ordinary['batch_task_store_database'] == 'trajectory_test'
    assert config.batch_task_store_database == 'trajectory_test'
    assert ordinary['run_trace_sinks']['tdsql'] == {
        'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
        'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
        'config': {
            'host': 'tdsql.example',
            'port': 3307,
            'user': 'trajectory',
            'db': 'trajectory_test',
        },
        'secretConfig': {'password': 'tdsql-password'},
    }
    assert config.read_run_trace_credential('tdsql-password') == 'trajectory-password'
    assert ordinary['trace_database_access'] == {
        'tdsql': {
            'read_host': 'tdsql-read.example',
            'read_port': 3308,
            'database': 'trajectory_test',
            'maintenance_credential_ref': 'tdsql-maintenance',
            'enabled': True,
        }
    }
    assert 'trace_database_credentials' not in secrets
    assert control_secrets == {
        'trace_database_credentials': {
            'tdsql-maintenance': {
                'username': 'trajectory-admin',
                'password': 'maintenance-password',
            }
        }
    }
    config.validate_for_service('control-gateway')
    assert 'trajectory-password' not in json.dumps(ordinary)
    assert 'maintenance-password' not in json.dumps(ordinary)
    assert 'trajectory-password' not in repr(trace)
    assert 'maintenance-password' not in repr(trace)

    rotated = cluster_config.TDSQLTraceConfig(
        host=trace.host,
        port=trace.port,
        user=trace.user,
        password='rotated-password',
        db=trace.db,
        read_host=trace.read_host,
        read_port=trace.read_port,
        maintenance_user=trace.maintenance_user,
        maintenance_password='rotated-maintenance-password',
    )
    other_database = cluster_config.TDSQLTraceConfig(
        host=trace.host,
        port=trace.port,
        user=trace.user,
        password=trace.password,
        db='other_database',
        read_host=trace.read_host,
        read_port=trace.read_port,
        maintenance_user=trace.maintenance_user,
        maintenance_password=trace.maintenance_password,
    )
    assert trace.target_fingerprint() == rotated.target_fingerprint()
    assert trace.target_fingerprint() != other_database.target_fingerprint()
    assert 'trajectory-password' not in trace.target_fingerprint()


def test_cluster_platform_config_adds_dcdb_account_provider_without_exposing_cloud_secret() -> None:
    trace = cluster_config.TDSQLTraceConfig(
        host='21.248.112.134',
        user='nexus_dev',
        password='writer-password',
        db='runner',
        read_host='21.248.112.134',
        read_port=3306,
        account_provider_type='tencent_cloud_dcdb',
        dcdb_region='ap-nanjing',
        dcdb_instance_id='tdsqlshard-9biily2z7z',
        dcdb_api_endpoint='dcdb.internal.tencentcloudapi.com',
        dcdb_secret_id='test-secret-id',
        dcdb_secret_key='test-secret-key',
    )

    ordinary = cluster_config.platform_config_values(
        application_public_base_url='http://127.0.0.1:8082',
        application_internal_base_url='http://application:8082',
        workload_image='registry/workload:test',
        builder_image='registry/builder:test',
        namespace='workloads',
        release_name='nexus',
        database_secret_name='nexus-database',
        tdsql_trace=trace,
    )
    control_secrets = cluster_config.platform_control_gateway_secret_values(tdsql_trace=trace)
    config = DistributedConfig.model_validate({**ordinary, 'secrets': control_secrets})

    assert ordinary['trace_database_access']['tdsql']['account_provider'] == {
        'provider_type': 'tencent_cloud_dcdb',
        'region': 'ap-nanjing',
        'instance_id': 'tdsqlshard-9biily2z7z',
        'api_endpoint': 'dcdb.internal.tencentcloudapi.com',
        'account_host': '%',
    }
    assert control_secrets == {
        'trace_database_credentials': {
            'tdsql-maintenance': {
                'provider_type': 'tencent_cloud_dcdb',
                'secret_id': 'test-secret-id',
                'secret_key': 'test-secret-key',
            }
        }
    }
    config.validate_for_service('control-gateway')
    assert 'test-secret-id' not in json.dumps(ordinary)
    assert 'test-secret-key' not in json.dumps(ordinary)
    assert 'test-secret-id' not in repr(trace)
    assert 'test-secret-key' not in repr(trace)


def test_load_tdsql_trace_config_accepts_dcdb_account_provider(tmp_path: Path) -> None:
    path = tmp_path / 'tdsql-cloud.json'
    path.write_text(
        json.dumps({
            'host': '21.248.112.134',
            'port': 3306,
            'user': 'nexus_dev',
            'password': 'writer-password',
            'db': 'runner',
            'read_host': '21.248.112.134',
            'read_port': 3306,
            'account_provider_type': 'tencent_cloud_dcdb',
            'dcdb_region': 'ap-nanjing',
            'dcdb_instance_id': 'tdsqlshard-9biily2z7z',
            'dcdb_api_endpoint': 'dcdb.internal.tencentcloudapi.com',
            'dcdb_secret_id': 'test-secret-id',
            'dcdb_secret_key': 'test-secret-key',
        }),
        encoding='utf-8',
    )
    path.chmod(0o600)

    config = cluster_config.load_tdsql_trace_config(path)

    assert config.account_provider_type == 'tencent_cloud_dcdb'
    assert config.dcdb_region == 'ap-nanjing'
    assert config.dcdb_instance_id == 'tdsqlshard-9biily2z7z'
    assert config.dcdb_api_endpoint == 'dcdb.internal.tencentcloudapi.com'
    assert config.dcdb_secret_id == 'test-secret-id'
    assert config.dcdb_secret_key == 'test-secret-key'


def test_load_tdsql_trace_config_requires_private_valid_json_file(tmp_path: Path) -> None:
    path = tmp_path / 'tdsql.json'
    path.write_text(
        json.dumps({
            'host': 'tdsql.example',
            'port': 3307,
            'user': 'trajectory',
            'password': 'trajectory-password',
            'db': 'trajectory_test',
            'read_host': 'tdsql-read.example',
            'read_port': 3308,
            'maintenance_user': 'trajectory-admin',
            'maintenance_password': 'maintenance-password',
        }),
        encoding='utf-8',
    )
    path.chmod(0o600)

    config = cluster_config.load_tdsql_trace_config(path)

    assert config.host == 'tdsql.example'
    assert config.port == 3307
    assert config.user == 'trajectory'
    assert config.db == 'trajectory_test'
    assert config.password == 'trajectory-password'
    assert config.read_host == 'tdsql-read.example'
    assert config.read_port == 3308
    assert config.maintenance_user == 'trajectory-admin'
    assert config.maintenance_password == 'maintenance-password'

    path.chmod(0o640)
    with pytest.raises(ValueError, match='mode 0600'):
        cluster_config.load_tdsql_trace_config(path)


def test_seed_platform_config_scopes_trace_database_maintenance_to_control_gateway() -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args, **_kwargs):
        payload_path = Path(args[args.index('--file') + 1])
        calls.append((args, json.loads(payload_path.read_text(encoding='utf-8'))))
        return subprocess.CompletedProcess(args, 0)

    cluster_config.seed_platform_config(
        fake_run,
        python=sys.executable,
        database_url='mysql+asyncmy://control.example/nexus_dist',
        config_values={'auth_mode': 'disabled'},
        secret_values={'internal_token': 'internal'},
        role_secret_values={
            ('control-gateway', '*'): {
                'trace_database_credentials': {'tdsql-maintenance': {'username': 'admin', 'password': 'secret'}}
            }
        },
    )

    assert [
        (args[args.index('--role') + 1], args[args.index('--identity') + 1], payload) for args, payload in calls
    ] == [
        ('*', '*', {'auth_mode': 'disabled'}),
        ('*', '*', {'internal_token': 'internal'}),
        (
            'control-gateway',
            '*',
            {'trace_database_credentials': {'tdsql-maintenance': {'username': 'admin', 'password': 'secret'}}},
        ),
    ]


def test_predefined_cli_batch_args_use_real_submit_with_wait_and_config_only(tmp_path: Path) -> None:
    generator = tmp_path / 'generator.py'

    args = k8s_cluster.predefined_cli_batch_args(
        generator,
        control_url='http://127.0.0.1:18081',
        run_tag='cli-unit',
    )

    assert args[:3] == ['batch', '--generator', str(generator)]
    assert '--config-only' in args
    assert args[args.index('--environment') + 1] == 'predefined-local'
    assert args[args.index('--pool') + 1] == 'interactive'
    assert args[args.index('--control-url') + 1] == 'http://127.0.0.1:18081'
    assert args[args.index('--max-inflight') + 1] == str(k8s_cluster.EXPECTED_CLUSTER_SLOT_COUNT)
    assert '--wait' in args


def test_predefined_cli_generator_materializes_expected_config_only_tasks(tmp_path: Path) -> None:
    generator = tmp_path / 'generator.py'
    k8s_cluster.write_predefined_cli_generator(generator, run_tag='cli-unit')

    materialized = materialize_generator(generator, tmp_path / 'tasks.jsonl')
    records = [json.loads(line) for line in materialized.path.read_text(encoding='utf-8').splitlines()]

    assert materialized.total_items == k8s_cluster.PREDEFINED_BATCH_ITEM_COUNT
    assert [record['item_id'] for record in records] == [
        f'cli-unit-cli-item-{index:04d}' for index in range(k8s_cluster.PREDEFINED_BATCH_ITEM_COUNT)
    ]
    assert [record['run_input']['task_config']['message'] for record in records] == [
        f'predefined-cli-ok:{index}' for index in range(k8s_cluster.PREDEFINED_BATCH_ITEM_COUNT)
    ]


@pytest.mark.asyncio
async def test_predefined_cli_batch_report_uses_cli_execution_and_children(monkeypatch, tmp_path: Path) -> None:
    generator = tmp_path / 'generator.py'
    generator.write_text('# test generator\n', encoding='utf-8')
    execution_id = 'exe_cli_parent'
    child_ids = ['exe_cli_child_1', 'exe_cli_child_2']
    cli_calls: list[tuple[list[str], float]] = []

    def fake_cli(args: list[str], *, timeout: float):
        cli_calls.append((args, timeout))
        return {
            'args': ['batch', '--generator', '<generator>', '--wait'],
            'accepted': [{'event': 'accepted', 'execution_id': execution_id, 'status': 'queued'}],
            'result': {
                'execution_id': execution_id,
                'batch_url': f'http://control/v1/executions/{execution_id}/batch',
                'result_url': f'http://application/v1/executions/{execution_id}/result',
                'terminal': {'execution_id': execution_id, 'status': 'succeeded'},
            },
        }

    async def fake_request(method: str, url: str, **_kwargs):
        assert method == 'GET'
        if url == f'http://control/v1/executions/{execution_id}':
            return {
                'execution_id': execution_id,
                'owner_principal_id': 'local-dev',
                'status': 'succeeded',
                'workload': 'orchestration',
            }
        if url.endswith('/batch'):
            return {'summary': {'total': 2, 'succeeded': 2, 'failed': 0}}
        if url.endswith('/result'):
            return {'batch': {'total': 2, 'succeeded': 2, 'failed': 0}}
        raise AssertionError(url)

    async def fake_list_items(_control_url: str, parent_execution_id: str):
        assert parent_execution_id == execution_id
        return [
            {'item_id': f'item-{index}', 'status': 'succeeded', 'current_child_execution_id': child_id}
            for index, child_id in enumerate(child_ids)
        ]

    monkeypatch.setattr(k8s_cluster, '_run_submit_cli', fake_cli)
    monkeypatch.setattr(k8s_cluster, 'request_json', fake_request)
    monkeypatch.setattr(k8s_cluster, 'list_batch_items', fake_list_items)

    report = await k8s_cluster.assert_cli_predefined_batch(
        'http://control',
        generator=generator,
        run_tag='cli-unit',
    )

    assert len(cli_calls) == 1
    assert cli_calls[0][0] == k8s_cluster.predefined_cli_batch_args(
        generator,
        control_url='http://control',
        run_tag='cli-unit',
    )
    assert report == {
        'submission_path': 'nexus submit batch',
        'args': ['batch', '--generator', '<generator>', '--wait'],
        'execution_id': execution_id,
        'owner_principal_id': 'local-dev',
        'status': 'succeeded',
        'summary': {'total': 2, 'succeeded': 2, 'failed': 0},
        'child_execution_ids': child_ids,
    }


@pytest.mark.asyncio
async def test_focused_tdsql_flow_asserts_trace_from_real_cli_batch(monkeypatch) -> None:
    cli_batch = {
        'submission_path': 'nexus submit batch',
        'args': ['batch', '--wait'],
        'execution_id': 'exe_cli_parent',
        'owner_principal_id': 'local-dev',
        'status': 'succeeded',
        'summary': {'total': 2, 'succeeded': 2},
        'child_execution_ids': ['exe_cli_child_1', 'exe_cli_child_2'],
    }
    trace_calls: list[dict[str, object]] = []
    account_calls: list[dict[str, object]] = []

    async def noop(*_args, **_kwargs):
        return None

    async def fake_typed(*_args, **_kwargs):
        return {'batch': {'execution_id': 'exe_direct_api_parent'}}

    async def fake_cli(*_args, **_kwargs):
        return cli_batch

    async def fake_trace(_config, **kwargs):
        trace_calls.append(kwargs)
        return {'table_name': 'trajectory_span_events_pr_test'}

    async def fake_account(_control_url, _config, **kwargs):
        account_calls.append(kwargs)
        return {'account_removed': True}

    monkeypatch.setattr(k8s_cluster, 'wait_http_ok', noop)
    monkeypatch.setattr(k8s_cluster, 'assert_inventory', noop)
    monkeypatch.setattr(k8s_cluster, 'assert_typed_predefined_submissions', fake_typed)
    monkeypatch.setattr(k8s_cluster, 'assert_cli_predefined_batch', fake_cli)
    monkeypatch.setattr(k8s_cluster, 'assert_tdsql_batch_trace_in_cluster', fake_trace)
    monkeypatch.setattr(k8s_cluster, 'assert_trace_database_account_e2e', fake_account)

    report = await k8s_cluster.run_api_tests(
        'http://control',
        'http://application',
        'http://compat',
        release_name='nexus',
        kubectl='kubectl',
        namespace='workloads',
        run_opus48=False,
        opus48_timeout=1,
        opus48_gongfeng_api_base_url='http://gongfeng',
        opus48_runtime_image='current',
        check_runtime_aiomonitor=False,
        environment_values={},
        tdsql_trace=cluster_config.TDSQLTraceConfig(
            host='tdsql.example',
            user='writer',
            password='secret',
            db='trajectory',
            read_host='tdsql-read.example',
            read_port=3306,
            maintenance_user='admin',
            maintenance_password='admin-secret',
        ),
        tdsql_batch_e2e=True,
    )

    assert report['cli_predefined_batch'] == cli_batch
    assert report['trace_database_account'] == {'account_removed': True}
    assert account_calls == [
        {
            'kubectl': 'kubectl',
            'namespace': 'workloads',
            'release_name': 'nexus',
            'parent_execution_id': 'exe_cli_parent',
            'owner_principal_id': 'local-dev',
        }
    ]
    assert trace_calls == [
        {
            'kubectl': 'kubectl',
            'namespace': 'workloads',
            'release_name': 'nexus',
            'owner_principal_id': 'local-dev',
            'parent_execution_id': 'exe_cli_parent',
            'child_execution_ids': ['exe_cli_child_1', 'exe_cli_child_2'],
        }
    ]


@pytest.mark.asyncio
async def test_trace_database_account_e2e_creates_then_rotates_without_reporting_password(monkeypatch) -> None:
    table_name = k8s_cluster.principal_batch_trace_table_name(k8s_cluster.principal_trace_scope_id('local-dev'))
    catalog_account = {
        'backend_name': 'tdsql',
        'status': 'not_created',
        'batch_table_name': table_name,
    }
    created = {
        **catalog_account,
        'status': 'ready',
        'username': 'nxs_ro_0123456789abcdef01234567',
        'read_host': 'tdsql-read.example',
        'read_port': 3306,
        'database': 'trajectory',
        'password': 'first-password',
        'credential_version': 1,
    }
    rotated = {**created, 'password': 'second-password', 'credential_version': 2}
    requests: list[tuple[str, str, dict[str, object] | None]] = []
    checks: list[tuple[str, tuple[str, ...]]] = []
    dropped: list[str] = []

    async def fake_request(method: str, url: str, *, body=None, **_kwargs):
        requests.append((method, url, body))
        if method == 'GET' and url.endswith('/v1/my-environments'):
            return {'trace_database_accounts': [catalog_account]}
        if method == 'POST' and body['operation'] == 'create':
            return created
        if method == 'POST' and body['operation'] == 'reset':
            return rotated
        raise AssertionError((method, url, body))

    async def fake_check(_config, *, credential, rejected_passwords, **_kwargs):
        checks.append((credential['password'], tuple(rejected_passwords)))
        return {
            'selected_rows': 4,
            'write_denied': True,
            'rejected_password_count': len(rejected_passwords),
        }

    async def fake_drop(_config, *, username, **_kwargs):
        dropped.append(username)
        return 'removed'

    monkeypatch.setattr(k8s_cluster, 'request_json', fake_request)
    monkeypatch.setattr(k8s_cluster, 'assert_tdsql_readonly_credential_in_cluster', fake_check)
    monkeypatch.setattr(k8s_cluster, 'drop_tdsql_test_account_in_cluster', fake_drop)

    report = await k8s_cluster.assert_trace_database_account_e2e(
        'http://control',
        cluster_config.TDSQLTraceConfig(
            host='tdsql.example',
            user='writer',
            password='writer-password',
            db='trajectory',
            read_host='tdsql-read.example',
            read_port=3306,
            maintenance_user='admin',
            maintenance_password='admin-password',
        ),
        kubectl='kubectl',
        namespace='workloads',
        release_name='nexus',
        parent_execution_id='exe_cli_parent',
        owner_principal_id='local-dev',
    )

    assert checks == [
        ('first-password', ()),
        ('second-password', ('first-password',)),
    ]
    assert [request[2] for request in requests if request[0] == 'POST'] == [
        {'backend_name': 'tdsql', 'operation': 'create'},
        {'backend_name': 'tdsql', 'operation': 'reset'},
    ]
    assert report == {
        'backend_name': 'tdsql',
        'username': 'nxs_ro_0123456789abcdef01234567',
        'batch_table_name': table_name,
        'credential_versions': [1, 2],
        'create': {'selected_rows': 4, 'write_denied': True, 'rejected_password_count': 0},
        'reset': {'selected_rows': 4, 'write_denied': True, 'rejected_password_count': 1},
        'account_cleanup': 'removed',
        'account_removed': True,
    }
    assert dropped == ['nxs_ro_0123456789abcdef01234567']
    serialized = json.dumps(report)
    assert 'first-password' not in serialized
    assert 'second-password' not in serialized


@pytest.mark.asyncio
async def test_dcdb_trace_database_account_cleanup_uses_cloud_api(monkeypatch) -> None:
    requests = []

    class FakeClient:
        def DeleteAccount(self, request):  # noqa: N802
            requests.append(request)

    monkeypatch.setattr(k8s_cluster, '_create_dcdb_e2e_client', lambda _config: FakeClient())
    config = cluster_config.TDSQLTraceConfig(
        host='21.248.112.134',
        user='nexus_dev',
        password='writer-password',
        db='runner',
        read_host='21.248.112.134',
        read_port=3306,
        account_provider_type='tencent_cloud_dcdb',
        dcdb_region='ap-nanjing',
        dcdb_instance_id='tdsqlshard-9biily2z7z',
        dcdb_api_endpoint='dcdb.internal.tencentcloudapi.com',
        dcdb_secret_id='test-secret-id',
        dcdb_secret_key='test-secret-key',
    )

    cleanup_status = await k8s_cluster.drop_tdsql_test_account_in_cluster(
        config,
        kubectl='kubectl',
        namespace='workloads',
        release_name='nexus',
        username='nxs_ro_0123456789abcdef01234567',
    )

    assert len(requests) == 1
    assert requests[0].InstanceId == 'tdsqlshard-9biily2z7z'
    assert requests[0].UserName == 'nxs_ro_0123456789abcdef01234567'
    assert requests[0].Host == '%'
    assert cleanup_status == 'removed'


@pytest.mark.asyncio
async def test_dcdb_trace_database_account_cleanup_rejects_unmanaged_username(monkeypatch) -> None:
    client_created = False

    def create_client(_config):
        nonlocal client_created
        client_created = True
        return object()

    monkeypatch.setattr(k8s_cluster, '_create_dcdb_e2e_client', create_client)
    config = cluster_config.TDSQLTraceConfig(
        host='21.248.112.134',
        user='nexus_dev',
        password='writer-password',
        db='runner',
        read_host='21.248.112.134',
        read_port=3306,
        account_provider_type='tencent_cloud_dcdb',
        dcdb_region='ap-nanjing',
        dcdb_instance_id='tdsqlshard-9biily2z7z',
        dcdb_api_endpoint='dcdb.internal.tencentcloudapi.com',
        dcdb_secret_id='test-secret-id',
        dcdb_secret_key='test-secret-key',
    )

    with pytest.raises(ValueError, match='invalid TDSQL test account username'):
        await k8s_cluster.drop_tdsql_test_account_in_cluster(
            config,
            kubectl='kubectl',
            namespace='workloads',
            release_name='nexus',
            username='root',
        )

    assert client_created is False


@pytest.mark.asyncio
async def test_dcdb_trace_database_account_cleanup_revokes_access_when_delete_is_denied(monkeypatch) -> None:
    calls = []

    class FakeClient:
        def DeleteAccount(self, request):  # noqa: N802
            calls.append(('delete', request))
            raise TencentCloudSDKException('UnauthorizedOperation.PermissionDenied', 'denied', 'delete-request')

        def DescribeDatabaseObjects(self, request):  # noqa: N802
            calls.append(('inventory', request))
            return SimpleNamespace(
                Tables=[SimpleNamespace(Table='trajectory_span_events_pr_0123456789abcdef0123456789abcdef')]
            )

        def ModifyAccountPrivileges(self, request):  # noqa: N802
            calls.append(('modify', request))
            return SimpleNamespace(FlowId=42)

        def DescribeFlow(self, request):  # noqa: N802
            calls.append(('flow', request))
            return SimpleNamespace(Status=0)

    monkeypatch.setattr(k8s_cluster, '_create_dcdb_e2e_client', lambda _config: FakeClient())
    config = cluster_config.TDSQLTraceConfig(
        host='21.248.112.134',
        user='nexus_dev',
        password='writer-password',
        db='runner',
        read_host='21.248.112.134',
        read_port=3306,
        account_provider_type='tencent_cloud_dcdb',
        dcdb_region='ap-nanjing',
        dcdb_instance_id='tdsqlshard-9biily2z7z',
        dcdb_api_endpoint='dcdb.internal.tencentcloudapi.com',
        dcdb_secret_id='test-secret-id',
        dcdb_secret_key='test-secret-key',
    )

    cleanup_status = await k8s_cluster.drop_tdsql_test_account_in_cluster(
        config,
        kubectl='kubectl',
        namespace='workloads',
        release_name='nexus',
        username='nxs_ro_0123456789abcdef01234567',
    )

    assert cleanup_status == 'deactivated'
    assert [action for action, _request in calls] == ['delete', 'inventory', 'modify', 'flow']
    privilege_request = calls[2][1]
    # A revocation has to name each object and pass an empty privilege list. For the
    # object-level fields of this API an empty list means "keep the existing privileges",
    # so clearing the lists outright would report success while leaving the account able to
    # read -- including tables belonging to other principals, if it ever held a database
    # level grant. GlobalPrivileges is the one exception: it is a plain string list, where
    # an empty list does clear the grant.
    assert [(item.Table, item.Privileges) for item in privilege_request.TablePrivileges] == [
        ('trajectory_span_events_pr_0123456789abcdef0123456789abcdef', [])
    ]
    assert [(item.Database, item.Privileges) for item in privilege_request.DatabasePrivileges] == [('runner', [])]
    assert privilege_request.GlobalPrivileges == []


def test_dcdb_e2e_client_uses_configured_endpoint(monkeypatch) -> None:
    captured = {}

    def create_client(_credential, region, profile):
        captured.update(region=region, endpoint=profile.httpProfile.endpoint)
        return object()

    monkeypatch.setattr(k8s_cluster.dcdb_client, 'DcdbClient', create_client)
    config = cluster_config.TDSQLTraceConfig(
        host='21.248.112.134',
        user='nexus_dev',
        password='writer-password',
        db='runner',
        read_host='21.248.112.134',
        read_port=3306,
        account_provider_type='tencent_cloud_dcdb',
        dcdb_region='ap-nanjing',
        dcdb_instance_id='tdsqlshard-9biily2z7z',
        dcdb_api_endpoint='dcdb.internal.tencentcloudapi.com',
        dcdb_secret_id='test-secret-id',
        dcdb_secret_key='test-secret-key',
    )

    client = k8s_cluster._create_dcdb_e2e_client(config)

    assert client is not None
    assert captured == {
        'region': 'ap-nanjing',
        'endpoint': 'dcdb.internal.tencentcloudapi.com',
    }


@pytest.mark.asyncio
async def test_tdsql_batch_e2e_assertion_checks_principal_table_and_parent_child_runs(monkeypatch) -> None:
    owner_principal_id = 'principal-owner-a'
    parent_execution_id = 'exe_0123456789abcdef0123456789abcdef'
    child_execution_ids = [
        'exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        'exe_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    ]
    principal_scope_id = k8s_cluster.principal_trace_scope_id(owner_principal_id)
    principal_table = k8s_cluster.principal_batch_trace_table_name(principal_scope_id)

    class Cursor:
        def __init__(self) -> None:
            self.sql = ''
            self.params: tuple = ()
            self.deleted_parents: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute(self, sql, params=()):
            self.sql = sql
            self.params = params
            if sql.startswith('DELETE FROM'):
                self.deleted_parents.append(str(params[0]))

        async def fetchone(self):
            if 'information_schema.TABLES' in self.sql:
                return (1 if self.params[1] == principal_table else 0,)
            if self.sql.startswith('SHOW CREATE TABLE'):
                return (principal_table, f'CREATE TABLE `{principal_table}` (...) SHARDKEY=`run_id`')
            if 'COUNT(DISTINCT run_id)' in self.sql:
                return (12, len(child_execution_ids))
            if 'FROM trajectory_span_events WHERE run_id IN' in self.sql:
                return (0,)
            raise AssertionError(self.sql)

        async def fetchall(self):
            if 'information_schema.COLUMNS' in self.sql:
                return [('id',), ('run_id',), ('run_span_id',), ('parent_execution_id',), ('event_data',)]
            if self.sql.startswith('SELECT DISTINCT run_id'):
                return [(execution_id,) for execution_id in child_execution_ids]
            raise AssertionError(self.sql)

    class Connection:
        def __init__(self) -> None:
            self.cursor_instance = Cursor()
            self.closed = False
            self.commits = 0

        def cursor(self):
            return self.cursor_instance

        def close(self):
            self.closed = True

        async def commit(self):
            self.commits += 1

    connection = Connection()

    async def connect(**_kwargs):
        return connection

    monkeypatch.setattr(k8s_cluster, 'mysql_connect', connect)
    report = await k8s_cluster.assert_tdsql_batch_trace(
        cluster_config.TDSQLTraceConfig(
            host='tdsql.example',
            user='trajectory',
            password='trajectory-password',
            db='trajectory_test',
        ),
        owner_principal_id=owner_principal_id,
        parent_execution_id=parent_execution_id,
        child_execution_ids=child_execution_ids,
    )

    assert report == {
        'table_name': principal_table,
        'row_count': 12,
        'distinct_child_runs': 2,
        'expected_child_runs': 2,
        'has_trajectory_group_id': False,
        'has_parent_execution_id': True,
        'shard_key': 'run_id',
        'shared_table_rows': 0,
        'parent_rows_removed': True,
        'table_preserved': True,
        'principal_scope_id': principal_scope_id,
        'target_fingerprint': cluster_config.TDSQLTraceConfig(
            host='tdsql.example',
            user='trajectory',
            password='ignored',
            db='trajectory_test',
        ).target_fingerprint(),
    }
    assert connection.cursor_instance.deleted_parents == [parent_execution_id]
    assert connection.commits == 1
    assert connection.closed is True


@pytest.mark.asyncio
async def test_tdsql_batch_e2e_assertion_runs_inside_batch_manager_without_exposing_credentials(monkeypatch) -> None:
    config = cluster_config.TDSQLTraceConfig(
        host='tdsql.example',
        port=3307,
        user='trajectory',
        password='trajectory-password',
        db='trajectory_test',
    )
    parent_execution_id = 'exe_0123456789abcdef0123456789abcdef'
    owner_principal_id = 'principal-owner-a'
    child_execution_ids = ['exe_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa']
    principal_scope_id = k8s_cluster.principal_trace_scope_id(owner_principal_id)
    expected_report = {
        'table_name': k8s_cluster.principal_batch_trace_table_name(principal_scope_id),
        'row_count': 5,
        'distinct_child_runs': 1,
        'expected_child_runs': 1,
        'has_trajectory_group_id': False,
        'has_parent_execution_id': True,
        'shard_key': 'run_id',
        'shared_table_rows': 0,
        'parent_rows_removed': True,
        'table_preserved': True,
        'principal_scope_id': principal_scope_id,
        'target_fingerprint': config.target_fingerprint(),
    }
    calls: list[tuple[list[str], str | None]] = []

    monkeypatch.setattr(
        k8s_cluster,
        'ready_pod_resource',
        lambda *_args, **_kwargs: 'pod/nexus-batch-manager-0',
    )

    def fake_run(args, *, input_text=None, **_kwargs):
        calls.append((args, input_text))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=f'boot message\nNEXUS_TDSQL_ASSERTION={json.dumps(expected_report)}\n',
        )

    monkeypatch.setattr(k8s_cluster, 'run', fake_run)

    report = await k8s_cluster.assert_tdsql_batch_trace_in_cluster(
        config,
        kubectl='kubectl',
        namespace='workloads',
        release_name='nexus',
        owner_principal_id=owner_principal_id,
        parent_execution_id=parent_execution_id,
        child_execution_ids=child_execution_ids,
    )

    assert report == expected_report
    assert len(calls) == 1
    command, script = calls[0]
    assert command == [
        'kubectl',
        '-n',
        'workloads',
        'exec',
        '-i',
        'pod/nexus-batch-manager-0',
        '--',
        '/opt/nexus-venv/bin/python',
        '-c',
        "exec(__import__('sys').stdin.read())",
    ]
    assert script is not None
    assert 'trajectory-password' in script
    assert 'SHARED_TRACE_TABLE, principal_batch_trace_table_name' in script
    assert 'principal_trace_scope_id' in script
    assert 'FROM {SHARED_TRACE_TABLE} WHERE run_id IN' in script
    assert 'DELETE FROM `{table_name}` WHERE parent_execution_id = %s' in script
    assert 'DROP TABLE IF EXISTS' not in script
    compile(script, '<tdsql-batch-assertion>', 'exec')
    assert all('trajectory-password' not in part for part in command)


def test_predefined_cluster_cleanup_removes_helm_hook_pods(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        if args[3:5] == ['get', 'pods']:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps({
                    'items': [
                        {
                            'metadata': {
                                'name': 'nexus-e2e-db-init-finished',
                                'deletionTimestamp': '2026-08-02T00:00:00Z',
                                'finalizers': ['batch.kubernetes.io/job-tracking'],
                            },
                            'status': {'phase': 'Succeeded'},
                        },
                        {
                            'metadata': {
                                'name': 'nexus-e2e-manager-0',
                                'finalizers': ['batch.kubernetes.io/job-tracking'],
                            },
                            'status': {'phase': 'Running'},
                        },
                    ]
                }),
            )
        return subprocess.CompletedProcess(args, 0, stdout='')

    monkeypatch.setattr(k8s_cluster, 'run', fake_run)

    k8s_cluster.cleanup_release_resources(
        helm='helm',
        kubectl='kubectl',
        namespace='workloads',
        release_name='nexus-e2e',
        database_secret_name='nexus-e2e-secrets',
        mysql_secret_name='nexus-e2e-mysql',
    )

    assert [
        'kubectl',
        '-n',
        'workloads',
        'delete',
        'pods',
        '-l',
        'app.kubernetes.io/instance=nexus-e2e',
        '--ignore-not-found=true',
        '--wait=true',
    ] in calls
    assert [
        'kubectl',
        '-n',
        'workloads',
        'patch',
        'pod',
        'nexus-e2e-db-init-finished',
        '--subresource=status',
        '--type=json',
        '-p',
        '[{"op":"remove","path":"/metadata/finalizers"}]',
    ] in calls
    assert all('nexus-e2e-manager-0' not in call for call in calls if 'patch' in call)


def test_cluster_platform_config_carries_dynamic_manager_environment() -> None:
    extra_env = [
        {'name': 'TAIJI_DATASET_SERVER_URL', 'value': 'http://dataset:8084'},
        {'name': 'TAIJI_DATASET_AUTH_TOKEN', 'value': 'dataset-token'},
    ]

    ordinary = cluster_config.platform_config_values(
        application_public_base_url='http://127.0.0.1:8082',
        application_internal_base_url='http://application:8082',
        workload_image='registry/workload:test',
        builder_image='registry/builder:test',
        namespace='workloads',
        release_name='nexus',
        database_secret_name='nexus-database',
        enabled_providers=['kubernetes'],
        environment_extra_env=extra_env,
    )

    assert ordinary['environment_extra_env'] == extra_env


def test_local_application_public_url_defaults_to_application_gateway() -> None:
    assert DistributedConfig().application_public_base_url == 'http://127.0.0.1:8082'


def test_cluster_config_seed_uses_files_and_never_places_secret_values_in_argv() -> None:
    calls: list[tuple[list[str], dict]] = []

    def capture_run(command, **kwargs):
        path = Path(command[command.index('--file') + 1])
        calls.append((list(command), json.loads(path.read_text(encoding='utf-8'))))
        return SimpleNamespace(stdout='')

    cluster_config.seed_platform_config(
        capture_run,
        python=sys.executable,
        database_url='sqlite+aiosqlite:///config.db',
        config_values={'scheduler_poll_interval': 0.25},
        secret_values={'internal_token': 'never-in-argv'},
    )

    assert [call[0][5] for call in calls] == ['set', 'secret-set']
    assert calls[0][1] == {'scheduler_poll_interval': 0.25}
    assert calls[1][1] == {'internal_token': 'never-in-argv'}
    assert all('never-in-argv' not in part for command, _ in calls for part in command)


def test_predefined_db_init_accepts_succeeded_pod_when_job_condition_lags(monkeypatch):
    pod_calls = 0

    def capture_run(command, **_kwargs):
        nonlocal pod_calls
        if command[4:8] == ['job', 'predefined-db-init', '-o', 'json']:
            return SimpleNamespace(stdout=json.dumps({'metadata': {'uid': 'current-job-uid'}}))
        assert command[4:8] == ['pods', '-l', 'job-name=predefined-db-init', '-o']
        pod_calls += 1
        return SimpleNamespace(
            stdout=json.dumps({
                'items': [
                    {
                        'metadata': {
                            'name': 'predefined-db-init-stale',
                            'ownerReferences': [{'kind': 'Job', 'uid': 'stale-job-uid'}],
                        },
                        'status': {'phase': 'Succeeded'},
                    },
                    {
                        'metadata': {
                            'name': 'predefined-db-init-current',
                            'ownerReferences': [{'kind': 'Job', 'uid': 'current-job-uid'}],
                        },
                        'status': {'phase': 'Running' if pod_calls == 1 else 'Succeeded'},
                    },
                ]
            })
        )

    monkeypatch.setattr(k8s_cluster, 'run', capture_run)

    k8s_cluster.wait_for_job_pod_success(
        'kubectl',
        'namespace',
        job_name='predefined-db-init',
        timeout='10s',
    )
    assert pod_calls == 2


async def test_dynamic_environment_claims_default_quota_through_the_user_catalog(monkeypatch):
    requests = []

    async def fake_request_json(method, url, *, body=None, **_kwargs):
        requests.append((method, url, body))
        if method == 'GET' and url.endswith('/v1/environment-catalog'):
            return {
                'system_token': {'label': '\u9ed8\u8ba4'},
                'items': [
                    {
                        'environment_id': dynamic_cluster.DYNAMIC_ENVIRONMENT_ID,
                        'claimable': True,
                        'grant': None,
                    }
                ],
            }
        if method == 'PUT' and url.endswith(f'/v1/my-environments/{dynamic_cluster.DYNAMIC_ENVIRONMENT_ID}/quota'):
            return {'grant': {'limits': requests[3][2]['default_quota'], 'enabled': True}}
        return {'items': []} if method == 'GET' else {}

    monkeypatch.setattr(dynamic_cluster, 'request_json', fake_request_json)

    await dynamic_cluster.ensure_dynamic_environment('http://control', namespace='workloads')

    provider_body = requests[1][2]
    create_body = requests[3][2]
    assert provider_body['provider_config_ref'] == 'kubernetes/devcloud'
    assert provider_body['config']['namespace'] == 'workloads'
    assert provider_body['config']['auth_type'] == 'in_cluster'
    assert provider_body['config']['manager_resource_request_ratio'] == 0.1
    assert create_body['name'] == 'DevCloud Dynamic'
    assert create_body['default_quota'] == {
        'manager_count': 60,
        'cpu_millicores': 60_000,
        'memory_bytes': 64 * 1024**3,
        'slots': 1200,
        'active_batches': 4,
    }
    assert create_body['resource_limits']['cpu_millicores'] == 60_000
    assert create_body['resource_limits']['memory_bytes'] == 64 * 1024**3
    assert create_body['resource_limits']['max_provisioning_timeout_seconds'] == 900
    assert create_body['resource_limits']['max_generation_timeout_seconds'] == 600
    assert create_body['resource_limits']['max_timeout_per_run'] == dynamic_cluster.HEAVY_TIMEOUT_PER_RUN_SECONDS == 300
    assert requests[4] == ('GET', 'http://control/v1/environment-catalog', None)
    assert requests[5] == (
        'PUT',
        f'http://control/v1/my-environments/{dynamic_cluster.DYNAMIC_ENVIRONMENT_ID}/quota',
        None,
    )


def test_instance_manager_chart_bounds_batch_control_spool():
    chart_root = k8s_cluster.ROOT / 'deployment/distributed/helm/nexus-distributed'
    template = (chart_root / 'templates/instance-managers.yaml').read_text()
    values = yaml.safe_load((chart_root / 'values.yaml').read_text())

    assert values['instanceManager']['cacheSizeLimit'] == '4Gi'
    assert values['batchManager']['cacheSizeLimit'] == '4Gi'
    assert 'NEXUS_DIST_BATCH_CONTROL_SPOOL_DIR' in template
    assert 'value: /data/nexus-distributed/batch-control' in template
    assert 'kind: Service' not in template
    assert 'fieldPath: status.podIP' in template
    assert 'fieldPath: metadata.uid' in template
    assert '--endpoint-url' not in template
    assert '--service-name' not in template
    assert '--environment-id' in template
    assert '--pool-id' in template
    assert '--workload ' in template
    assert '--environment-name' not in template
    assert '--environment-key' not in template
    assert '--workload-class' not in template
    assert 'NEXUS_DIST_ENVIRONMENT_HEADLESS_SERVICE' not in template
    assert 'mountPath: /data' in template
    assert 'volumeClaimTemplates:' in template
    assert 'sizeLimit: {{ $manager.cacheSizeLimit | quote }}' in template
    assert 'readOnlyRootFilesystem: true' in template
    assert template.count('value: /tmp') == 3
    assert 'mountPath: /tmp' in template
    assert 'mountPath: /data/nexus-distributed' not in template
    assert 'mountPath: {{ .Values.dynamicEnvironment.cacheRoot | quote }}' not in template


def test_environment_controller_rbac_can_manage_workloads_and_remove_legacy_services():
    chart_root = k8s_cluster.ROOT / 'deployment/distributed/helm/nexus-distributed'
    template = (chart_root / 'templates/environment-controller-rbac.yaml').read_text()

    assert 'resources: ["services"]' in template
    assert 'resources: ["statefulsets"]' in template
    assert 'resources: ["services"]\n    verbs: ["get", "delete"]' in template
    assert 'resources: ["secrets"]\n    verbs: ["get", "create", "patch", "delete"]' in template
    assert 'resources: ["statefulsets"]\n    verbs: ["get", "create", "delete"]' in template
    assert 'resources: ["pods"]\n    verbs: ["list"]' in template
    assert 'resources: ["pods/log"]\n    verbs: ["get"]' in template
    assert 'resources: ["events"]\n    verbs: ["list"]' in template
    assert '"watch"' not in template


def test_chart_mounts_only_database_secret_and_keeps_db_pools_in_config():
    chart_root = k8s_cluster.ROOT / 'deployment/distributed/helm/nexus-distributed'
    helpers = (chart_root / 'templates/_helpers.tpl').read_text()
    values = yaml.safe_load((chart_root / 'values.yaml').read_text())

    assert 'NEXUS_DIST_DATABASE_URL_FILE' in helpers
    assert 'path: database/url' in helpers
    assert 'NEXUS_DIST_INTERNAL_TOKEN' not in helpers
    assert 'NEXUS_DIST_ARTIFACT_UPLOAD_TICKET_KEYS' not in helpers
    assert 'NEXUS_DIST_SUBMIT_TOKEN_HASH_KEYS' not in helpers
    assert 'NEXUS_DIST_TAI_APPLICATION_TOKEN' not in helpers
    assert 'dbPool' not in values['controlGateway']
    assert DistributedConfig().db_pool_size is None
    assert DistributedConfig(db_pool_size=5, db_max_overflow=5).db_pool_size == 5

    for template_name in ('gateways.yaml', 'schedulers.yaml', 'instance-managers.yaml'):
        template = (chart_root / 'templates' / template_name).read_text()
        assert 'nexus-distributed.databaseSecretVolumeMount' in template
        assert 'nexus-distributed.databaseSecretVolume' in template


def test_chart_can_enable_default_deny_core_ingress_with_explicit_allowed_sources():
    chart_root = k8s_cluster.ROOT / 'deployment/distributed/helm/nexus-distributed'
    values = yaml.safe_load((chart_root / 'values.yaml').read_text())
    template = (chart_root / 'templates/network-policy.yaml').read_text()

    assert values['networkPolicy']['enabled'] is False
    assert 'kind: NetworkPolicy' in template
    assert 'policyTypes:' in template
    assert '- Ingress' in template
    assert 'app.kubernetes.io/instance' in template
    assert 'additionalIngressFrom' in template


def test_application_gateway_upload_backpressure_lives_in_distributed_config():
    chart_root = k8s_cluster.ROOT / 'deployment/distributed/helm/nexus-distributed'
    template = (chart_root / 'templates/gateways.yaml').read_text()
    values = yaml.safe_load((chart_root / 'values.yaml').read_text())
    config = DistributedConfig()

    assert config.artifact_upload_total_concurrency == 12
    assert config.artifact_upload_input_concurrency == 4
    assert config.artifact_upload_result_concurrency == 8
    assert config.artifact_upload_retry_attempts == 8
    assert 'artifactStore' not in values
    for name in (
        'NEXUS_DIST_ARTIFACT_UPLOAD_TOTAL_CONCURRENCY',
        'NEXUS_DIST_ARTIFACT_UPLOAD_INPUT_CONCURRENCY',
        'NEXUS_DIST_ARTIFACT_UPLOAD_RESULT_CONCURRENCY',
        'NEXUS_DIST_ARTIFACT_UPLOAD_RETRY_ATTEMPTS',
    ):
        assert name not in template


def test_split_image_handles_registry_ports_and_tags():
    assert k8s_cluster.split_image('nexus-distributed-control:local') == ('nexus-distributed-control', 'local')
    assert k8s_cluster.split_image('localhost:5000/nexus-distributed-workload:local') == (
        'localhost:5000/nexus-distributed-workload',
        'local',
    )
    assert k8s_cluster.split_image('localhost:5000/nexus-distributed-workload') == (
        'localhost:5000/nexus-distributed-workload',
        '',
    )


def test_cluster_runners_use_existing_devcloud_cluster_without_kind_lifecycle():
    predefined_source = (k8s_cluster.ROOT / 'tests/integration/distributed/run_k8s_predefined_cluster.py').read_text()
    dynamic_source = (k8s_cluster.ROOT / 'tests/integration/distributed/run_k8s_dynamic_cluster.py').read_text()

    for source in (predefined_source, dynamic_source):
        assert "require_binary('kind'" not in source
        assert "'delete', 'namespace'" not in source
        assert "'create', 'cluster'" not in source
        assert "'delete', 'cluster'" not in source


async def test_execution_poll_fails_fast_on_unexpected_terminal_status(monkeypatch):
    async def failed_execution(*_args, **_kwargs):
        return {
            'execution_id': 'execution-a',
            'status': 'failed',
            'error_json': {'type': 'RunFailed', 'message': 'boom'},
        }

    monkeypatch.setattr(k8s_cluster, 'request_json', failed_execution)

    with pytest.raises(AssertionError, match='unexpected terminal state'):
        await k8s_cluster.poll_execution(
            'http://control',
            'execution-a',
            lambda item: item['status'] == 'succeeded',
            timeout=30,
        )


async def test_batch_summary_poll_fails_fast_when_control_finalizes_with_error(monkeypatch):
    async def failed_summary(*_args, **_kwargs):
        return {
            'parent_execution_id': 'execution-a',
            'generation_status': 'pending',
            'control_state': 'finalized',
            'stop_reason': 'orchestration_error',
        }

    monkeypatch.setattr(dynamic_cluster, 'request_json', failed_summary)

    with pytest.raises(AssertionError, match='batch entered unexpected terminal state'):
        await dynamic_cluster._wait_batch_summary(
            'http://control',
            'execution-a',
            lambda value: value['generation_status'] == 'complete',
            timeout=30,
        )


def test_dynamic_cleanup_deletes_release_scoped_controllers_before_pods(monkeypatch):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(stdout='{"items":[]}')

    monkeypatch.setattr(dynamic_cluster, 'run', fake_run)

    dynamic_cluster.delete_dynamic_provider_resources('kubectl', 'workloads', release_name='release-a')

    assert commands[0][4] == 'statefulsets,jobs,services,secrets'
    assert commands[0][6] == 'app.kubernetes.io/managed-by=nexus,app.kubernetes.io/instance=release-a'
    assert commands[0][-1] == '--wait=true'
    assert commands[1][4] == 'pods'
    assert commands[1][6] == 'app.kubernetes.io/managed-by=nexus,app.kubernetes.io/instance=release-a'
    assert commands[1][-1] == '--wait=false'


def test_dynamic_database_secret_sync_compares_content_without_reporting_it(monkeypatch):
    commands = []
    payloads = [
        {'data': {'database-url': 'BASE64_DATABASE_URL_SENTINEL'}},
        {
            'metadata': {
                'labels': {'app.kubernetes.io/instance': 'release-a'},
                'annotations': {
                    'nexus.io/managed-by': 'nexus',
                    'nexus.io/shared-secret': 'database-url',
                },
            },
            'data': {'database-url': 'BASE64_DATABASE_URL_SENTINEL'},
        },
    ]

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps(payloads.pop(0)))

    monkeypatch.setattr(dynamic_cluster, 'run', fake_run)

    report = dynamic_cluster.assert_database_secret_synced(
        'kubectl',
        'workloads',
        source_secret_name='control-database',
        workload_secret_name='workload-database',
        release_name='release-a',
    )

    assert report == {'secret_name': 'workload-database', 'managed_by': 'nexus'}
    assert 'BASE64_DATABASE_URL_SENTINEL' not in str(report)
    assert commands[0][5] == 'control-database'
    assert commands[1][5] == 'workload-database'


def test_dynamic_cleanup_only_tolerates_deleted_job_tracking_tombstones():
    tombstone = {
        'metadata': {
            'name': 'builder-pod',
            'deletionTimestamp': '2026-07-24T07:39:05Z',
            'finalizers': ['batch.kubernetes.io/job-tracking'],
            'labels': {
                'app.kubernetes.io/component': 'dependency-builder',
                'batch.kubernetes.io/job-name': 'builder-job',
            },
        }
    }

    assert dynamic_cluster.is_tolerated_job_tracking_tombstone(tombstone) is True
    assert (
        dynamic_cluster.is_tolerated_job_tracking_tombstone({
            **tombstone,
            'metadata': {
                **tombstone['metadata'],
                'labels': {
                    'app.kubernetes.io/component': 'db-init',
                    'batch.kubernetes.io/job-name': 'db-init-job',
                },
            },
        })
        is True
    )
    assert (
        dynamic_cluster.is_tolerated_job_tracking_tombstone({
            **tombstone,
            'metadata': {**tombstone['metadata'], 'deletionTimestamp': None},
        })
        is False
    )
    assert (
        dynamic_cluster.is_tolerated_job_tracking_tombstone({
            **tombstone,
            'metadata': {
                **tombstone['metadata'],
                'labels': {'app.kubernetes.io/component': 'db-init'},
            },
        })
        is False
    )
    assert (
        dynamic_cluster.is_tolerated_job_tracking_tombstone({
            **tombstone,
            'metadata': {
                **tombstone['metadata'],
                'finalizers': ['batch.kubernetes.io/job-tracking', 'other/finalizer'],
            },
        })
        is False
    )


def test_repeated_cluster_runner_processes_ready_live_out_of_submission_order():
    cases = k8s_cluster.build_repeated_cluster_cases('unit')
    snapshots = {case.execution_id: {'status': 'queued'} for case in cases}
    snapshots['unit-live-alpha-2'] = {'status': 'waiting_input'}
    snapshots['unit-live-beta-1'] = {'status': 'queued'}

    ready = k8s_cluster.ready_live_cases(cases, snapshots, completed_execution_ids=set())

    assert [case.execution_id for case in ready] == ['unit-live-alpha-2']


def test_aiomonitor_static_port_checks_cover_all_default_k8s_loops():
    assert k8s_cluster.AIOMONITOR_STATIC_PORT_CHECKS == (
        ('control-gateway', 50100),
        ('application-gateway', 50100),
        ('scheduler', 50100),
        ('reconciler', 50100),
        ('instance-manager', 50100),
        ('instance-manager', 50101),
        ('batch-manager', 50100),
    )


def test_compatibility_health_url_uses_subapp_health_path():
    assert k8s_cluster.compatibility_health_url('http://127.0.0.1:8082/compat/live/v1') == (
        'http://127.0.0.1:8082/compat/live/health'
    )
    assert k8s_cluster.compatibility_health_url('http://127.0.0.1:8082/compat/live') == (
        'http://127.0.0.1:8082/compat/live/health'
    )


def test_dynamic_devcloud_source_smoke_uses_typed_remote_generator_and_expected_capacity(tmp_path):
    assert dynamic_cluster.BATCH_MANAGER_WORKLOAD == 'orchestration'
    payload = dynamic_cluster.source_smoke_submission(
        source_ref={
            'artifact_id': 'source',
            'kind': 'source_package',
            'sha256': 'a' * 64,
            'size_bytes': 10,
        },
        nexus_wheel_ref={'artifact_id': 'wheel', 'kind': 'nexus_wheel', 'sha256': 'b' * 64, 'size_bytes': 20},
        uv_lock_ref={'artifact_id': 'lock', 'kind': 'uv_lock', 'sha256': 'c' * 64, 'size_bytes': 30},
        idempotency_key='source-smoke',
    )

    submission = BatchSubmission.model_validate(payload)
    assert isinstance(submission.code, SourcePackageCode)
    assert submission.environment_id == dynamic_cluster.DYNAMIC_ENVIRONMENT_ID
    assert submission.resources.batch_manager.model_dump() == {'cpu_cores': 0.25, 'memory_gib': 1.0}
    assert submission.resources.managers.model_dump() == {
        'count': 2,
        'cpu_cores': 0.25,
        'memory_gib': 1.0,
        'slots': 2,
    }
    assert submission.batch.max_inflight == 4
    assert submission.generation.source.path == dynamic_cluster.SOURCE_FIXTURE_MODULE
    generator = tmp_path / dynamic_cluster.SOURCE_FIXTURE_MODULE
    generator.write_text(dynamic_cluster._source_fixture_text())
    completed = subprocess.run(
        [
            sys.executable,
            '-m',
            'nexus.distributed.batch.materialize',
            '--generator',
            str(generator),
            '--output',
            str(tmp_path / 'tasks.jsonl'),
            '--result',
            str(tmp_path / 'result.json'),
            '--max-items',
            '100',
            '--max-bytes',
            str(1024 * 1024),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads((tmp_path / 'result.json').read_text())['total_items'] == dynamic_cluster.SOURCE_SMOKE_ITEM_COUNT


def test_dynamic_devcloud_image_smoke_uses_one_version_for_single_and_remote_batch():
    single = SingleSubmission.model_validate(
        dynamic_cluster.image_single_submission(
            image_version='user-build-20260802-001',
            idempotency_key='image-single',
        )
    )
    batch = BatchSubmission.model_validate(
        dynamic_cluster.image_batch_submission(
            image_version='user-build-20260802-001',
            idempotency_key='image-batch',
        )
    )

    assert isinstance(single.code, ImageCode)
    assert isinstance(batch.code, ImageCode)
    assert single.code.image_version == batch.code.image_version == 'user-build-20260802-001'
    assert single.run_input['imports'] == [dynamic_cluster.IMAGE_FIXTURE_MODULE]
    assert batch.generation.source.path == dynamic_cluster.IMAGE_FIXTURE_PATH
    assert batch.resources.managers.count == 2
    assert batch.resources.managers.slots == 2
    assert batch.batch.max_inflight == 4
    assert dynamic_cluster.IMAGE_SMOKE_ITEM_COUNT == 20


def test_dynamic_devcloud_image_smoke_builds_user_image_from_platform_workload():
    command = dynamic_cluster.user_workload_build_command(
        workload_image='registry.example/team/workload:platform',
        user_image='registry.example/team/workload:user-build-1',
        dockerfile=Path('Dockerfile.user'),
    )

    assert command == [
        'docker',
        'build',
        '-f',
        'Dockerfile.user',
        '--build-arg',
        'NEXUS_WORKLOAD_BASE_IMAGE=registry.example/team/workload:platform',
        '-t',
        'registry.example/team/workload:user-build-1',
        '.',
    ]


def test_dynamic_devcloud_cli_accepts_image_smoke_scenario_and_explicit_version():
    args = dynamic_cluster.parse_args(['--scenario', 'image-smoke', '--image-mode-version', 'user-build-20260802-001'])

    assert args.scenario == ['image-smoke']
    assert args.image_mode_version == 'user-build-20260802-001'


def test_dynamic_devcloud_cli_accepts_local_dataset_scenario():
    args = dynamic_cluster.parse_args(['--scenario', 'local-dataset', '--tdsql-config-file', '/protected/tdsql.json'])

    assert args.scenario == ['local-dataset']
    assert args.tdsql_config_file == Path('/protected/tdsql.json')


def test_dynamic_devcloud_cli_accepts_harbor_scenario():
    args = dynamic_cluster.parse_args(['--scenario', 'harbor', '--tdsql-config-file', '/protected/tdsql.json'])

    assert args.scenario == ['harbor']
    assert args.tdsql_config_file == Path('/protected/tdsql.json')


def test_dynamic_devcloud_cli_accepts_dependency_build_scenario_without_tdsql():
    args = dynamic_cluster.parse_args(['--scenario', 'dependency-build'])

    assert args.scenario == ['dependency-build']
    assert args.tdsql_config_file is None


def test_dynamic_devcloud_local_dataset_requires_real_tdsql_config():
    with pytest.raises(SystemExit):
        dynamic_cluster.parse_args(['--scenario', 'local-dataset'])


def test_dynamic_devcloud_harbor_requires_real_tdsql_config():
    with pytest.raises(SystemExit):
        dynamic_cluster.parse_args(['--scenario', 'harbor'])


def test_dynamic_dataset_service_is_enabled_for_local_dataset_only():
    assert dynamic_cluster.dataset_service_enabled({'local-dataset'}) is True
    assert dynamic_cluster.dataset_service_enabled({'dataset'}) is True
    assert dynamic_cluster.dataset_service_enabled({'source-smoke'}) is False


@pytest.mark.parametrize('skip_flag', ['--skip-build', '--skip-push'])
def test_dynamic_devcloud_image_smoke_skip_requires_explicit_version(skip_flag):
    with pytest.raises(SystemExit):
        dynamic_cluster.parse_args(['--scenario', 'image-smoke', skip_flag])


def test_dynamic_devcloud_single_live_smoke_reserves_bootstrap_memory():
    payload = dynamic_cluster._typed_single_live_submission(
        source_ref={
            'artifact_id': 'source',
            'kind': 'source_package',
            'sha256': 'a' * 64,
            'size_bytes': 10,
        },
        nexus_wheel_ref={'artifact_id': 'wheel', 'kind': 'nexus_wheel', 'sha256': 'b' * 64, 'size_bytes': 20},
        uv_lock_ref={'artifact_id': 'lock', 'kind': 'uv_lock', 'sha256': 'c' * 64, 'size_bytes': 30},
        run_input={'task_name': 'identity'},
        live=False,
    )

    assert payload['resources']['managers'] == {
        'count': 1,
        'cpu_cores': 0.25,
        'memory_gib': 1.0,
        'slots': 1,
    }
    assert dynamic_cluster._live_run_input()['imports'] == [
        'nexus.extensions.tasks.identity',
        'devcloud_source_package_fixture',
    ]
    source = dynamic_cluster._source_fixture_text()
    compile(source, dynamic_cluster.SOURCE_FIXTURE_MODULE, 'exec')
    assert dynamic_cluster.SOURCE_SMOKE_TASK_SLEEP_SECONDS == 20
    assert 'task_config={"index": index, "sleep_seconds": 20}' in source
    assert '@agent_registry.register("distributed_interactive_agent")' in source
    assert '@runtime_provider_registry.register("distributed_fake_runtime")' in source


def test_dynamic_devcloud_dataset_service_is_stored_in_dynamic_manager_config():
    release_name = 'dataset-e2e'
    config = dynamic_cluster.dataset_environment_config(release_name=release_name)
    manifest = json.loads(
        dynamic_cluster.fake_dataset_manifest(
            release_name=release_name,
            image='registry.example/nexus-workload:test',
        )
    )

    assert config == [
        {'name': 'TAIJI_DATASET_SERVER_URL', 'value': 'http://dataset-e2e-fake-dataset:8084'},
        {'name': 'TAIJI_DATASET_AUTH_TOKEN', 'value': dynamic_cluster.DATASET_AUTH_TOKEN},
    ]
    deployment = next(item for item in manifest['items'] if item['kind'] == 'Deployment')
    assert deployment['metadata']['name'] == 'dataset-e2e-fake-dataset'
    assert deployment['metadata']['labels']['app.kubernetes.io/managed-by'] == 'nexus-test'
    container = deployment['spec']['template']['spec']['containers'][0]
    assert container['image'] == 'registry.example/nexus-workload:test'
    server_code = container['args'][0]
    compile(server_code, '<fake-dataset>', 'exec')
    assert {item['name']: item['value'] for item in container['env']}['DATASET_SERVICE_BASE_URL'] == (
        'http://dataset-e2e-fake-dataset:8084'
    )
    assert {item['name']: item['value'] for item in container['env']}['DATASET_SERVICE_PORT'] == '8084'


def test_dynamic_fake_dataset_cleanup_deletes_each_resource_by_qualified_name(monkeypatch):
    calls = []
    monkeypatch.setattr(dynamic_cluster, 'run', lambda command, **kwargs: calls.append((command, kwargs)))

    dynamic_cluster.delete_fake_dataset_service('kubectl', 'workloads', release_name='dataset-e2e')

    assert calls == [
        (
            [
                'kubectl',
                '-n',
                'workloads',
                'delete',
                'deployment/dataset-e2e-fake-dataset',
                'service/dataset-e2e-fake-dataset',
                '--ignore-not-found=true',
                '--wait=false',
            ],
            {'check': False},
        )
    ]


def test_dynamic_fake_dataset_serves_signed_zip_byte_ranges():
    port = dynamic_cluster.free_port()
    base_url = f'http://127.0.0.1:{port}'
    environment = {
        **os.environ,
        'DATASET_AUTH_TOKEN': dynamic_cluster.DATASET_AUTH_TOKEN,
        'DATASET_ITEM_COUNT': '1',
        'DATASET_PAGE_SIZE': '1',
        'DATASET_SERVICE_BASE_URL': base_url,
        'DATASET_SERVICE_PORT': str(port),
    }
    process = subprocess.Popen(
        [sys.executable, '-u', '-c', dynamic_cluster.fake_dataset_server_code()],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    def request(path: str, *, method: str = 'GET', body: dict | None = None, headers: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        request_headers = {'secure-authentication': dynamic_cluster.DATASET_AUTH_TOKEN, **(headers or {})}
        if body is not None:
            request_headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(f'{base_url}{path}', method=method, data=data, headers=request_headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, dict(response.headers.items()), response.read()

    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                request('/health')
                break
            except (urllib.error.URLError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

        _, _, items_body = request('/api/datasets/devcloud/echo/versions/v1/items?split=test&limit=1')
        item = json.loads(items_body)['data']['items'][0]
        assert item['body'] == {
            'imports': ['tests.integration.distributed.cloud_tasks'],
            'task_name': 'dataset_task_files_echo',
            'task_config': {'expected_marker': 'remote-dataset-ok:0', 'index': 0},
        }
        files = item['files']
        assert files['type'] == 'zip'
        assert files['path'] == 'blobs/dataset-0000.zip'
        assert files['size'] > 0
        assert len(files['sha256']) == 64

        _, _, signed_body = request(
            '/api/datasets/devcloud/echo/versions/v1/files:sign?split=test',
            method='POST',
            body={'files': [files['path']]},
        )
        signed_url = json.loads(signed_body)['data']['urls'][files['path']]
        assert signed_url == f'{base_url}/{files["path"]}'

        range_request = urllib.request.Request(signed_url, headers={'Range': f'bytes=0-{files["size"] - 1}'})
        with urllib.request.urlopen(range_request, timeout=5) as response:
            assert response.status == 206
            assert response.headers['Content-Range'] == f'bytes 0-{files["size"] - 1}/{files["size"]}'
            archive = response.read()
        index = read_zip_index_from_bytes(archive)
        assert sorted(index.entries) == ['marker.txt']
        assert index.entries['marker.txt'].sha256 == hashlib.sha256(b'remote-dataset-ok:0\n').hexdigest()

        _, _, stats_body = request('/stats')
        stats = json.loads(stats_body)
        assert stats == {
            'full_blob_requests': 0,
            'pagination_requests': 1,
            'range_requests': 1,
            'sign_requests': 1,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_dynamic_fake_dataset_barrier_holds_until_released():
    port = dynamic_cluster.free_port()
    base_url = f'http://127.0.0.1:{port}'
    environment = {
        **os.environ,
        'DATASET_AUTH_TOKEN': dynamic_cluster.DATASET_AUTH_TOKEN,
        'DATASET_ITEM_COUNT': '1',
        'DATASET_PAGE_SIZE': '1',
        'DATASET_SERVICE_BASE_URL': base_url,
        'DATASET_SERVICE_PORT': str(port),
    }
    process = subprocess.Popen(
        [sys.executable, '-u', '-c', dynamic_cluster.fake_dataset_server_code()],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    def request(path: str, *, method: str = 'GET'):
        request_headers = {'secure-authentication': dynamic_cluster.DATASET_AUTH_TOKEN}
        req = urllib.request.Request(f'{base_url}{path}', method=method, headers=request_headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read())

    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                request('/health')
                break
            except (urllib.error.URLError, ConnectionError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

        _, arrived = request('/barriers/local-dataset/arrive?member=local-directory', method='POST')
        assert arrived == {'arrived': 1, 'expected': 2, 'released': False}
        _, status = request('/barriers/local-dataset')
        assert status == {'arrived': 1, 'expected': 2, 'released': False}
        _, released = request('/barriers/local-dataset/release', method='POST')
        assert released == {'arrived': 1, 'expected': 2, 'released': True}
        _, status = request('/barriers/local-dataset')
        assert status == {'arrived': 1, 'expected': 2, 'released': True}
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_dynamic_local_dataset_fixture_contains_directory_and_zip_items(tmp_path: Path):
    dataset_root = dynamic_cluster.write_local_dataset_fixture(tmp_path / 'local-dataset', barrier_name='local-dataset')

    dataset = yaml.safe_load((dataset_root / 'dataset.yaml').read_text())
    item_paths = dataset['splits']['test']['items']
    assert item_paths == [
        'items/test/local-directory/item.yaml',
        'items/test/local-zip/item.yaml',
    ]

    directory_item = yaml.safe_load((dataset_root / item_paths[0]).read_text())
    zip_item = yaml.safe_load((dataset_root / item_paths[1]).read_text())
    assert directory_item['files']['type'] == 'directory'
    assert directory_item['files']['path'] == 'files'
    assert zip_item['files']['type'] == 'zip'
    assert zip_item['files']['path'] == 'files.zip'
    assert directory_item['body']['task_config']['barrier_name'] == 'local-dataset'
    assert directory_item['body']['task_config']['barrier_member'] == 'local-directory'
    assert zip_item['body']['task_config']['barrier_name'] == 'local-dataset'
    assert zip_item['body']['task_config']['barrier_member'] == 'local-zip'
    assert (dataset_root / 'items/test/local-directory/files/marker.txt').read_text() == 'local-directory-ok\n'
    assert (dataset_root / 'items/test/local-zip/files.zip').is_file()


def test_dynamic_harbor_fixture_contains_task_files_and_barrier_contract(tmp_path: Path):
    dataset_root = dynamic_cluster.write_harbor_dataset_fixture(
        tmp_path / 'harbor',
        docker_image='registry.example/nexus/workload:test',
        release_name='harbor-e2e',
        barrier_name='harbor-barrier',
    )

    task_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    assert [path.name for path in task_dirs] == list(dynamic_cluster.HARBOR_DATASET_ITEM_IDS)
    for task_dir in task_dirs:
        task_config = task_dir.joinpath('task.toml').read_text()
        assert 'docker_image = "registry.example/nexus/workload:test"' in task_config
        assert 'DATASET_BARRIER_NAME = "harbor-barrier"' in task_config
        assert (task_dir / 'instruction.md').is_file()
        assert (task_dir / 'fixture' / 'marker.txt').is_file()
        assert (task_dir / 'tests' / 'test.sh').stat().st_mode & 0o111


@pytest.mark.asyncio
async def test_dynamic_local_dataset_observes_assignment_before_barrier_release(monkeypatch):
    events: list[str] = []

    async def fake_request_json(method, url, **_kwargs):
        events.append(f'execution:{method}:{url}')
        return {'status': 'running'}

    async def fake_release(*_args, **_kwargs):
        events.append('barrier-release')
        return {'released': True}

    def fake_workspace(*_args, **_kwargs):
        events.append(f'workspace:{_kwargs["allocation_id"]}')
        return {'workspace': 'clean'}

    def fake_records(*_args, **_kwargs):
        events.append('_read_v3_task_records')
        return {'records': {'local-directory': {}}}

    monkeypatch.setattr(dynamic_cluster, 'request_json', fake_request_json)
    monkeypatch.setattr(dynamic_cluster, '_wait_for_local_dataset_barrier_and_release', fake_release)
    monkeypatch.setattr(dynamic_cluster, '_assert_batch_workspace_sanitized', fake_workspace)
    monkeypatch.setattr(dynamic_cluster, '_read_v3_task_records', fake_records)
    monkeypatch.setattr(dynamic_cluster, '_assert_assignment_task_files_clean', lambda *_args, **_kwargs: {})
    result = await dynamic_cluster._observe_local_dataset_active_state(
        'kubectl',
        'namespace',
        'control-url',
        'execution-id',
        orchestration_allocation_id='orchestration-id',
        execution_allocation_id='execution-id',
        release_name='release',
        barrier_name='barrier',
        dataset_root=Path('/submit-host/local-dataset'),
        tdsql_trace=cluster_config.TDSQLTraceConfig(
            host='tdsql.example',
            port=3307,
            user='writer',
            password='password',
            db='dataset_test',
        ),
    )
    assert events == [
        'workspace:orchestration-id',
        'execution:GET:control-url/v1/executions/execution-id',
        '_read_v3_task_records',
        'barrier-release',
    ]
    assert result['workspace'] == {'workspace': 'clean'}


@pytest.mark.asyncio
async def test_dynamic_local_dataset_terminal_cleanup_probes_assignment_before_task_store_cleanup(monkeypatch):
    events: list[str] = []

    async def fake_poll_execution(*_args, **_kwargs):
        events.append('terminal')
        return {'status': 'succeeded'}

    async def fake_task_store(*_args, **_kwargs):
        events.append('task-store-clean')
        return {'records': {}}

    def fake_assignment_cleanup(*_args, **_kwargs):
        events.append('assignment-clean')
        return {'pod': 'execution-pod', 'stale_assignment_task_files': []}

    monkeypatch.setattr(dynamic_cluster, 'poll_execution', fake_poll_execution)
    monkeypatch.setattr(dynamic_cluster, '_wait_v3_task_records_removed', fake_task_store)
    monkeypatch.setattr(dynamic_cluster, '_assert_assignment_task_files_clean', fake_assignment_cleanup)

    result = await dynamic_cluster._wait_local_dataset_terminal_cleanup(
        'kubectl',
        'namespace',
        'control-url',
        'execution-id',
        release_name='release',
        tdsql_trace=cluster_config.TDSQLTraceConfig(
            host='tdsql.example',
            port=3307,
            user='writer',
            password='password',
            db='dataset_test',
        ),
        forbidden_path=Path('/submit-host/local-dataset'),
        allocation_id='execution-allocation-id',
    )

    assert events == ['terminal', 'assignment-clean', 'task-store-clean']
    assert result == {
        'terminal': {'status': 'succeeded'},
        'task_store': {'records': {}},
        'assignment': {'pod': 'execution-pod', 'stale_assignment_task_files': []},
    }


def test_dynamic_assignment_cleanup_rejects_release_before_observation(monkeypatch):
    def released_run(args, **_kwargs):
        return subprocess.CompletedProcess(
            args,
            1,
            stdout='',
            stderr='Error from server (NotFound): statefulsets.apps "nexus-execution-id-af369142" not found',
        )

    monkeypatch.setattr(dynamic_cluster, 'run', released_run)

    with pytest.raises(AssertionError, match='cleanup was not observed before allocation release'):
        dynamic_cluster._assert_assignment_task_files_clean(
            'kubectl',
            'namespace',
            allocation_id='execution-id',
        )


def test_dynamic_assignment_cleanup_rejects_release_when_deleting_workload_has_no_ready_pod(monkeypatch):
    commands: list[list[str]] = []
    responses = [
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({'spec': {'selector': {'matchLabels': {'nexus.io/allocation-id': 'execution-id'}}}}),
            stderr='',
        ),
        subprocess.CompletedProcess([], 0, stdout=json.dumps({'items': []}), stderr=''),
        subprocess.CompletedProcess(
            [],
            1,
            stdout='Error from server (NotFound): statefulsets.apps "nexus-execution-id-af369142" not found',
            stderr='',
        ),
    ]

    def deleting_run(args, **_kwargs):
        commands.append(args)
        return responses.pop(0)

    monkeypatch.setattr(dynamic_cluster, 'run', deleting_run)
    monkeypatch.setattr(dynamic_cluster.time, 'sleep', lambda _seconds: None)

    with pytest.raises(AssertionError, match='cleanup was not observed before allocation release'):
        dynamic_cluster._assert_assignment_task_files_clean(
            'kubectl',
            'namespace',
            allocation_id='execution-id',
        )
    assert [command[4] for command in commands] == ['statefulset', 'pods', 'statefulset']


def test_dynamic_assignment_cleanup_rejects_release_when_pod_disappears_before_exec(monkeypatch):
    commands: list[list[str]] = []
    responses = [
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({'spec': {'selector': {'matchLabels': {'nexus.io/allocation-id': 'execution-id'}}}}),
            stderr='',
        ),
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({
                'items': [
                    {
                        'metadata': {'name': 'execution-pod'},
                        'status': {'conditions': [{'type': 'Ready', 'status': 'True'}]},
                    }
                ]
            }),
            stderr='',
        ),
        subprocess.CompletedProcess([], 1, stdout='Error from server (NotFound): pods "execution-pod" not found'),
        subprocess.CompletedProcess(
            [],
            1,
            stdout='Error from server (NotFound): statefulsets.apps "nexus-execution-id-af369142" not found',
            stderr='',
        ),
    ]

    def disappearing_run(args, **_kwargs):
        commands.append(args)
        return responses.pop(0)

    monkeypatch.setattr(dynamic_cluster, 'run', disappearing_run)
    monkeypatch.setattr(dynamic_cluster.time, 'sleep', lambda _seconds: None)

    with pytest.raises(AssertionError, match='cleanup was not observed before allocation release'):
        dynamic_cluster._assert_assignment_task_files_clean(
            'kubectl',
            'namespace',
            allocation_id='execution-id',
        )
    assert [(command[4], command[5] if len(command) > 5 else None) for command in commands] == [
        ('statefulset', 'nexus-execution-id-af369142'),
        ('pods', '-l'),
        ('pod/execution-pod', '-c'),
        ('statefulset', 'nexus-execution-id-af369142'),
    ]


def test_dynamic_assignment_cleanup_does_not_hide_unrelated_kubectl_failure(monkeypatch):
    def failed_run(args, **_kwargs):
        return subprocess.CompletedProcess(args, 1, stdout='', stderr='forbidden')

    monkeypatch.setattr(dynamic_cluster, 'run', failed_run)

    with pytest.raises(subprocess.CalledProcessError):
        dynamic_cluster._assert_assignment_task_files_clean(
            'kubectl',
            'namespace',
            allocation_id='execution-id',
        )


def test_dynamic_assignment_cleanup_does_not_treat_forbidden_statefulset_message_as_not_found(monkeypatch):
    def forbidden_run(args, **_kwargs):
        return subprocess.CompletedProcess(
            args,
            1,
            stdout=(
                'Error from server (Forbidden): statefulsets.apps "nexus-execution-id-af369142" is forbidden: '
                'authorization dependency not found'
            ),
            stderr='',
        )

    monkeypatch.setattr(dynamic_cluster, 'run', forbidden_run)

    with pytest.raises(subprocess.CalledProcessError) as error:
        dynamic_cluster._assert_assignment_task_files_clean(
            'kubectl',
            'namespace',
            allocation_id='execution-id',
        )
    assert 'Forbidden' in error.value.stdout


def test_dynamic_assignment_cleanup_does_not_hide_unrelated_exec_forbidden(monkeypatch):
    responses = [
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({'spec': {'selector': {'matchLabels': {'nexus.io/allocation-id': 'execution-id'}}}}),
            stderr='',
        ),
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({
                'items': [
                    {
                        'metadata': {'name': 'execution-pod'},
                        'status': {'conditions': [{'type': 'Ready', 'status': 'True'}]},
                    }
                ]
            }),
            stderr='',
        ),
        subprocess.CompletedProcess([], 1, stdout='Error from server (Forbidden)', stderr=''),
    ]

    monkeypatch.setattr(dynamic_cluster, 'run', lambda *_args, **_kwargs: responses.pop(0))

    with pytest.raises(subprocess.CalledProcessError) as error:
        dynamic_cluster._assert_assignment_task_files_clean(
            'kubectl',
            'namespace',
            allocation_id='execution-id',
        )
    assert 'Forbidden' in error.value.stdout


def test_dynamic_assignment_cleanup_does_not_treat_internal_exec_message_as_pod_not_found(monkeypatch):
    responses = [
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({'spec': {'selector': {'matchLabels': {'nexus.io/allocation-id': 'execution-id'}}}}),
            stderr='',
        ),
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({
                'items': [
                    {
                        'metadata': {'name': 'execution-pod'},
                        'status': {'conditions': [{'type': 'Ready', 'status': 'True'}]},
                    }
                ]
            }),
            stderr='',
        ),
        subprocess.CompletedProcess(
            [],
            1,
            stdout='Error from server (InternalError): container runtime dependency not found',
            stderr='',
        ),
    ]

    monkeypatch.setattr(dynamic_cluster, 'run', lambda *_args, **_kwargs: responses.pop(0))

    with pytest.raises(subprocess.CalledProcessError) as error:
        dynamic_cluster._assert_assignment_task_files_clean(
            'kubectl',
            'namespace',
            allocation_id='execution-id',
        )
    assert 'InternalError' in error.value.stdout


def test_dynamic_assignment_cleanup_timeout_stops_after_terminal_probe(monkeypatch):
    commands: list[list[str]] = []
    responses = [
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({'spec': {'selector': {'matchLabels': {'nexus.io/allocation-id': 'execution-id'}}}}),
            stderr='',
        ),
        subprocess.CompletedProcess([], 0, stdout=json.dumps({'items': []}), stderr=''),
    ]

    def deleting_run(args, **_kwargs):
        commands.append(args)
        return responses.pop(0)

    monkeypatch.setattr(dynamic_cluster, 'run', deleting_run)

    with pytest.raises(TimeoutError, match='assignment task files were not cleaned before allocation release'):
        dynamic_cluster._assert_assignment_task_files_clean(
            'kubectl',
            'namespace',
            allocation_id='execution-id',
            timeout=0,
        )
    assert [command[4] for command in commands] == ['statefulset', 'pods']


def test_dynamic_assignment_cleanup_passes_fresh_remaining_timeout_to_each_kubectl_call(monkeypatch):
    clock = [100.0]
    call_timeouts: list[float | None] = []
    responses = [
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({'spec': {'selector': {'matchLabels': {'nexus.io/allocation-id': 'execution-id'}}}}),
            stderr='',
        ),
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({
                'items': [
                    {
                        'metadata': {'name': 'execution-pod'},
                        'status': {'conditions': [{'type': 'Ready', 'status': 'True'}]},
                    }
                ]
            }),
            stderr='',
        ),
        subprocess.CompletedProcess(
            [],
            0,
            stdout=json.dumps({'roots': ['/tmp/assignment-files-clean'], 'files': []}),
            stderr='',
        ),
    ]

    def timed_run(_args, **kwargs):
        call_timeouts.append(kwargs.get('timeout'))
        clock[0] += 1
        return responses.pop(0)

    monkeypatch.setattr(dynamic_cluster.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(dynamic_cluster, 'run', timed_run)

    result = dynamic_cluster._assert_assignment_task_files_clean(
        'kubectl',
        'namespace',
        allocation_id='execution-id',
        timeout=6,
    )

    assert result['stale_assignment_task_files'] == []
    assert call_timeouts == pytest.approx([6, 5, 4])


@pytest.mark.parametrize('timed_out_call', [0, 1, 2])
def test_dynamic_assignment_cleanup_translates_kubectl_timeout_to_domain_timeout(monkeypatch, timed_out_call):
    calls = 0

    def timed_run(args, **kwargs):
        nonlocal calls
        current_call = calls
        calls += 1
        if current_call == timed_out_call:
            raise subprocess.TimeoutExpired(args, kwargs.get('timeout'))
        if args[4] == 'statefulset':
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps({'spec': {'selector': {'matchLabels': {'nexus.io/allocation-id': 'execution-id'}}}}),
                stderr='',
            )
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps({
                'items': [
                    {
                        'metadata': {'name': 'execution-pod'},
                        'status': {'conditions': [{'type': 'Ready', 'status': 'True'}]},
                    }
                ]
            }),
            stderr='',
        )

    monkeypatch.setattr(dynamic_cluster, 'run', timed_run)

    with pytest.raises(TimeoutError, match='assignment task files were not cleaned before allocation release') as error:
        dynamic_cluster._assert_assignment_task_files_clean(
            'kubectl',
            'namespace',
            allocation_id='execution-id',
            timeout=5,
        )
    assert type(error.value) is TimeoutError


def test_dynamic_assignment_cleanup_sleep_does_not_overshoot_deadline(monkeypatch):
    clock = [50.0]
    sleeps: list[float] = []

    def deleting_run(args, **_kwargs):
        if args[4] == 'statefulset':
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps({'spec': {'selector': {'matchLabels': {'nexus.io/allocation-id': 'execution-id'}}}}),
                stderr='',
            )
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({'items': []}), stderr='')

    def advance_clock(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(dynamic_cluster.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(dynamic_cluster.time, 'sleep', advance_clock)
    monkeypatch.setattr(dynamic_cluster, 'run', deleting_run)

    with pytest.raises(TimeoutError, match='assignment task files were not cleaned before allocation release'):
        dynamic_cluster._assert_assignment_task_files_clean(
            'kubectl',
            'namespace',
            allocation_id='execution-id',
            timeout=2.5,
            poll_interval=10,
        )
    assert sleeps == [2.5]


def test_dynamic_assignment_cleanup_inspects_manager_assignment_temp_root(monkeypatch):
    def fake_run(command, **_kwargs):
        rendered = '\0'.join(command)
        if 'get\0statefulset' in rendered:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({'spec': {'selector': {'matchLabels': {'nexus.io/allocation-id': 'execution-id'}}}}),
                stderr='',
            )
        if 'get\0pods' in rendered:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({
                    'items': [
                        {
                            'metadata': {'name': 'execution-pod'},
                            'status': {'conditions': [{'type': 'Ready', 'status': 'True'}]},
                        }
                    ]
                }),
                stderr='',
            )
        if 'NEXUS_DIST_BATCH_CONTROL_SPOOL_DIR' in rendered and 'assignment-files-*' in rendered:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({
                    'roots': ['/data/nexus-distributed/batch-control/assignment-files-fixture'],
                    'files': [],
                }),
                stderr='',
            )
        raise AssertionError(command)

    monkeypatch.setattr(dynamic_cluster, 'run', fake_run)

    result = dynamic_cluster._assert_assignment_task_files_clean(
        'kubectl',
        'namespace',
        allocation_id='alc-local-dataset',
    )

    assert result == {
        'pod': 'execution-pod',
        'assignment_temp_roots': ['/data/nexus-distributed/batch-control/assignment-files-fixture'],
        'stale_assignment_task_files': [],
    }


def test_dynamic_workspace_sanitization_uses_manager_spool_dir(monkeypatch):
    def fake_run(command, **_kwargs):
        if 'statefulset' in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({'spec': {'selector': {'matchLabels': {'nexus.io/allocation-id': 'alc-local'}}}}),
                stderr='',
            )
        if command[:5] == ['kubectl', '-n', 'namespace', 'get', 'pods']:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({
                    'items': [
                        {
                            'metadata': {'name': 'orchestration-pod'},
                            'status': {'conditions': [{'type': 'Ready', 'status': 'True'}]},
                        }
                    ]
                }),
                stderr='',
            )
        if 'exec' in command:
            script = next(value for value in command if 'batch-control' in value)
            assert 'NEXUS_DIST_BATCH_CONTROL_SPOOL_DIR' in script
            assert 'Path(os.environ.get("NEXUS_DIST_BATCH_CONTROL_SPOOL_DIR"' in script
            assert '/data/nexus-distributed/batch-control' not in script
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({'files': ['/data/nexus-environment/batch-control/tasks.jsonl'], 'bad': []}),
                stderr='',
            )
        raise AssertionError(command)

    monkeypatch.setattr(dynamic_cluster, 'run', fake_run)

    result = dynamic_cluster._assert_batch_workspace_sanitized(
        'kubectl',
        'namespace',
        allocation_id='alc-local',
        forbidden_path=Path('/submit-host/local-dataset'),
    )

    assert result == {
        'pod': 'orchestration-pod',
        'task_list_files': ['/data/nexus-environment/batch-control/tasks.jsonl'],
        'forbidden_path_matches': [],
    }


def test_dynamic_tdsql_task_record_check_allows_normalized_null_task_files():
    script = dynamic_cluster._tdsql_v3_task_record_assertion_script(
        cluster_config.TDSQLTraceConfig(
            host='tdsql.example',
            port=3307,
            user='writer',
            password='password',
            db='dataset_test',
        ),
        parent_execution_id='exe-local',
        forbidden_path=Path('/submit-host/local-dataset'),
        expect_present=True,
    )

    assert "assert decoded.run_input.get('task_files') is None" in script


def test_dynamic_local_dataset_inspects_v3_records_without_raw_sql_cleanup(monkeypatch):
    config = cluster_config.TDSQLTraceConfig(
        host='tdsql.example',
        port=3307,
        user='dataset-writer',
        password='dataset-password',
        db='dataset_test',
    )
    parent_execution_id = 'exe_0123456789abcdef0123456789abcdef'
    forbidden_path = Path('/submit-host/local-dataset')
    expected_records = {
        'local-directory': {
            'record_version': 3,
            'record_size_bytes': 321,
            'data_ref': {'offset': 0, 'length': 123},
            'frame_zip_size': 99,
            'frame_zip_sha256': 'a' * 64,
        },
        'local-zip': {
            'record_version': 3,
            'record_size_bytes': 654,
            'data_ref': {'offset': 123, 'length': 456},
            'frame_zip_size': 432,
            'frame_zip_sha256': 'b' * 64,
        },
    }
    expected_report = {
        'table_name': 'batch_task_records',
        'shard_key': 'parent_execution_id',
        'records': expected_records,
        'remaining_parent_rows': 2,
    }
    calls: list[tuple[list[str], str | None, dict]] = []

    monkeypatch.setattr(
        dynamic_cluster,
        'ready_pod_resource',
        lambda *_args, **_kwargs: 'pod/nexus-control-gateway-0',
    )

    def fake_run(args, *, input_text=None, **kwargs):
        calls.append((args, input_text, kwargs))
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=f'boot message\nNEXUS_TDSQL_TASK_RECORDS={json.dumps(expected_report)}\n',
        )

    monkeypatch.setattr(dynamic_cluster, 'run', fake_run)

    report = dynamic_cluster._read_v3_task_records(
        'kubectl',
        'workloads',
        'nexus',
        config=config,
        parent_execution_id=parent_execution_id,
        forbidden_path=forbidden_path,
        expect_present=True,
    )

    assert report == {
        **expected_report,
        'target_fingerprint': config.target_fingerprint(),
    }
    assert len(calls) == 1
    command, script, kwargs = calls[0]
    assert command == [
        'kubectl',
        '-n',
        'workloads',
        'exec',
        '-i',
        'pod/nexus-control-gateway-0',
        '--',
        '/opt/nexus-venv/bin/python',
        '-c',
        "exec(__import__('sys').stdin.read())",
    ]
    assert kwargs == {'capture': True, 'timeout': 90}
    assert script is not None
    compile(script, '<tdsql-task-records>', 'exec')
    assert 'DELETE FROM batch_task_records' not in script
    assert 'dataset-password' in script
    assert all('dataset-password' not in part for part in command)
    assert 'dataset-password' not in json.dumps(report)


def test_dynamic_devcloud_heavy_batch_is_2000_sleep_tasks_on_25_by_20_slots(tmp_path):
    values = yaml.safe_load(dynamic_cluster.DYNAMIC_VALUES.read_text())
    task_list = tmp_path / 'tasks.jsonl'
    manifest = dynamic_cluster.write_heavy_task_list(task_list)
    ref = {
        'artifact_id': 'task-list',
        'kind': 'task_list',
        'sha256': manifest['sha256'],
        'size_bytes': manifest['size_bytes'],
    }
    payload = dynamic_cluster.heavy_batch_submission(
        task_list_ref=ref,
        manifest=manifest,
        idempotency_key='heavy',
    )

    submission = BatchSubmission.model_validate(payload)
    assert isinstance(submission.code, ConfigOnlyCode)
    assert manifest['item_count'] == dynamic_cluster.HEAVY_ITEM_COUNT == 2000
    assert submission.resources.batch_manager.cpu_cores == 1.0
    assert submission.resources.batch_manager.memory_gib == 1.0
    assert submission.resources.managers.count == dynamic_cluster.HEAVY_MANAGER_COUNT == 25
    assert submission.resources.managers.cpu_cores == 1.0
    assert submission.resources.managers.slots == dynamic_cluster.HEAVY_SLOTS_PER_MANAGER == 20
    assert submission.resources.managers.memory_gib == 1.0
    assert submission.batch.max_inflight == dynamic_cluster.HEAVY_MAX_INFLIGHT == 500
    assert submission.batch.timeout_per_run == dynamic_cluster.HEAVY_TIMEOUT_PER_RUN_SECONDS == 300
    assert dynamic_cluster.heavy_peak_target(recovery_mode=None) == 450
    assert dynamic_cluster.heavy_peak_target(recovery_mode='execution') == 400
    assert dynamic_cluster.heavy_peak_target(recovery_mode='batch') == 400
    assert values['scheduler'] == {'replicas': 1}
    assert values['dynamicEnvironment']['enabledProviders'] == ['kubernetes']
    assert 'providerConfigs' not in values['dynamicEnvironment']
    assert 'extraEnv' not in values['dynamicEnvironment']
    assert 'artifactStore' not in values
    first, *_, last = [json.loads(line) for line in task_list.read_text().splitlines()]
    assert first['version'] == 1
    assert first['run_input']['imports'] == ['tests.integration.distributed.cloud_tasks']
    assert first['run_input']['task_name'] == 'sleep_echo'
    assert first['run_input']['task_config']['sleep_seconds'] == dynamic_cluster.HEAVY_TASK_SLEEP_SECONDS == 20
    assert last['item_id'] == 'item-000000001999'


def test_dynamic_devcloud_heavy_failure_diagnostics_group_deadline_margins_and_managers():
    facts = {
        'succeeded': {
            'execution_id': 'succeeded',
            'status': 'succeeded',
            'updated_at': '2026-07-24T00:00:30',
            'execution_expires_at': '2026-07-24T00:02:00',
            'result_artifact_id': 'artifact-succeeded',
            'stop_reason': None,
            'current_assignment': {'manager_id': 'manager-0'},
        },
        'expired-1': {
            'execution_id': 'expired-1',
            'status': 'expired',
            'updated_at': '2026-07-24T00:02:02',
            'execution_expires_at': '2026-07-24T00:02:00',
            'result_artifact_id': None,
            'stop_reason': 'execution_ttl',
            'current_assignment': {'manager_id': 'manager-1'},
        },
        'expired-2': {
            'execution_id': 'expired-2',
            'status': 'expired',
            'updated_at': '2026-07-24T00:02:04',
            'execution_expires_at': '2026-07-24T00:02:00',
            'result_artifact_id': None,
            'stop_reason': 'execution_ttl',
            'current_assignment': {'manager_id': 'manager-1'},
        },
    }

    diagnostics = dynamic_cluster.heavy_child_diagnostics(facts)

    assert diagnostics['status_counts'] == {'expired': 2, 'succeeded': 1}
    assert diagnostics['result_artifact_counts'] == {'expired': 0, 'succeeded': 1}
    assert diagnostics['deadline_margin_seconds']['succeeded'] == {'min': 90.0, 'median': 90.0, 'max': 90.0}
    assert diagnostics['deadline_margin_seconds']['expired'] == {'min': -4.0, 'median': -3.0, 'max': -2.0}
    assert diagnostics['expired_by_manager'] == {'manager-1': 2}
    assert diagnostics['stop_reason_counts'] == {'execution_ttl': 2}
    assert [item['execution_id'] for item in diagnostics['expired_samples']] == ['expired-1', 'expired-2']


def test_dynamic_devcloud_mysql_status_ignores_client_warning_and_uses_pod_secret(monkeypatch):
    commands = []

    monkeypatch.setattr(dynamic_cluster, 'ready_pod_resource', lambda *_args, **_kwargs: 'pod/mysql')

    def fake_run(command, *, capture, check):
        assert capture is True
        assert check is False
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=(
                'mysql: [Warning] client notice\n'
                'Connections\t421\n'
                'Max_used_connections\t24\n'
                'Slow_queries\t0\n'
                'Threads_connected\t18\n'
            ),
        )

    monkeypatch.setattr(dynamic_cluster, 'run', fake_run)

    assert dynamic_cluster.mysql_status('kubectl', 'namespace', 'phase2-e2e') == {
        'Connections': 421,
        'Max_used_connections': 24,
        'Slow_queries': 0,
        'Threads_connected': 18,
    }
    assert any('$MYSQL_ROOT_PASSWORD' in item for item in commands[0])
    mysql_command = next(item for item in commands[0] if item.startswith('exec mysql '))
    assert '--protocol=tcp' in mysql_command
    assert '-h127.0.0.1' in mysql_command
    assert not any('nexus_dist_local' in item for item in commands[0])


def test_dynamic_devcloud_mysql_query_retries_transient_kubectl_exec_failure(monkeypatch):
    attempts = []
    responses = [
        SimpleNamespace(returncode=1, stdout='error: transient exec failure'),
        SimpleNamespace(returncode=0, stdout='Connections\t1\n'),
    ]

    monkeypatch.setattr(dynamic_cluster, 'ready_pod_resource', lambda *_args, **_kwargs: 'pod/mysql')
    monkeypatch.setattr(dynamic_cluster.time, 'sleep', lambda _seconds: None)

    def fake_run(command, **kwargs):
        attempts.append((command, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(dynamic_cluster, 'run', fake_run)

    assert dynamic_cluster._mysql_query('kubectl', 'namespace', 'phase2-e2e', 'SHOW STATUS;') == 'Connections\t1\n'
    assert len(attempts) == 2
    assert all(kwargs == {'capture': True, 'check': False} for _, kwargs in attempts)


def test_dynamic_devcloud_defaults_to_lightweight_config_only_workload_image():
    args = dynamic_cluster.parse_args(['--tdsql-config-file', '/protected/tdsql.json'])
    dockerfile = dynamic_cluster.TEST_WORKLOAD_DOCKERFILE.read_text()

    assert args.kubeconfig == k8s_cluster.DEFAULT_DEVCLOUD_KUBECONFIG
    assert args.namespace == k8s_cluster.DEFAULT_DEVCLOUD_NAMESPACE
    assert args.release_name == 'nexus-phase2-devcloud-dynamic'
    assert args.workload_image == 'mirrors.tencent.com/nexus_utopia/nexus-distributed-workload:phase2-devcloud'
    assert args.builder_image == 'mirrors.tencent.com/nexus_utopia/nexus-distributed-builder:phase2-devcloud'
    assert args.workload_dockerfile == dynamic_cluster.TEST_WORKLOAD_DOCKERFILE
    assert 'ARG NEXUS_WORKLOAD_BASE_IMAGE' in dockerfile
    assert 'FROM ${NEXUS_WORKLOAD_BASE_IMAGE}' in dockerfile
    assert '/opt/nexus-workload-venv' not in dockerfile
    assert 'COPY tests ./tests' in dockerfile
    assert 'RUN chmod -R a+rX /app' in dockerfile.splitlines()
    assert 'nexus-runtime' not in dockerfile


def test_dynamic_devcloud_test_workload_imports_fixture_from_config_only_workspace(tmp_path: Path):
    from nexus.distributed.instance_manager import bootstrap

    image_root = tmp_path / 'image-root'
    fixture_package = image_root / 'tests/integration/distributed'
    fixture_package.mkdir(parents=True)
    for package in (image_root / 'tests', image_root / 'tests/integration', fixture_package):
        (package / '__init__.py').write_text('', encoding='utf-8')
    (fixture_package / 'cloud_tasks.py').write_text("MARKER = 'fixture-imported'\n", encoding='utf-8')

    declared_pythonpath = next(
        (
            line.removeprefix('ENV PYTHONPATH=')
            for line in dynamic_cluster.TEST_WORKLOAD_DOCKERFILE.read_text(encoding='utf-8').splitlines()
            if line.startswith('ENV PYTHONPATH=')
        ),
        None,
    )
    image_pythonpath = (
        os.pathsep.join(
            str(image_root) if entry == '/app' else entry for entry in declared_pythonpath.split(os.pathsep)
        )
        if declared_pythonpath is not None
        else None
    )
    workspace = tmp_path / 'allocation-workspace'
    workspace.mkdir()
    environment = dict(os.environ)
    environment['PYTHONPATH'] = bootstrap._workspace_pythonpath(workspace, image_pythonpath)

    completed = subprocess.run(
        [
            sys.executable,
            '-S',
            '-c',
            'from tests.integration.distributed.cloud_tasks import MARKER; assert MARKER == "fixture-imported"',
        ],
        cwd=workspace,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_dynamic_devcloud_workload_build_uses_selected_workload_base_and_dockerfile():
    command = dynamic_cluster.test_workload_build_command(
        workload_image='workload:test',
        dockerfile=dynamic_cluster.TEST_WORKLOAD_DOCKERFILE,
        workload_base_image='workload-base:test',
    )

    assert command == [
        'docker',
        'build',
        '-f',
        str(dynamic_cluster.TEST_WORKLOAD_DOCKERFILE),
        '--build-arg',
        'NEXUS_WORKLOAD_BASE_IMAGE=workload-base:test',
        '-t',
        'workload:test',
        '.',
    ]


def test_dynamic_devcloud_builder_uses_test_workload_as_base():
    assert dynamic_cluster.builder_build_command(
        workload_image='workload:test',
        builder_image='builder:test',
    ) == [
        'docker',
        'build',
        '-f',
        'deployment/distributed/Dockerfile.builder',
        '--build-arg',
        'NEXUS_WORKLOAD_BASE_IMAGE=workload:test',
        '-t',
        'builder:test',
        '.',
    ]


async def test_distributed_sleep_echo_task_returns_index_and_message():
    result = await distributed_fixtures.distributed_sleep_echo_task(
        distributed_fixtures.DistributedSleepEchoInput(index=7, message='ready', sleep_seconds=0)
    )

    assert result.model_dump() == {'index': 7, 'message': 'ready'}
