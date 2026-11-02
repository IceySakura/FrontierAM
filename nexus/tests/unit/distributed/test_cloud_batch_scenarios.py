import json

from dataclasses import replace

import pytest

from tests.integration.distributed.cloud_batch_scenarios import (
    DEFAULT_WHEELHOUSE_EXTRAS,
    GIB,
    MAX_STRESS_CONTAINERS,
    STRESS_ENVIRONMENT_LIMITS,
    STRESS_STOP_POLICY,
    TKEX_LOAD_PROFILES,
    BatchLoadProfile,
    StressObservation,
    batch_submission,
    environment_demand,
    fit_profile_to_container_budget,
    prepare_source_project,
    source_generator_path,
    stress_stop_reasons,
    validate_profile_for_cluster,
    write_config_task_list,
)

from nexus.distributed.submission.specs import BatchSubmission, ConfigOnlyCode, SourcePackageCode
from nexus.runner.batch.loader import GeneratorLoader


def test_tkex_load_profiles_cover_approved_stages_in_order():
    assert list(TKEX_LOAD_PROFILES) == [
        'wheelhouse-smoke',
        'sandbox-agent-smoke',
        'wheelhouse-fanout',
        'config-baseline',
        'source-heavy',
        'container-scale',
        'container-ceiling',
        'manager-2000',
        'sandbox-agent-10000',
        'sandbox-agent-60000',
        'extreme',
    ]

    expected = {
        'wheelhouse-smoke': ('source_package', 20, 2, 2, 4, False),
        'sandbox-agent-smoke': ('config_only', 20, 5, 4, 20, False),
        'wheelhouse-fanout': ('source_package', 499, 499, 1, 499, False),
        'config-baseline': ('config_only', 2000, 25, 20, 500, False),
        'source-heavy': ('source_package', 2000, 25, 20, 500, False),
        'container-scale': ('config_only', 4000, 199, 10, 1990, False),
        'container-ceiling': ('config_only', 4000, 499, 4, 1996, False),
        'manager-2000': ('config_only', 16_000, 1999, 4, 7996, True),
        'sandbox-agent-10000': ('config_only', 16_000, 1999, 6, 10_100, True),
        'sandbox-agent-60000': ('config_only', 100_000, 3000, 20, 60_000, True),
        'extreme': ('config_only', 20000, 499, 20, 9000, True),
    }
    assert {
        name: (
            profile.code_mode,
            profile.item_count,
            profile.execution_manager_count,
            profile.slots_per_manager,
            profile.max_inflight,
            profile.opt_in,
        )
        for name, profile in TKEX_LOAD_PROFILES.items()
    } == expected


def test_manager_2000_profile_uses_exactly_2000_manager_pods():
    profile = TKEX_LOAD_PROFILES['manager-2000']

    assert MAX_STRESS_CONTAINERS == 3001
    assert profile.code_mode == 'config_only'
    assert profile.item_count == 16_000
    assert profile.manager_pod_count == 2000
    assert profile.execution_manager_count == 1999
    assert profile.slots_per_manager == 4
    assert profile.execution_slot_count == 7996
    assert profile.max_inflight == 7996
    assert profile.minimum_peak == 7200
    assert profile.task_sleep_seconds == 22
    assert profile.timeout_per_run_seconds == 2400
    assert profile.opt_in is True
    assert environment_demand(profile) == {
        'manager_count': 2000,
        'cpu_millicores': 2_003_000,
        'memory_bytes': 2003 * GIB,
        'slots': 7997,
        'active_batches': 1,
    }
    validate_profile_for_cluster(profile, containers_per_manager_pod=1)


def test_sandbox_agent_10000_profile_requires_10000_real_ready_sandboxes():
    profile = TKEX_LOAD_PROFILES['sandbox-agent-10000']

    assert profile.task_kind == 'sandbox_agent_echo'
    assert profile.item_count == 16_000
    assert profile.manager_pod_count == 2000
    assert profile.execution_slot_count == 11_994
    assert profile.max_inflight == 10_100
    assert profile.minimum_peak == 10_000
    assert profile.minimum_sandbox_peak == 10_000
    assert profile.retry_failed == 2
    assert profile.task_sleep_seconds == 1200
    assert profile.timeout_per_run_seconds == 3600
    assert profile.opt_in is True
    assert environment_demand(profile)['slots'] == 11_995
    validate_profile_for_cluster(profile, containers_per_manager_pod=1)


def test_sandbox_agent_60000_profile_uses_all_3000_manager_slot_capacity():
    profile = TKEX_LOAD_PROFILES['sandbox-agent-60000']

    assert profile.task_kind == 'sandbox_agent_echo'
    assert profile.item_count == 100_000
    assert profile.manager_pod_count == 3001
    assert profile.execution_manager_count == 3000
    assert profile.slots_per_manager == 20
    assert profile.execution_slot_count == 60_000
    assert profile.max_inflight == 60_000
    assert profile.minimum_peak == 60_000
    assert profile.minimum_sandbox_peak == 60_000
    assert profile.retry_failed == 2
    assert profile.task_sleep_seconds == 3600
    assert profile.timeout_per_run_seconds == 7200
    assert profile.stop_timeout_seconds == 3600
    assert profile.opt_in is True
    assert environment_demand(profile) == {
        'manager_count': 3001,
        'cpu_millicores': 3_004_000,
        'memory_bytes': 3004 * GIB,
        'slots': 60_001,
        'active_batches': 1,
    }
    validate_profile_for_cluster(profile, containers_per_manager_pod=1)


def test_tkex_profiles_use_one_cpu_one_gib_per_manager_and_respect_the_container_ceiling():
    for profile in TKEX_LOAD_PROFILES.values():
        expected_batch_manager_size = (
            4.0 if profile.name in {'manager-2000', 'sandbox-agent-10000', 'sandbox-agent-60000'} else 1.0
        )
        assert profile.batch_manager_cpu_cores == expected_batch_manager_size
        assert profile.batch_manager_memory_gib == expected_batch_manager_size
        assert profile.execution_manager_cpu_cores == 1.0
        assert profile.execution_manager_memory_gib == 1.0
        assert profile.manager_pod_count == 1 + profile.execution_manager_count
        assert profile.manager_container_count(containers_per_pod=1) <= MAX_STRESS_CONTAINERS
        assert profile.max_inflight <= profile.execution_slot_count

    assert TKEX_LOAD_PROFILES['container-ceiling'].manager_container_count(containers_per_pod=1) == 500
    assert TKEX_LOAD_PROFILES['extreme'].manager_container_count(containers_per_pod=1) == 500
    assert TKEX_LOAD_PROFILES['manager-2000'].manager_container_count(containers_per_pod=1) == 2000
    assert TKEX_LOAD_PROFILES['sandbox-agent-10000'].manager_container_count(containers_per_pod=1) == 2000
    assert TKEX_LOAD_PROFILES['sandbox-agent-60000'].manager_container_count(containers_per_pod=1) == 3001


def test_large_container_profiles_keep_tasks_running_long_enough_to_measure_steady_concurrency():
    assert TKEX_LOAD_PROFILES['wheelhouse-smoke'].task_sleep_seconds == 20
    assert TKEX_LOAD_PROFILES['container-scale'].task_sleep_seconds == 22
    assert TKEX_LOAD_PROFILES['container-ceiling'].task_sleep_seconds == 22
    assert TKEX_LOAD_PROFILES['manager-2000'].task_sleep_seconds == 22
    assert TKEX_LOAD_PROFILES['sandbox-agent-10000'].task_sleep_seconds == 1200
    assert TKEX_LOAD_PROFILES['sandbox-agent-60000'].task_sleep_seconds == 3600


def test_cluster_validation_counts_admission_sidecars_against_container_ceiling():
    validate_profile_for_cluster(TKEX_LOAD_PROFILES['manager-2000'], containers_per_manager_pod=1)

    with pytest.raises(ValueError, match='manager-2000.*4000.*3001'):
        validate_profile_for_cluster(TKEX_LOAD_PROFILES['manager-2000'], containers_per_manager_pod=2)
    with pytest.raises(ValueError, match='sandbox-agent-60000.*6002.*3001'):
        validate_profile_for_cluster(TKEX_LOAD_PROFILES['sandbox-agent-60000'], containers_per_manager_pod=2)


@pytest.mark.parametrize(
    ('name', 'execution_managers', 'max_inflight', 'minimum_peak', 'container_count'),
    [
        ('wheelhouse-fanout', 499, 499, 450, 1000),
        ('container-ceiling', 499, 1996, 1800, 1000),
        ('manager-2000', 1499, 5996, 5399, 3000),
        ('sandbox-agent-60000', 1499, 29_980, 29_980, 3000),
        ('extreme', 499, 9000, 8500, 1000),
    ],
)
def test_container_ceiling_profiles_fit_observed_sidecars(
    name: str,
    execution_managers: int,
    max_inflight: int,
    minimum_peak: int,
    container_count: int,
):
    fitted = fit_profile_to_container_budget(
        TKEX_LOAD_PROFILES[name],
        containers_per_manager_pod=2,
    )

    assert fitted.execution_manager_count == execution_managers
    assert fitted.max_inflight == max_inflight
    assert fitted.minimum_peak == minimum_peak
    assert fitted.item_count == TKEX_LOAD_PROFILES[name].item_count
    assert fitted.manager_container_count(containers_per_pod=2) == container_count
    validate_profile_for_cluster(fitted, containers_per_manager_pod=2)


def test_profiles_below_container_budget_are_unchanged():
    profile = TKEX_LOAD_PROFILES['config-baseline']

    assert fit_profile_to_container_budget(profile, containers_per_manager_pod=2) is profile


def test_environment_demand_accounts_for_parent_and_execution_managers():
    demand = environment_demand(TKEX_LOAD_PROFILES['container-ceiling'])

    assert demand == {
        'manager_count': 500,
        'cpu_millicores': 500_000,
        'memory_bytes': 500 * GIB,
        'slots': 1_997,
        'active_batches': 1,
    }
    assert STRESS_ENVIRONMENT_LIMITS == {
        'manager_count': 3001,
        'cpu_millicores': 3_010_000,
        'memory_bytes': 3010 * GIB,
        'slots': 60_001,
        'active_batches': 1,
        'max_items': 100_000,
    }


def test_cluster_validation_rejects_profile_above_environment_item_limit():
    profile = replace(TKEX_LOAD_PROFILES['sandbox-agent-60000'], item_count=100_001)

    with pytest.raises(ValueError, match='max_items.*100001'):
        validate_profile_for_cluster(profile, containers_per_manager_pod=1)


def test_stress_stop_policy_matches_approved_operational_guardrails():
    assert STRESS_STOP_POLICY.max_running_containers == 3001
    assert STRESS_STOP_POLICY.minimum_cluster_headroom_ratio == 0.10
    assert STRESS_STOP_POLICY.maximum_database_connection_ratio == 0.70
    assert STRESS_STOP_POLICY.maximum_unreachable_managers == 2
    assert STRESS_STOP_POLICY.no_progress_seconds == 120
    assert STRESS_STOP_POLICY.maximum_oom_or_restarts == 5
    assert STRESS_STOP_POLICY.maximum_result_upload_failure_ratio == 0.01


def test_stress_observation_reports_every_guardrail_breach():
    observation = StressObservation(
        running_manager_containers=3002,
        cluster_headroom_ratio=0.09,
        database_connection_ratio=0.71,
        unreachable_managers=3,
        no_progress_seconds=121,
        oom_or_restarts=6,
        result_upload_failure_ratio=0.02,
        sustained_gateway_5xx=True,
    )

    assert stress_stop_reasons(observation) == [
        'running_manager_containers=3002 exceeds 3001',
        'cluster_headroom_ratio=0.090 below 0.100',
        'database_connection_ratio=0.710 exceeds 0.700',
        'unreachable_managers=3 exceeds 2',
        'no_progress_seconds=121 exceeds 120',
        'oom_or_restarts=6 reached 5',
        'result_upload_failure_ratio=0.020 exceeds 0.010',
        'sustained_gateway_5xx=true',
    ]


def test_stress_observation_accepts_values_on_the_boundary():
    assert (
        stress_stop_reasons(
            StressObservation(
                running_manager_containers=3001,
                cluster_headroom_ratio=0.10,
                database_connection_ratio=0.70,
                unreachable_managers=2,
                no_progress_seconds=120,
                oom_or_restarts=4,
                result_upload_failure_ratio=0.01,
                sustained_gateway_5xx=False,
            )
        )
        == []
    )

    reasons = stress_stop_reasons(StressObservation(running_manager_containers=3001, oom_or_restarts=5))
    assert reasons == ['oom_or_restarts=5 reached 5']


def test_config_task_list_and_submission_are_profile_driven(tmp_path):
    profile = TKEX_LOAD_PROFILES['config-baseline']
    task_list = tmp_path / 'tasks.jsonl'
    manifest = write_config_task_list(task_list, profile=profile, run_id_prefix='tkex-e2e')
    ref = {
        'artifact_id': 'task-list',
        'kind': 'task_list',
        'sha256': manifest['sha256'],
        'size_bytes': manifest['size_bytes'],
    }

    payload = batch_submission(
        profile=profile,
        environment_id='tkex-stress',
        idempotency_key='config-baseline',
        task_list_ref=ref,
        task_list_manifest=manifest,
    )
    submission = BatchSubmission.model_validate(payload)

    assert isinstance(submission.code, ConfigOnlyCode)
    assert submission.resources.batch_manager.model_dump() == {'cpu_cores': 1.0, 'memory_gib': 1.0}
    assert submission.resources.managers.model_dump() == {
        'count': 25,
        'cpu_cores': 1.0,
        'memory_gib': 1.0,
        'slots': 20,
    }
    assert submission.batch.max_inflight == 500
    assert submission.batch.retry_failed == profile.retry_failed
    assert submission.generation.task_list.manifest.item_count == 2000

    records = [json.loads(line) for line in task_list.read_text().splitlines()]
    assert len(records) == 2000
    assert records[0]['item_id'] == 'item-000000000000'
    assert records[0]['run_input']['run_id'] == 'tkex-e2e-000000000000'
    assert records[0]['run_input']['imports'] == ['tests.integration.distributed.cloud_tasks']
    assert records[0]['run_input']['task_name'] == 'sleep_echo'
    assert records[0]['run_input']['task_config'] == {
        'index': 0,
        'message': 'config-baseline-000000000000',
        'sleep_seconds': 20,
    }
    assert records[-1]['item_id'] == 'item-000000001999'


def test_sandbox_agent_task_list_runs_main_agent_in_a_one_cpu_one_gib_tlinux_sandbox(tmp_path):
    profile = TKEX_LOAD_PROFILES['sandbox-agent-smoke']
    task_list = tmp_path / 'tasks.jsonl'

    manifest = write_config_task_list(task_list, profile=profile, run_id_prefix='sandbox-smoke')

    assert manifest['item_count'] == 20
    record = json.loads(task_list.read_text().splitlines()[7])
    run_input = record['run_input']
    assert run_input['imports'] == [
        'tests.integration.distributed.cloud_tasks',
        'nexus.extensions.agents.main',
        'nexus.extensions.agents.swe_agent.tools',
    ]
    assert run_input['task_name'] == 'sandbox_agent_echo'
    assert run_input['task_config'] == {
        'hold_seconds': 360,
        'index': 7,
        'marker': 'sandbox-agent-smoke-000000000007',
    }
    assert run_input['runtime_provider_name'] == 'gongfeng'
    assert run_input['runtime_provider_config'] == {
        'api_base_url': 'http://hyrl-sandbox.prod.woa.com:8052',
        'auto_connect_ws': False,
        'batch_id': 'sandbox-smoke',
        'print_port_mappings': False,
        'purpose': 'test',
        'queue_wait_max': 1800,
        'request_timeout': 1800,
        'resource_spec': {'cpu': 1, 'memory': 1},
        'rpc_protocol': 'http',
        'runtime_image': 'current',
        'runtime_provider': 'gongfeng',
        'startup_jitter_max': 30,
        'token__env': 'GONGFENG_TOKEN',
    }
    assert run_input['runtime_configs'] == {'main': {'image': 'mirrors.tencent.com/tlinux/tlinux4.0-minimal'}}
    assert run_input['agent_name'] == 'main_agent'
    assert run_input['agent_config'] == {
        'max_iterations': 3,
        'parallel_tool_calls': False,
        'tools': ['swe_agent/bash'],
    }
    assert run_input['openai_clients']['main']['config']['request_args'] == {'model': '${oc.env:NEXUS_MODEL,"gpt-5.5"}'}
    assert run_input['openai_clients']['main']['config']['client_args']['base_url'] == (
        '${oc.env:MODEL_SERVICE_API_BASE_URL}'
    )


def test_manager_2000_submission_allows_for_scheduler_queue_latency(tmp_path):
    profile = TKEX_LOAD_PROFILES['manager-2000']
    task_list = tmp_path / 'tasks.jsonl'
    manifest = write_config_task_list(task_list, profile=profile, run_id_prefix='manager-2000')
    ref = {
        'artifact_id': 'task-list',
        'kind': 'task_list',
        'sha256': manifest['sha256'],
        'size_bytes': manifest['size_bytes'],
    }

    payload = batch_submission(
        profile=profile,
        environment_id='tkex-stress',
        idempotency_key='manager-2000',
        task_list_ref=ref,
        task_list_manifest=manifest,
    )

    assert payload['batch']['timeout_per_run'] == 2400


def test_sandbox_10000_submission_preserves_the_approved_retry_budget(tmp_path):
    profile = TKEX_LOAD_PROFILES['sandbox-agent-10000']
    task_list = tmp_path / 'tasks.jsonl'
    manifest = write_config_task_list(task_list, profile=profile, run_id_prefix='sandbox-10000')
    ref = {
        'artifact_id': 'task-list',
        'kind': 'task_list',
        'sha256': manifest['sha256'],
        'size_bytes': manifest['size_bytes'],
    }

    payload = batch_submission(
        profile=profile,
        environment_id='tkex-stress',
        idempotency_key='sandbox-10000',
        task_list_ref=ref,
        task_list_manifest=manifest,
    )

    assert payload['batch']['max_inflight'] == 10_100
    assert payload['batch']['retry_failed'] == 2


def test_sandbox_60000_submission_uses_full_slot_capacity_and_extended_stop_timeout():
    profile = TKEX_LOAD_PROFILES['sandbox-agent-60000']
    manifest = {
        'version': 1,
        'item_count': profile.item_count,
        'size_bytes': 1,
        'sha256': 'a' * 64,
        'source': {'mode': 'cloud_batch_e2e', 'profile': profile.name},
        'datasets': [],
    }
    ref = {'artifact_id': 'task-list', 'kind': 'task_list', 'sha256': 'a' * 64, 'size_bytes': 1}

    payload = batch_submission(
        profile=profile,
        environment_id='tkex-stress',
        idempotency_key='sandbox-60000',
        task_list_ref=ref,
        task_list_manifest=manifest,
    )

    assert payload['resources']['managers'] == {
        'count': 3000,
        'cpu_cores': 1.0,
        'memory_gib': 1.0,
        'slots': 20,
    }
    assert payload['batch']['max_inflight'] == 60_000
    assert payload['batch']['retry_failed'] == 2
    assert payload['batch']['timeout_per_run'] == 7200
    assert payload['execution_ttl_seconds'] == 16_200
    assert payload['stop_timeout_seconds'] == 3600


def test_source_submission_reuses_inputs_and_selects_profile_generator():
    profile = TKEX_LOAD_PROFILES['source-heavy']
    source_ref = {'artifact_id': 'source', 'kind': 'source_package', 'sha256': 'a' * 64, 'size_bytes': 10}
    wheel_ref = {'artifact_id': 'wheel', 'kind': 'nexus_wheel', 'sha256': 'b' * 64, 'size_bytes': 20}
    lock_ref = {'artifact_id': 'lock', 'kind': 'uv_lock', 'sha256': 'c' * 64, 'size_bytes': 30}

    payload = batch_submission(
        profile=profile,
        environment_id='tkex-stress',
        idempotency_key='source-heavy',
        source_package_ref=source_ref,
        nexus_wheel_ref=wheel_ref,
        uv_lock_ref=lock_ref,
    )
    submission = BatchSubmission.model_validate(payload)

    assert isinstance(submission.code, SourcePackageCode)
    assert submission.code.extras == list(DEFAULT_WHEELHOUSE_EXTRAS)
    assert submission.generation.source.path == source_generator_path(profile)
    assert submission.resources.batch_manager.model_dump() == {'cpu_cores': 1.0, 'memory_gib': 1.0}
    assert submission.resources.managers.model_dump() == {
        'count': 25,
        'cpu_cores': 1.0,
        'memory_gib': 1.0,
        'slots': 20,
    }


def test_batch_submission_rejects_inputs_for_the_wrong_code_mode():
    config_profile = TKEX_LOAD_PROFILES['config-baseline']
    source_profile = TKEX_LOAD_PROFILES['source-heavy']

    with pytest.raises(ValueError, match='task list'):
        batch_submission(
            profile=config_profile,
            environment_id='tkex-stress',
            idempotency_key='missing-task-list',
        )
    with pytest.raises(ValueError, match='source package inputs'):
        batch_submission(
            profile=source_profile,
            environment_id='tkex-stress',
            idempotency_key='missing-source',
        )


def test_source_project_reuses_one_package_for_all_source_profiles(tmp_path):
    project_root = tmp_path / 'project'
    project_root.mkdir()
    (project_root / 'pyproject.toml').write_text('[project]\nname = "nexus-agent"\n')
    (project_root / 'uv.lock').write_text('version = 1\n')
    (project_root / 'README.md').write_text('fixture\n')
    source_root = tmp_path / 'source'

    prepare_source_project(source_root, project_root=project_root)

    assert (source_root / 'pyproject.toml').read_text() == (project_root / 'pyproject.toml').read_text()
    assert (source_root / 'uv.lock').read_text() == (project_root / 'uv.lock').read_text()
    fixture = (source_root / 'cloud_source_fixture.py').read_text()
    compile(fixture, 'cloud_source_fixture.py', 'exec')
    assert 'importlib.metadata.version("mycli")' in fixture
    assert 'sys.prefix' in fixture
    assert 'nexus.__file__' in fixture

    generated = {
        profile.name: source_root / source_generator_path(profile)
        for profile in TKEX_LOAD_PROFILES.values()
        if profile.code_mode == 'source_package'
    }
    assert set(generated) == {'wheelhouse-smoke', 'wheelhouse-fanout', 'source-heavy'}
    for path in generated.values():
        compile(path.read_text(), str(path), 'exec')

    smoke_items = list(GeneratorLoader(generated['wheelhouse-smoke']).generate())
    assert len(smoke_items) == 20
    assert smoke_items[0].run_id == 'wheelhouse-smoke-000000000000'
    assert smoke_items[-1].run_id == 'wheelhouse-smoke-000000000019'
    assert smoke_items[0].config.imports == ['cloud_source_fixture']
    assert smoke_items[0].config.task_config == {'index': 0, 'sleep_seconds': 20}
    assert smoke_items[0].metadata == {}


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'execution_manager_count': 3001}, 'manager pod count'),
        ({'max_inflight': 21, 'execution_manager_count': 2, 'slots_per_manager': 10}, 'max_inflight'),
        ({'minimum_peak': 21, 'max_inflight': 20}, 'minimum_peak'),
        ({'item_count': 19, 'max_inflight': 20}, 'item_count'),
        ({'timeout_per_run_seconds': 79}, 'timeout_per_run_seconds'),
        ({'batch_manager_cpu_cores': 0}, 'batch manager resources'),
        ({'code_mode': 'unknown'}, 'code_mode'),
    ],
)
def test_load_profile_rejects_invalid_resource_and_concurrency_shapes(overrides, message):
    values = {
        'name': 'invalid',
        'code_mode': 'config_only',
        'item_count': 20,
        'execution_manager_count': 2,
        'slots_per_manager': 10,
        'max_inflight': 20,
        'minimum_peak': 18,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        BatchLoadProfile(**values)
