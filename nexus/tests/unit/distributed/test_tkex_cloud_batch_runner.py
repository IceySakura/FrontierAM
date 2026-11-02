import argparse
import asyncio
import json

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.e2e.distributed import run_tkex_cloud_batch as tkex_runner
from tests.integration.distributed.cloud_batch_scenarios import BatchLoadProfile


def test_tkex_observer_reads_api_token_file_before_creating_client(monkeypatch, tmp_path):
    token_file = tmp_path / 'tkex-token'
    token_file.write_text('tkex-secret', encoding='utf-8')
    token_file.chmod(0o600)
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(tkex_runner, 'TkexWorkloadClient', FakeClient)

    tkex_runner.TkexObserver(
        tkex_runner.TkexObserverConfig(
            base_url='https://tkex.example',
            namespace='nexus-test',
            api_account='account',
            api_token_file=token_file,
            cluster_name='cluster',
            project_name='project',
        )
    )

    assert captured['api_token'] == 'tkex-secret'
    assert 'api_token_file' not in captured


def test_tkex_runner_defaults_to_plan_only_and_requires_explicit_profiles(tmp_path):
    token_file = tmp_path / 'submit-token'
    token_file.write_text('secret')

    args = tkex_runner.parse_args(
        [
            '--control-url',
            'https://control.example',
            '--application-url',
            'https://application.example',
            '--environment-id',
            'tkex-stress',
            '--taihu-token-file',
            str(token_file),
            '--profile',
            'wheelhouse-smoke',
        ]
    )

    assert args.execute is False
    assert args.profile == ['wheelhouse-smoke']
    assert args.typed_smoke is False
    assert args.containers_per_manager_pod == 1
    assert args.control_database_metrics is False
    assert args.gongfeng_api_base_url == 'http://hyrl-sandbox.prod.woa.com:8052'
    assert args.gongfeng_token_file is None
    assert args.manager_crash == 'none'
    assert args.execution_manager_crash_count == 20
    assert args.confirm_manager_pod_delete is False


def test_tkex_runner_builds_typed_single_and_live_source_submissions():
    refs = {
        kind: {
            'artifact_id': f'artifact-{kind}',
            'kind': kind,
            'sha256': str(index) * 64,
            'size_bytes': index,
        }
        for index, kind in enumerate(('source_package', 'nexus_wheel', 'uv_lock'), start=1)
    }

    single = tkex_runner.typed_source_submission(
        environment_id='tkex-dynamic',
        source_refs=refs,
        run_input=tkex_runner.typed_single_run_input(),
        live=False,
    )
    live = tkex_runner.typed_source_submission(
        environment_id='tkex-dynamic',
        source_refs=refs,
        run_input=tkex_runner.typed_live_run_input(),
        live=True,
    )

    assert single['environment_id'] == live['environment_id'] == 'tkex-dynamic'
    assert single['code'] == live['code']
    assert single['code']['extras'] == ['distributed-db-shell']
    assert single['resources']['managers'] == {
        'count': 1,
        'cpu_cores': 1.0,
        'memory_gib': 1.0,
        'slots': 1,
    }
    assert single['run_input']['task_name'] == 'cloud_source_package_echo'
    assert 'live' not in single
    assert live['run_input']['agent_name'] == 'cloud_interactive_agent'
    assert live['live'] == {'working_dir': '/workspace'}


def test_dependency_build_ids_group_base_and_augmented_builds_by_workload():
    allocations = [
        {
            'workload': 'orchestration',
            'dependency_build_id': 'build-base',
            'dependency_build': {'dependency_build_id': 'build-base'},
        },
        {
            'workload': 'execution',
            'dependency_build_id': 'build-augmented',
            'dependency_build': {'dependency_build_id': 'build-augmented'},
        },
    ]

    assert tkex_runner.dependency_build_ids_by_workload(allocations) == {
        'execution': ['build-augmented'],
        'orchestration': ['build-base'],
    }


def test_dependency_build_ids_fall_back_to_nested_snapshot_and_keep_missing_workloads_visible():
    allocations = [
        {
            'workload': 'orchestration',
            'dependency_build': {'dependency_build_id': 'build-base'},
        },
        {'workload': 'execution', 'dependency_build': None},
    ]

    assert tkex_runner.dependency_build_ids_by_workload(allocations) == {
        'execution': [],
        'orchestration': ['build-base'],
    }


def test_validate_source_dependency_builds_allows_one_build_per_workload():
    allocations = [
        {'workload': 'orchestration', 'dependency_build_id': 'build-base'},
        {'workload': 'execution', 'dependency_build_id': 'build-augmented'},
    ]

    assert tkex_runner.validate_source_dependency_builds('wheelhouse-smoke', allocations) == {
        'execution': ['build-augmented'],
        'orchestration': ['build-base'],
    }


def test_validate_source_dependency_builds_rejects_multiple_builds_for_one_workload():
    allocations = [
        {'workload': 'orchestration', 'dependency_build_id': 'build-base'},
        {'workload': 'execution', 'dependency_build_id': 'build-a'},
        {'workload': 'execution', 'dependency_build_id': 'build-b'},
    ]

    with pytest.raises(AssertionError, match='one DependencyBuild per workload'):
        tkex_runner.validate_source_dependency_builds('wheelhouse-smoke', allocations)


async def test_tkex_typed_smoke_runs_single_live_input_and_release(monkeypatch):
    released = []

    async def wait_released(_api, _observer, *, execution_id, timeout):
        assert timeout == 900
        released.append(execution_id)

    monkeypatch.setattr(tkex_runner, '_wait_stage_released', wait_released)

    class FakeApi:
        def __init__(self):
            self.live_completed = False

        async def control_json(self, method, path, *, body=None):
            if (method, path) == ('POST', '/v1/submissions/single'):
                assert body['run_input']['task_name'] == 'cloud_source_package_echo'
                return {'execution_id': 'typed-single'}
            if (method, path) == ('POST', '/v1/submissions/live'):
                assert body['run_input']['agent_name'] == 'cloud_interactive_agent'
                return {'execution_id': 'typed-live'}
            if path == '/v1/executions/typed-single':
                return {
                    'execution_id': 'typed-single',
                    'status': 'succeeded',
                    'result_artifact_id': 'result-single',
                    'dependency_build_id': 'build-1',
                }
            if path == '/v1/executions/typed-live':
                return {
                    'execution_id': 'typed-live',
                    'status': 'succeeded' if self.live_completed else 'waiting_input',
                    'result_artifact_id': 'result-live',
                    'dependency_build_id': 'build-1',
                }
            raise AssertionError((method, path, body))

        async def resolved_result(self, execution):
            if execution['execution_id'] == 'typed-single':
                return {
                    'task_output': {
                        'marker': 'source-package-ok:900001',
                        'dependency_version': '1.0',
                        'sys_prefix': '/data/manager-venvs/build-1',
                    }
                }
            return {'agent_output': {'choice': 'tkex-typed-live-ok'}}

        async def read_application_sse_until(self, path, event_type, *, timeout):
            assert path == '/v1/executions/typed-live/live/events?cursor=0'
            assert event_type == 'input.required'
            assert timeout == 300
            return {'type': 'input.required'}

        async def application_json(self, method, path, *, body=None):
            assert (method, path) == ('POST', '/v1/executions/typed-live/commands')
            assert body['payload'] == {'text': 'tkex-typed-live-ok'}
            self.live_completed = True
            return {'accepted': True}

    refs = {
        kind: {'artifact_id': kind, 'kind': kind, 'sha256': str(index) * 64, 'size_bytes': index}
        for index, kind in enumerate(('source_package', 'nexus_wheel', 'uv_lock'), start=1)
    }
    report = await tkex_runner._run_typed_smoke(
        FakeApi(),
        object(),
        environment_id='tkex-dynamic',
        source_refs=refs,
    )

    assert report['status'] == 'passed'
    assert report['single']['marker'] == 'source-package-ok:900001'
    assert report['live']['choice'] == 'tkex-typed-live-ok'
    assert released == ['typed-single', 'typed-live']


def test_tkex_runner_parses_explicit_manager_crash_authorization(tmp_path):
    token_file = tmp_path / 'submit-token'
    token_file.write_text('secret')

    args = tkex_runner.parse_args(
        [
            '--control-url',
            'https://control.example',
            '--application-url',
            'https://application.example',
            '--environment-id',
            'tkex-stress',
            '--taihu-token-file',
            str(token_file),
            '--profile',
            'sandbox-agent-10000',
            '--manager-crash',
            'execution',
            '--execution-manager-crash-count',
            '12',
            '--confirm-manager-pod-delete',
        ]
    )

    assert args.manager_crash == 'execution'
    assert args.execution_manager_crash_count == 12
    assert args.confirm_manager_pod_delete is True


def test_artifact_failure_ratio_uses_only_current_stage_uploads():
    baseline = {'completed_uploads': 10, 'failed_uploads': 2}

    assert tkex_runner.result_upload_failure_ratio(
        baseline,
        {'completed_uploads': 108, 'failed_uploads': 4},
    ) == pytest.approx(0.02)
    assert (
        tkex_runner.result_upload_failure_ratio(
            baseline,
            {'completed_uploads': 10, 'failed_uploads': 3},
        )
        == 1
    )
    assert (
        tkex_runner.result_upload_failure_ratio(
            baseline,
            {'completed_uploads': 9, 'failed_uploads': 1},
        )
        == 0
    )


def test_sandbox_profile_no_progress_budget_covers_agent_and_hold_time():
    assert (
        tkex_runner._stress_policy_for_profile(tkex_runner.TKEX_LOAD_PROFILES['config-baseline']).no_progress_seconds
        == 120
    )
    assert (
        tkex_runner._stress_policy_for_profile(
            tkex_runner.TKEX_LOAD_PROFILES['sandbox-agent-10000']
        ).no_progress_seconds
        == 1500
    )
    assert (
        tkex_runner._stress_policy_for_profile(
            tkex_runner.TKEX_LOAD_PROFILES['sandbox-agent-60000']
        ).no_progress_seconds
        == 3900
    )


def test_stage_progress_fingerprint_counts_completed_artifact_uploads():
    execution = {'status': 'running'}
    summary = {
        'generation_status': 'complete',
        'materialized_count': 4000,
        'submitted_items': 1990,
        'terminal_items': 0,
    }

    before = tkex_runner._stage_progress_fingerprint(execution, summary, completed_uploads=100)
    after = tkex_runner._stage_progress_fingerprint(execution, summary, completed_uploads=101)

    assert before != after


def test_no_progress_timer_starts_when_runtime_monitor_activates():
    elapsed, last_progress_at = tkex_runner._monitor_no_progress_seconds(
        monitor_active=True,
        monitor_was_active=False,
        last_progress_at=0,
        now=559,
    )

    assert elapsed == 0
    assert last_progress_at == 559
    assert tkex_runner._monitor_no_progress_seconds(
        monitor_active=True,
        monitor_was_active=True,
        last_progress_at=last_progress_at,
        now=680,
    ) == (121, 559)


async def test_concurrency_sampler_stop_cancels_a_blocked_sample():
    request_started = asyncio.Event()
    blocked = asyncio.Event()

    class FakeApi:
        async def control_json(self, method, path):
            assert method == 'GET'
            assert path == '/v1/executions/parent/batch?include_status_counts=true'
            request_started.set()
            await blocked.wait()
            raise AssertionError('blocked request should have been cancelled')

    sampler = tkex_runner._BatchConcurrencySampler(
        FakeApi(),
        execution_id='parent',
        stage={'peak_active_items': 0, 'peak_running_items': 0, 'concurrency_observations': []},
        minimum_peak=1,
        interval=1,
    )
    sampler.start()
    await asyncio.wait_for(request_started.wait(), timeout=1)

    await asyncio.wait_for(sampler.stop(), timeout=0.1)


async def test_prepare_source_inputs_uses_a_fresh_upload_scope_per_run(monkeypatch, tmp_path):
    packaged = {
        'source_package': SimpleNamespace(path=tmp_path / 'source.tar.zst', sha256='1' * 64, size_bytes=1),
        'nexus_wheel': SimpleNamespace(path=tmp_path / 'nexus.whl', sha256='2' * 64, size_bytes=2),
        'uv_lock': SimpleNamespace(path=tmp_path / 'uv.lock', sha256='3' * 64, size_bytes=3),
    }
    monkeypatch.setattr(tkex_runner, 'prepare_source_project', lambda *args, **kwargs: None)
    monkeypatch.setattr(tkex_runner, 'build_source_package', lambda *args, **kwargs: packaged['source_package'])
    monkeypatch.setattr(tkex_runner, '_build_wheel', lambda *args, **kwargs: packaged['nexus_wheel'])
    monkeypatch.setattr(tkex_runner, '_prepare_uv_lock', lambda *args, **kwargs: packaged['uv_lock'])

    class FakeApi:
        def __init__(self):
            self.keys = []

        async def upload_artifact(self, path, *, kind, sha256, size_bytes, idempotency_key, environment_id):
            self.keys.append(idempotency_key)
            return {
                'artifact_id': f'artifact-{kind}',
                'kind': kind,
                'sha256': sha256,
                'size_bytes': size_bytes,
            }

    first = FakeApi()
    second = FakeApi()
    await tkex_runner._prepare_source_inputs(
        first,
        project_root=tmp_path,
        temporary=tmp_path / 'first',
        environment_id='tkex-stress',
    )
    await tkex_runner._prepare_source_inputs(
        second,
        project_root=tmp_path,
        temporary=tmp_path / 'second',
        environment_id='tkex-stress',
    )

    assert set(first.keys).isdisjoint(second.keys)
    assert all(len(key) <= 128 for key in first.keys + second.keys)
    assert len(set(first.keys)) == len(set(second.keys)) == 3


def test_selected_profiles_preserve_approved_order_and_reject_extreme_without_opt_in():
    selected = tkex_runner.selected_profiles(
        ['container-ceiling', 'wheelhouse-smoke', 'config-baseline'],
        allow_extreme=False,
    )
    assert [profile.name for profile in selected] == [
        'wheelhouse-smoke',
        'config-baseline',
        'container-ceiling',
    ]

    with pytest.raises(ValueError, match='opt-in.*allow-extreme'):
        tkex_runner.selected_profiles(['extreme'], allow_extreme=False)
    assert [profile.name for profile in tkex_runner.selected_profiles(['extreme'], allow_extreme=True)] == ['extreme']
    with pytest.raises(ValueError, match='opt-in.*allow-extreme'):
        tkex_runner.selected_profiles(['manager-2000'], allow_extreme=False)


def test_build_execution_plan_is_sequential_and_enforces_headroom():
    profiles = tkex_runner.selected_profiles(
        ['wheelhouse-smoke', 'container-ceiling'],
        allow_extreme=False,
    )
    plan = tkex_runner.build_execution_plan(
        profiles,
        containers_per_manager_pod=1,
        available_cpu_cores=560,
        available_memory_gib=560,
    )

    assert plan['execution_mode'] == 'sequential'
    assert plan['maximum_running_manager_containers'] == 500
    assert plan['maximum_requested_cpu_cores'] == 500
    assert plan['maximum_requested_memory_gib'] == 500
    assert [stage['name'] for stage in plan['stages']] == ['wheelhouse-smoke', 'container-ceiling']
    assert plan['stages'][-1]['resources'] == {
        'manager_pods': 500,
        'running_manager_containers': 500,
        'cpu_cores': 500,
        'memory_gib': 500,
        'execution_slots': 1996,
        'environment_slots': 1997,
    }

    with pytest.raises(ValueError, match='CPU headroom'):
        tkex_runner.build_execution_plan(
            profiles,
            containers_per_manager_pod=1,
            available_cpu_cores=549,
            available_memory_gib=560,
        )
    with pytest.raises(ValueError, match='memory headroom'):
        tkex_runner.build_execution_plan(
            profiles,
            containers_per_manager_pod=1,
            available_cpu_cores=560,
            available_memory_gib=549,
        )


def test_build_execution_plan_supports_opt_in_2000_manager_stage():
    profiles = tkex_runner.selected_profiles(
        ['wheelhouse-smoke', 'manager-2000'],
        allow_extreme=True,
    )
    plan = tkex_runner.build_execution_plan(
        profiles,
        containers_per_manager_pod=1,
        available_cpu_cores=2210,
        available_memory_gib=2210,
    )

    assert [stage['name'] for stage in plan['stages']] == ['wheelhouse-smoke', 'manager-2000']
    assert plan['maximum_running_manager_containers'] == 2000
    assert plan['maximum_requested_cpu_cores'] == 2003
    assert plan['maximum_requested_memory_gib'] == 2003
    assert plan['minimum_available_cpu_cores'] == pytest.approx(2203.3)
    assert plan['minimum_available_memory_gib'] == pytest.approx(2203.3)
    assert plan['stages'][-1]['resources'] == {
        'manager_pods': 2000,
        'running_manager_containers': 2000,
        'cpu_cores': 2003,
        'memory_gib': 2003,
        'execution_slots': 7996,
        'environment_slots': 7997,
    }


def test_build_execution_plan_supports_opt_in_3000_manager_60000_sandbox_stage():
    profiles = tkex_runner.selected_profiles(['sandbox-agent-60000'], allow_extreme=True)
    plan = tkex_runner.build_execution_plan(
        profiles,
        containers_per_manager_pod=1,
        available_cpu_cores=3310,
        available_memory_gib=3310,
    )

    assert plan['maximum_running_manager_containers'] == 3001
    assert plan['maximum_requested_cpu_cores'] == 3004
    assert plan['maximum_requested_memory_gib'] == 3004
    assert plan['minimum_available_cpu_cores'] == pytest.approx(3304.4)
    assert plan['minimum_available_memory_gib'] == pytest.approx(3304.4)
    assert plan['stages'][0]['resources'] == {
        'manager_pods': 3001,
        'running_manager_containers': 3001,
        'cpu_cores': 3004,
        'memory_gib': 3004,
        'execution_slots': 60_000,
        'environment_slots': 60_001,
    }


def test_build_execution_plan_fits_sidecar_expansion_before_execute():
    profiles = tkex_runner.fit_profiles_to_container_budget(
        tkex_runner.selected_profiles(['sandbox-agent-60000'], allow_extreme=True),
        containers_per_manager_pod=2,
    )

    plan = tkex_runner.build_execution_plan(
        profiles,
        containers_per_manager_pod=2,
        available_cpu_cores=2210,
        available_memory_gib=2210,
    )

    assert plan['maximum_running_manager_containers'] == 3000
    assert plan['maximum_requested_cpu_cores'] == 1503
    assert plan['stages'][0]['resources'] == {
        'manager_pods': 1500,
        'running_manager_containers': 3000,
        'cpu_cores': 1503,
        'memory_gib': 1503,
        'execution_slots': 29_980,
        'environment_slots': 29_981,
    }


def test_summarize_tkex_pods_counts_actual_containers_pod_ips_and_restarts():
    pods = {
        'items': [
            {
                'metadata': {'name': 'execution-0', 'annotations': {'nexus.io/allocation-id': 'execution'}},
                'spec': {'containers': [{'name': 'manager'}, {'name': 'injected-sidecar'}]},
                'status': {
                    'phase': 'Running',
                    'podIP': '10.0.0.10',
                    'conditions': [{'type': 'Ready', 'status': 'True'}],
                    'containerStatuses': [
                        {
                            'name': 'manager',
                            'restartCount': 2,
                            'lastState': {'terminated': {'reason': 'OOMKilled'}},
                        },
                        {'name': 'injected-sidecar', 'restartCount': 0},
                    ],
                },
            },
            {
                'metadata': {'name': 'execution-1', 'annotations': {'nexus.io/allocation-id': 'execution'}},
                'spec': {'containers': [{'name': 'manager'}, {'name': 'injected-sidecar'}]},
                'status': {'phase': 'Pending', 'containerStatuses': []},
            },
            {
                'metadata': {'name': 'orchestration-0', 'annotations': {'nexus.io/allocation-id': 'orchestration'}},
                'spec': {'containers': [{'name': 'manager'}]},
                'status': {
                    'phase': 'Running',
                    'conditions': [{'type': 'Ready', 'status': 'True'}],
                    'containerStatuses': [{'name': 'manager', 'restartCount': 0}],
                },
            },
            {
                'metadata': {'name': 'old', 'annotations': {'nexus.io/allocation-id': 'other'}},
                'spec': {'containers': [{'name': 'manager'}]},
                'status': {'phase': 'Succeeded'},
            },
        ]
    }

    summary = tkex_runner.summarize_tkex_pods(pods, allocation_ids={'execution', 'orchestration'})

    assert summary == {
        'pod_count': 3,
        'active_pod_count': 3,
        'running_manager_containers': 5,
        'maximum_containers_per_pod': 2,
        'ready_pod_count': 2,
        'ready_pods_missing_ip': 1,
        'pod_ip_count': 1,
        'pod_ip_sample': ['10.0.0.10'],
        'restart_count': 2,
        'oom_killed_container_count': 1,
    }


def test_select_manager_crash_targets_is_allocation_scoped_and_requires_loaded_execution_managers():
    pods = {
        'items': [
            {
                'metadata': {
                    'name': 'nexus-execution-0',
                    'uid': 'uid-0',
                    'annotations': {'nexus.io/allocation-id': 'alc-execution'},
                    'labels': {'nexus.io/workload': 'execution'},
                },
                'status': {
                    'phase': 'Running',
                    'conditions': [{'type': 'Ready', 'status': 'True'}],
                },
            },
            {
                'metadata': {
                    'name': 'nexus-execution-1',
                    'uid': 'uid-1',
                    'annotations': {'nexus.io/allocation-id': 'alc-execution'},
                    'labels': {'nexus.io/workload': 'execution'},
                },
                'status': {
                    'phase': 'Running',
                    'conditions': [{'type': 'Ready', 'status': 'True'}],
                },
            },
            {
                'metadata': {
                    'name': 'nexus-other-0',
                    'uid': 'uid-other',
                    'annotations': {'nexus.io/allocation-id': 'alc-other'},
                    'labels': {'nexus.io/workload': 'execution'},
                },
                'status': {
                    'phase': 'Running',
                    'conditions': [{'type': 'Ready', 'status': 'True'}],
                },
            },
        ]
    }
    managers = {
        'items': [
            {
                'manager_id': 'alc-execution-0',
                'session_id': 'session-0',
                'healthy': True,
                'reserved_slots': 6,
            },
            {
                'manager_id': 'alc-execution-1',
                'session_id': 'session-1',
                'healthy': True,
                'reserved_slots': 5,
            },
            {
                'manager_id': 'alc-other-0',
                'session_id': 'session-other',
                'healthy': True,
                'reserved_slots': 6,
            },
        ]
    }

    targets = tkex_runner.select_manager_crash_targets(
        pods,
        managers,
        allocation_id='alc-execution',
        workload='execution',
        count=1,
        required_reserved_slots=6,
    )

    assert [target.as_report() for target in targets] == [
        {
            'manager_id': 'alc-execution-0',
            'pod_name': 'nexus-execution-0',
            'previous_pod_uid': 'uid-0',
            'previous_session_id': 'session-0',
            'reserved_slots': 6,
        }
    ]
    with pytest.raises(ValueError, match='requested 2.*found 1'):
        tkex_runner.select_manager_crash_targets(
            pods,
            managers,
            allocation_id='alc-execution',
            workload='execution',
            count=2,
            required_reserved_slots=6,
        )


def test_manager_crash_recovery_requires_new_pod_uid_and_new_manager_session():
    target = tkex_runner.ManagerCrashTarget(
        manager_id='alc-execution-0',
        pod_name='nexus-execution-0',
        previous_pod_uid='uid-old',
        previous_session_id='session-old',
        reserved_slots=6,
    )
    pods = {
        'items': [
            {
                'metadata': {'name': 'nexus-execution-0', 'uid': 'uid-new'},
                'status': {
                    'phase': 'Running',
                    'conditions': [{'type': 'Ready', 'status': 'True'}],
                },
            }
        ]
    }
    managers = {
        'items': [
            {
                'manager_id': 'alc-execution-0',
                'session_id': 'session-new',
                'healthy': True,
            }
        ]
    }

    recovered, pending = tkex_runner.manager_crash_recovery([target], pods, managers)

    assert pending == []
    assert recovered == [
        {
            'manager_id': 'alc-execution-0',
            'pod_name': 'nexus-execution-0',
            'replacement_pod_uid': 'uid-new',
            'replacement_session_id': 'session-new',
        }
    ]
    managers['items'][0]['session_id'] = 'session-old'
    assert tkex_runner.manager_crash_recovery([target], pods, managers) == ([], ['alc-execution-0'])


def test_manager_crash_trigger_and_guardrail_are_role_specific():
    profile = tkex_runner.TKEX_LOAD_PROFILES['sandbox-agent-10000']
    summary = {
        'generation_status': 'complete',
        'submitted_items': 10_100,
        'total_items': 16_000,
    }

    assert tkex_runner.manager_crash_trigger_ready(
        'batch',
        profile=profile,
        summary=summary,
        peak_running_items=10_000,
        live_ready_sandboxes=None,
    )
    assert not tkex_runner.manager_crash_trigger_ready(
        'execution',
        profile=profile,
        summary=summary,
        peak_running_items=10_100,
        live_ready_sandboxes=9_999,
    )
    assert tkex_runner.manager_crash_trigger_ready(
        'execution',
        profile=profile,
        summary=summary,
        peak_running_items=10_100,
        live_ready_sandboxes=10_000,
    )
    assert tkex_runner.unplanned_unreachable_managers(22, planned_pending=20) == 2
    assert tkex_runner.unplanned_unreachable_managers(22, planned_pending=0) == 22


def test_tkex_observation_rejects_unexpected_workload_images():
    assert tkex_runner._tkex_observation_stop_reasons(
        {
            'unexpected_workload_images': ['registry.example/workload@sha256:' + '1' * 64],
            'services_present': [],
            'workload_count': 2,
            'pod_ip_downward_api': True,
        }
    ) == ['unexpected workload images: registry.example/workload@sha256:' + '1' * 64]


async def test_tkex_observer_retries_transient_read_failure(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    class FakeClient:
        def __init__(self):
            self.workload_reads = 0

        async def get_stateful_workload(self, _name):
            self.workload_reads += 1
            if self.workload_reads == 1:
                raise RuntimeError("tkex_api_502: {'message': 'destroyed socket allocated'}")
            return {
                'spec': {
                    'replicas': 1,
                    'template': {
                        'spec': {
                            'containers': [
                                {
                                    'image': 'registry.example/workload@sha256:' + '1' * 64,
                                    'env': [
                                        {
                                            'name': 'POD_IP',
                                            'valueFrom': {'fieldRef': {'fieldPath': 'status.podIP'}},
                                        },
                                        {'name': 'NEXUS_DIST_MANAGER_WORKLOAD', 'value': 'execution'},
                                        {'name': 'NEXUS_DIST_DB_POOL_SIZE', 'value': '1'},
                                        {'name': 'NEXUS_DIST_DB_MAX_OVERFLOW', 'value': '0'},
                                    ],
                                }
                            ]
                        }
                    },
                },
                'status': {'readyReplicas': 1},
            }

        async def get_service(self, name):
            raise tkex_runner.WorkloadNotFoundError(name)

        async def list_pods(self):
            return {'items': []}

    monkeypatch.setattr(tkex_runner.asyncio, 'sleep', fake_sleep)
    observer = object.__new__(tkex_runner.TkexObserver)
    observer.client = FakeClient()
    observer.expected_workload_image = 'registry.example/workload@sha256:' + '2' * 64

    observation = await observer.observe([{'allocation_id': 'allocation-a'}])

    assert observer.client.workload_reads == 2
    assert sleeps == [0.5]
    assert observation['workload_count'] == 1
    assert observation['desired_replicas'] == 1
    assert observation['ready_replicas'] == 1
    assert observation['workload_images'] == ['registry.example/workload@sha256:' + '1' * 64]
    assert observation['unexpected_workload_images'] == ['registry.example/workload@sha256:' + '1' * 64]
    assert observation['db_pool_settings'] == {'execution': {'pool_size': '1', 'max_overflow': '0'}}


async def test_tkex_observer_does_not_retry_permanent_read_failure(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    class FakeClient:
        def __init__(self):
            self.workload_reads = 0

        async def get_stateful_workload(self, _name):
            self.workload_reads += 1
            raise RuntimeError("tkex_api_400: {'message': 'bad request'}")

    monkeypatch.setattr(tkex_runner.asyncio, 'sleep', fake_sleep)
    observer = object.__new__(tkex_runner.TkexObserver)
    observer.client = FakeClient()

    with pytest.raises(RuntimeError, match='tkex_api_400'):
        await observer.observe([{'allocation_id': 'allocation-a'}])

    assert observer.client.workload_reads == 1
    assert sleeps == []


async def test_tkex_observer_reads_allocations_with_bounded_concurrency():
    class FakeClient:
        def __init__(self):
            self.active_reads = 0
            self.maximum_active_reads = 0

        async def get_stateful_workload(self, _name):
            self.active_reads += 1
            self.maximum_active_reads = max(self.maximum_active_reads, self.active_reads)
            await asyncio.sleep(0.01)
            self.active_reads -= 1
            return {
                'spec': {
                    'replicas': 1,
                    'template': {
                        'spec': {
                            'containers': [
                                {
                                    'image': 'registry.example/workload@sha256:' + '1' * 64,
                                    'env': [
                                        {
                                            'name': 'POD_IP',
                                            'valueFrom': {'fieldRef': {'fieldPath': 'status.podIP'}},
                                        }
                                    ],
                                }
                            ]
                        }
                    },
                },
                'status': {'readyReplicas': 1},
            }

        async def get_service(self, name):
            raise tkex_runner.WorkloadNotFoundError(name)

        async def list_pods(self):
            return {'items': []}

    observer = object.__new__(tkex_runner.TkexObserver)
    observer.client = FakeClient()
    observer.expected_workload_image = 'registry.example/workload@sha256:' + '1' * 64
    observer.read_concurrency = 3

    observation = await observer.observe([{'allocation_id': f'allocation-{index}'} for index in range(8)])

    assert observation['workload_count'] == 8
    assert 1 < observer.client.maximum_active_reads <= 3


def test_validate_completed_batch_requires_unique_items_children_and_results():
    profile = BatchLoadProfile(
        name='tiny',
        code_mode='config_only',
        item_count=2,
        execution_manager_count=2,
        slots_per_manager=1,
        max_inflight=2,
        minimum_peak=2,
        task_sleep_seconds=0,
    )
    summary = {
        'summary': {'total': 2, 'succeeded': 2, 'failed': 0, 'killed': 0, 'lost': 0, 'expired': 0, 'cancelled': 0}
    }
    items = [
        {'item_id': 'item-1', 'status': 'succeeded', 'current_child_execution_id': 'child-1'},
        {'item_id': 'item-2', 'status': 'succeeded', 'current_child_execution_id': 'child-2'},
    ]
    facts = {
        'child-1': {'execution_id': 'child-1', 'status': 'succeeded', 'result_artifact_id': 'result-1'},
        'child-2': {'execution_id': 'child-2', 'status': 'succeeded', 'result_artifact_id': 'result-2'},
    }

    assert tkex_runner.validate_completed_batch(profile, summary=summary, items=items, facts=facts) == {
        'item_count': 2,
        'child_execution_count': 2,
        'unique_result_artifact_count': 2,
    }

    facts['child-2']['result_artifact_id'] = 'result-1'
    with pytest.raises(AssertionError, match='result artifacts are not unique'):
        tkex_runner.validate_completed_batch(profile, summary=summary, items=items, facts=facts)


def test_validate_source_outputs_proves_offline_dependency_and_manager_venv():
    profile = BatchLoadProfile(
        name='tiny-source',
        code_mode='source_package',
        item_count=2,
        execution_manager_count=2,
        slots_per_manager=1,
        max_inflight=2,
        minimum_peak=2,
        task_sleep_seconds=0,
    )
    outputs = [
        {
            'task_output': {
                'index': index,
                'marker': f'source-package-ok:{index}',
                'dependency_version': '1.31.2',
                'sys_prefix': '/cache/manager-venvs/allocation',
                'nexus_file': '/cache/manager-venvs/allocation/lib/python3.13/site-packages/nexus/__init__.py',
            }
        }
        for index in range(2)
    ]

    assert tkex_runner.validate_source_outputs(profile, outputs) == {
        'dependency_versions': ['1.31.2'],
        'sys_prefixes': ['/cache/manager-venvs/allocation'],
        'nexus_files': ['/cache/manager-venvs/allocation/lib/python3.13/site-packages/nexus/__init__.py'],
    }

    outputs[1]['task_output']['marker'] = 'wrong'
    with pytest.raises(AssertionError, match='source markers'):
        tkex_runner.validate_source_outputs(profile, outputs)


def test_validate_sandbox_outputs_uses_ready_intervals_not_child_running_samples():
    profile = BatchLoadProfile(
        name='sandbox-tiny',
        code_mode='config_only',
        item_count=3,
        execution_manager_count=3,
        slots_per_manager=1,
        max_inflight=3,
        minimum_peak=2,
        task_sleep_seconds=60,
        timeout_per_run_seconds=180,
        task_kind='sandbox_agent_echo',
        minimum_sandbox_peak=2,
    )
    outputs = [
        {
            'task_output': {
                'index': 0,
                'marker': 'sandbox-tiny-000000000000',
                'runtime_id': 'runtime-0',
                'sandbox_ready_at': 100.0,
                'sandbox_hold_finished_at': 200.0,
                'command_return_code': 0,
                'command_stdout': 'sandbox-tiny-000000000000',
                'agent_exit_status': 'finished',
            }
        },
        {
            'task_output': {
                'index': 1,
                'marker': 'sandbox-tiny-000000000001',
                'runtime_id': 'runtime-1',
                'sandbox_ready_at': 150.0,
                'sandbox_hold_finished_at': 250.0,
                'command_return_code': 0,
                'command_stdout': 'sandbox-tiny-000000000001',
                'agent_exit_status': 'finished',
            }
        },
        {
            'task_output': {
                'index': 2,
                'marker': 'sandbox-tiny-000000000002',
                'runtime_id': 'runtime-2',
                'sandbox_ready_at': 200.0,
                'sandbox_hold_finished_at': 300.0,
                'command_return_code': 0,
                'command_stdout': 'sandbox-tiny-000000000002',
                'agent_exit_status': 'finished',
            }
        },
    ]

    assert tkex_runner.validate_sandbox_outputs(profile, outputs) == {
        'agent_exit_statuses': {'finished': 3},
        'peak_ready_sandboxes': 2,
        'unique_marker_count': 3,
        'unique_runtime_id_count': 3,
    }

    profile = replace(profile, minimum_sandbox_peak=3)
    with pytest.raises(AssertionError, match='peak ready sandboxes 2 did not reach 3'):
        tkex_runner.validate_sandbox_outputs(profile, outputs)


async def test_gongfeng_release_observer_waits_for_every_runtime_to_stop(monkeypatch):
    class FakeProvider:
        def __init__(self):
            self.reads = 0

        async def list_runtimes(self, *, runtime_ids, page, per_page):
            assert page == 1
            assert per_page == len(runtime_ids)
            self.reads += 1
            status = 'scheduled' if self.reads <= 2 else 'stopped'
            return ([{'runtime_id': runtime_id, 'status': status} for runtime_id in runtime_ids], len(runtime_ids))

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(tkex_runner.asyncio, 'sleep', no_sleep)
    observer = object.__new__(tkex_runner.GongfengReleaseObserver)
    observer.provider = FakeProvider()
    observer.list_chunk_size = 2
    observer.list_concurrency = 1

    result = await observer.wait_released(
        ['runtime-0', 'runtime-1', 'runtime-2'],
        timeout=1,
        poll_interval=0,
    )

    assert result == {
        'checked_runtime_count': 3,
        'active_runtime_count': 0,
        'release_status_counts': {'stopped': 3},
    }


async def test_gongfeng_release_observer_emergency_stops_leaked_runtimes(monkeypatch):
    stopped = []

    class FakeProvider:
        async def list_runtimes(self, *, runtime_ids, page, per_page):
            return ([{'runtime_id': runtime_id, 'status': 'scheduled'} for runtime_id in runtime_ids], len(runtime_ids))

        async def _stop_runtime(self, runtime_ids):
            stopped.extend(runtime_ids)

    monotonic = iter([0.0, 2.0])
    monkeypatch.setattr(tkex_runner, '_monotonic', lambda: next(monotonic), raising=False)
    observer = object.__new__(tkex_runner.GongfengReleaseObserver)
    observer.provider = FakeProvider()
    observer.list_chunk_size = 100
    observer.list_concurrency = 1

    with pytest.raises(TimeoutError, match='2 Gongfeng runtimes remain active'):
        await observer.wait_released(['runtime-0', 'runtime-1'], timeout=1, poll_interval=0)

    assert stopped == ['runtime-0', 'runtime-1']


async def test_gongfeng_batch_snapshot_paginates_by_batch_id_and_counts_ready_runtimes():
    class FakeProvider:
        async def list_runtimes(self, *, runtime_ids=None, batch_id=None, page, per_page):
            assert runtime_ids is None
            assert batch_id == 'batch-a'
            assert per_page == 2
            pages = {
                1: [
                    {'runtime_id': 'runtime-0', 'batch_id': 'batch-a', 'status': 'scheduled'},
                    {'runtime_id': 'runtime-1', 'batch_id': 'batch-a', 'status': 'stopped'},
                ],
                2: [{'runtime_id': 'runtime-2', 'batch_id': 'batch-a', 'status': 'scheduled'}],
            }
            return pages[page], 0

    observer = object.__new__(tkex_runner.GongfengReleaseObserver)
    observer.provider = FakeProvider()
    observer.batch_list_page_size = 2

    snapshot = await observer.batch_snapshot('batch-a')

    assert snapshot == {
        'batch_id': 'batch-a',
        'runtime_count': 3,
        'ready_runtime_count': 2,
        'active_runtime_ids': ['runtime-0', 'runtime-2'],
        'status_counts': {'scheduled': 2, 'stopped': 1},
        'runtime_ids': ['runtime-0', 'runtime-1', 'runtime-2'],
    }


async def test_gongfeng_batch_snapshot_retries_mutating_page_boundaries(monkeypatch):
    class FakeProvider:
        def __init__(self):
            self.scan = 0

        async def list_runtimes(self, *, runtime_ids=None, batch_id=None, page, per_page):
            assert runtime_ids is None
            assert batch_id == 'batch-a'
            assert per_page == 2
            if page == 1:
                self.scan += 1
            if self.scan == 1:
                pages = {
                    1: [
                        {'runtime_id': 'runtime-0', 'batch_id': 'batch-a', 'status': 'scheduled'},
                        {'runtime_id': 'runtime-1', 'batch_id': 'batch-a', 'status': 'scheduled'},
                    ],
                    2: [
                        {'runtime_id': 'runtime-1', 'batch_id': 'batch-a', 'status': 'scheduled'},
                        {'runtime_id': 'runtime-2', 'batch_id': 'batch-a', 'status': 'scheduled'},
                    ],
                    3: [],
                }
                return pages[page], 4
            pages = {
                1: [
                    {'runtime_id': 'runtime-0', 'batch_id': 'batch-a', 'status': 'scheduled'},
                    {'runtime_id': 'runtime-1', 'batch_id': 'batch-a', 'status': 'scheduled'},
                ],
                2: [{'runtime_id': 'runtime-2', 'batch_id': 'batch-a', 'status': 'scheduled'}],
            }
            return pages[page], 3

    sleeps = []

    async def no_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(tkex_runner.asyncio, 'sleep', no_sleep)
    observer = object.__new__(tkex_runner.GongfengReleaseObserver)
    observer.provider = FakeProvider()
    observer.batch_list_page_size = 2
    observer.batch_snapshot_attempts = 3
    observer.batch_snapshot_retry_seconds = 0.25

    snapshot = await observer.batch_snapshot('batch-a')

    assert observer.provider.scan == 2
    assert sleeps == [0.25]
    assert snapshot['runtime_count'] == 3
    assert snapshot['ready_runtime_count'] == 3
    assert snapshot['runtime_ids'] == ['runtime-0', 'runtime-1', 'runtime-2']


async def test_gongfeng_batch_snapshot_uses_conservative_unique_lower_bound_after_retry_exhaustion(monkeypatch):
    class FakeProvider:
        def __init__(self):
            self.scan = 0

        async def list_runtimes(self, *, runtime_ids=None, batch_id=None, page, per_page):
            assert runtime_ids is None
            assert batch_id == 'batch-a'
            assert per_page == 2
            if page == 1:
                self.scan += 1
            pages = {
                1: [
                    {'runtime_id': 'runtime-0', 'batch_id': 'batch-a', 'status': 'scheduled'},
                    {'runtime_id': 'runtime-1', 'batch_id': 'batch-a', 'status': 'scheduled'},
                ],
                2: [
                    {'runtime_id': 'runtime-1', 'batch_id': 'batch-a', 'status': 'stopped'},
                    {'runtime_id': 'runtime-2', 'batch_id': 'batch-a', 'status': 'scheduled'},
                ],
                3: [],
            }
            return pages[page], 0

    sleeps = []

    async def no_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(tkex_runner.asyncio, 'sleep', no_sleep)
    observer = object.__new__(tkex_runner.GongfengReleaseObserver)
    observer.provider = FakeProvider()
    observer.batch_list_page_size = 2
    observer.batch_snapshot_attempts = 2
    observer.batch_snapshot_retry_seconds = 0.25

    snapshot = await observer.batch_snapshot('batch-a')

    assert observer.provider.scan == 2
    assert sleeps == [0.25]
    assert snapshot == {
        'batch_id': 'batch-a',
        'runtime_count': 3,
        'ready_runtime_count': 2,
        'active_runtime_ids': ['runtime-0', 'runtime-1', 'runtime-2'],
        'status_counts': {'inconsistent': 1, 'scheduled': 2},
        'runtime_ids': ['runtime-0', 'runtime-1', 'runtime-2'],
    }


async def test_gongfeng_batch_release_includes_failed_attempt_runtimes(monkeypatch):
    class FakeProvider:
        def __init__(self):
            self.reads = 0

        async def list_runtimes(self, *, runtime_ids=None, batch_id=None, page, per_page):
            assert runtime_ids is None
            assert batch_id == 'batch-a'
            assert page == 1
            self.reads += 1
            status = 'scheduled' if self.reads == 1 else 'stopped'
            return (
                [
                    {'runtime_id': 'runtime-final', 'batch_id': 'batch-a', 'status': status},
                    {'runtime_id': 'runtime-failed-attempt', 'batch_id': 'batch-a', 'status': status},
                ],
                0,
            )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(tkex_runner.asyncio, 'sleep', no_sleep)
    observer = object.__new__(tkex_runner.GongfengReleaseObserver)
    observer.provider = FakeProvider()
    observer.batch_list_page_size = 5000

    result = await observer.wait_batch_released(
        'batch-a',
        minimum_runtime_count=1,
        timeout=10,
        poll_interval=0,
    )

    assert result == {
        'batch_id': 'batch-a',
        'runtime_count': 2,
        'active_runtime_count': 0,
        'status_counts': {'stopped': 2},
    }


async def test_manager_crash_controller_deletes_only_selected_targets_and_records_recovery():
    profile = replace(
        tkex_runner.TKEX_LOAD_PROFILES['sandbox-agent-10000'],
        execution_manager_count=2,
        max_inflight=12,
        minimum_peak=10,
        minimum_sandbox_peak=10,
    )

    class FakeClient:
        def __init__(self):
            self.replaced = False
            self.deleted = []

        async def list_pods(self):
            uid = 'uid-new' if self.replaced else 'uid-old'
            return {
                'items': [
                    {
                        'metadata': {
                            'name': 'nexus-execution-0',
                            'uid': uid,
                            'annotations': {'nexus.io/allocation-id': 'alc-execution'},
                            'labels': {'nexus.io/workload': 'execution'},
                        },
                        'status': {
                            'phase': 'Running',
                            'conditions': [{'type': 'Ready', 'status': 'True'}],
                        },
                    }
                ]
            }

        async def get_pod(self, name):
            assert name == 'nexus-execution-0'
            return (await self.list_pods())['items'][0]

        async def delete_pod(self, name, *, uid):
            self.deleted.append((name, uid))
            return {'status': 'Success'}

    class FakeApi:
        def __init__(self, client):
            self.client = client

        async def control_json(self, method, path):
            assert method == 'GET'
            assert path == '/v1/managers'
            return {
                'items': [
                    {
                        'manager_id': 'alc-execution-0',
                        'session_id': 'session-new' if self.client.replaced else 'session-old',
                        'healthy': True,
                        'reserved_slots': 6,
                    }
                ]
            }

    client = FakeClient()
    api = FakeApi(client)
    controller = tkex_runner.TkexManagerCrashController(client, mode='execution', count=1)

    await controller.inject(
        api,
        allocations=[{'allocation_id': 'alc-execution', 'workload': 'execution', 'status': 'ready'}],
        profile=profile,
        task_list_ref={'artifact_id': 'task-list'},
    )

    assert client.deleted == [('nexus-execution-0', 'uid-old')]
    assert controller.planned_pending == 1
    assert controller.report['status'] == 'recovering'
    client.replaced = True
    assert await controller.observe_recovery(api)
    assert controller.planned_pending == 0
    assert controller.report['status'] == 'recovered'
    assert controller.report['replacements'][0]['replacement_pod_uid'] == 'uid-new'


async def test_manager_crash_controller_waits_for_exact_file_approval(tmp_path):
    profile = replace(
        tkex_runner.TKEX_LOAD_PROFILES['sandbox-agent-10000'],
        execution_manager_count=2,
        max_inflight=12,
        minimum_peak=10,
        minimum_sandbox_peak=10,
    )
    approval_file = tmp_path / 'manager-crash-approved.json'

    class FakeClient:
        base_url = 'https://tkex.example'
        cluster_name = 'cluster-a'
        project_name = 'project-a'
        namespace = 'namespace-a'

        def __init__(self):
            self.deleted = []

        async def list_pods(self):
            return {
                'items': [
                    {
                        'metadata': {
                            'name': 'nexus-execution-0',
                            'uid': 'uid-old',
                            'annotations': {'nexus.io/allocation-id': 'alc-execution'},
                            'labels': {'nexus.io/workload': 'execution'},
                        },
                        'status': {
                            'phase': 'Running',
                            'conditions': [{'type': 'Ready', 'status': 'True'}],
                        },
                    }
                ]
            }

        async def get_pod(self, name):
            assert name == 'nexus-execution-0'
            return (await self.list_pods())['items'][0]

        def pod_delete_request(self, name, *, uid):
            return (
                f'/v2/forward/stke/api/v1/namespaces/{self.namespace}/pods/{name}',
                {
                    'apiVersion': 'v1',
                    'kind': 'DeleteOptions',
                    'gracePeriodSeconds': 0,
                    'propagationPolicy': 'Background',
                    'preconditions': {'uid': uid},
                },
            )

        async def delete_pod(self, name, *, uid):
            self.deleted.append((name, uid))
            return {'status': 'Success'}

    class FakeApi:
        async def control_json(self, method, path):
            assert method == 'GET'
            assert path == '/v1/managers'
            return {
                'items': [
                    {
                        'manager_id': 'alc-execution-0',
                        'session_id': 'session-old',
                        'healthy': True,
                        'reserved_slots': 6,
                    }
                ]
            }

    client = FakeClient()
    controller = tkex_runner.TkexManagerCrashController(
        client,
        mode='execution',
        count=1,
        approval_file=approval_file,
    )
    kwargs = {
        'allocations': [{'allocation_id': 'alc-execution', 'workload': 'execution', 'status': 'ready'}],
        'profile': profile,
        'task_list_ref': {'artifact_id': 'task-list'},
    }

    await controller.inject(FakeApi(), **kwargs)

    assert client.deleted == []
    assert not controller.injected
    assert controller.report['status'] == 'awaiting_confirmation'
    request = json.loads(controller.approval_request_file.read_text())
    assert request['approval_id'] == controller.report['approval_id']
    assert request['operations'] == [
        {
            'body': {
                'apiVersion': 'v1',
                'gracePeriodSeconds': 0,
                'kind': 'DeleteOptions',
                'preconditions': {'uid': 'uid-old'},
                'propagationPolicy': 'Background',
            },
            'cluster': 'cluster-a',
            'expected_effect': 'Delete Pod nexus-execution-0 and let its StatefulSetPlus recreate it.',
            'method': 'DELETE',
            'namespace': 'namespace-a',
            'path': '/v2/forward/stke/api/v1/namespaces/namespace-a/pods/nexus-execution-0',
            'project': 'project-a',
            'resource': {'kind': 'Pod', 'name': 'nexus-execution-0', 'uid': 'uid-old'},
            'url': 'https://tkex.example/v2/forward/stke/api/v1/namespaces/namespace-a/pods/nexus-execution-0',
            'verification': 'Require a new Pod UID and a new healthy manager session for alc-execution-0.',
        }
    ]

    approval_file.write_text(json.dumps({'approval_id': request['approval_id']}))
    await controller.inject(FakeApi(), **kwargs)

    assert client.deleted == [('nexus-execution-0', 'uid-old')]
    assert controller.injected
    assert controller.report['status'] == 'recovering'


async def test_manager_crash_controller_rejects_approved_target_after_allocation_starts_draining(tmp_path):
    profile = replace(
        tkex_runner.TKEX_LOAD_PROFILES['sandbox-agent-10000'],
        execution_manager_count=2,
        max_inflight=12,
        minimum_peak=10,
        minimum_sandbox_peak=10,
    )
    approval_file = tmp_path / 'manager-crash-approved.json'

    class FakeClient:
        base_url = 'https://tkex.example'
        cluster_name = 'cluster-a'
        project_name = 'project-a'
        namespace = 'namespace-a'

        def __init__(self):
            self.deleted = []

        async def list_pods(self):
            return {
                'items': [
                    {
                        'metadata': {
                            'name': 'nexus-execution-0',
                            'uid': 'uid-old',
                            'annotations': {'nexus.io/allocation-id': 'alc-execution'},
                            'labels': {'nexus.io/workload': 'execution'},
                        },
                        'status': {
                            'phase': 'Running',
                            'conditions': [{'type': 'Ready', 'status': 'True'}],
                        },
                    }
                ]
            }

        async def get_pod(self, name):
            assert name == 'nexus-execution-0'
            return (await self.list_pods())['items'][0]

        def pod_delete_request(self, name, *, uid):
            return (
                f'/v2/forward/stke/api/v1/namespaces/{self.namespace}/pods/{name}',
                {'preconditions': {'uid': uid}},
            )

        async def delete_pod(self, name, *, uid):
            self.deleted.append((name, uid))
            return {'status': 'Success'}

    class FakeApi:
        async def control_json(self, method, path):
            assert (method, path) == ('GET', '/v1/managers')
            return {
                'items': [
                    {
                        'manager_id': 'alc-execution-0',
                        'session_id': 'session-old',
                        'healthy': True,
                        'reserved_slots': 6,
                    }
                ]
            }

    client = FakeClient()
    controller = tkex_runner.TkexManagerCrashController(
        client,
        mode='execution',
        count=1,
        approval_file=approval_file,
    )
    ready_allocations = [{'allocation_id': 'alc-execution', 'workload': 'execution', 'status': 'ready'}]

    await controller.inject(
        FakeApi(),
        allocations=ready_allocations,
        profile=profile,
        task_list_ref={'artifact_id': 'task-list'},
    )
    request = json.loads(controller.approval_request_file.read_text())
    approval_file.write_text(json.dumps({'approval_id': request['approval_id']}))
    await controller.inject(
        FakeApi(),
        allocations=[{'allocation_id': 'alc-execution', 'workload': 'execution', 'status': 'draining'}],
        profile=profile,
        task_list_ref={'artifact_id': 'task-list'},
    )

    assert client.deleted == []
    assert not controller.injected
    assert controller.report['status'] == 'approval_stale'
    assert controller.report['stale_reason'] == 'allocation alc-execution entered draining'


def test_batch_manager_crash_validation_requires_parent_retry_and_stable_task_list():
    controller = tkex_runner.TkexManagerCrashController(object(), mode='batch', count=1)
    controller.report.update(
        {
            'status': 'recovered',
            'task_list_ref_before_crash': {'artifact_id': 'task-list'},
        }
    )
    controller._injected_at_monotonic = 1
    controller.targets = [
        tkex_runner.ManagerCrashTarget(
            manager_id='alc-batch-0',
            pod_name='nexus-batch-0',
            previous_pod_uid='uid-old',
            previous_session_id='session-old',
            reserved_slots=1,
        )
    ]

    controller.validate_completed(
        parent={'attempt_count': 2},
        summary={'task_list_ref': {'artifact_id': 'task-list'}},
    )
    with pytest.raises(AssertionError, match='parent attempt'):
        controller.validate_completed(
            parent={'attempt_count': 1},
            summary={'task_list_ref': {'artifact_id': 'task-list'}},
        )


async def test_advance_manager_crash_injects_once_and_tracks_recovery():
    profile = tkex_runner.TKEX_LOAD_PROFILES['sandbox-agent-10000']

    class FakeController:
        mode = 'batch'
        injected = False
        recovered = False
        planned_pending = 0
        report = {'status': 'armed'}

        def __init__(self):
            self.inject_calls = 0
            self.recovery_calls = 0

        async def inject(self, api, *, allocations, profile, task_list_ref):
            self.inject_calls += 1
            self.injected = True
            self.planned_pending = 1
            self.report = {'status': 'recovering', 'task_list_ref_before_crash': task_list_ref}

        async def observe_recovery(self, api):
            self.recovery_calls += 1
            self.recovered = True
            self.planned_pending = 0
            self.report = {'status': 'recovered'}
            return True

    controller = FakeController()
    stage = {}
    summary = {
        'generation_status': 'complete',
        'submitted_items': 10_100,
        'total_items': 16_000,
        'task_list_ref': {'artifact_id': 'task-list'},
    }

    await tkex_runner._advance_manager_crash(
        controller,
        object(),
        allocations=[{'allocation_id': 'alc-batch', 'workload': 'orchestration'}],
        profile=profile,
        summary=summary,
        peak_running_items=10_000,
        live_ready_sandboxes=None,
        stage=stage,
    )
    await tkex_runner._advance_manager_crash(
        controller,
        object(),
        allocations=[{'allocation_id': 'alc-batch', 'workload': 'orchestration'}],
        profile=profile,
        summary=summary,
        peak_running_items=10_000,
        live_ready_sandboxes=None,
        stage=stage,
    )

    assert controller.inject_calls == 1
    assert controller.recovery_calls == 1
    assert stage['manager_crash'] == {'status': 'recovered'}


async def test_pending_manager_crash_approval_is_polled_without_a_tkex_observation():
    profile = tkex_runner.TKEX_LOAD_PROFILES['sandbox-agent-10000']

    class FakeController:
        mode = 'batch'
        awaiting_confirmation = True
        injected = False
        recovered = False
        planned_pending = 0
        report = {'status': 'awaiting_confirmation'}

        def __init__(self):
            self.inject_calls = 0

        async def inject(self, api, *, allocations, profile, task_list_ref):
            self.inject_calls += 1

    controller = FakeController()
    stage = {}

    await tkex_runner._poll_manager_crash(
        controller,
        object(),
        observed=False,
        allocations=[{'allocation_id': 'alc-batch', 'workload': 'orchestration', 'status': 'ready'}],
        profile=profile,
        summary={
            'generation_status': 'complete',
            'submitted_items': 16_000,
            'total_items': 16_000,
            'task_list_ref': {'artifact_id': 'task-list'},
        },
        peak_running_items=10_000,
        live_ready_sandboxes=None,
        stage=stage,
    )

    assert controller.inject_calls == 1
    assert stage['manager_crash'] == {'status': 'awaiting_confirmation'}


def test_pending_manager_crash_approval_suppresses_full_tkex_observation():
    controller = SimpleNamespace(awaiting_confirmation=True)

    assert not tkex_runner._tkex_observation_due(
        allocations=[{'allocation_id': 'alc-batch'}],
        now=20,
        next_observation_at=10,
        manager_crash_controller=controller,
    )
    controller.awaiting_confirmation = False
    assert tkex_runner._tkex_observation_due(
        allocations=[{'allocation_id': 'alc-batch'}],
        now=20,
        next_observation_at=10,
        manager_crash_controller=controller,
    )


async def test_submit_batch_retries_retryable_quota_rejection(monkeypatch):
    sleeps = []

    class FakeApi:
        def __init__(self):
            self.calls = 0

        async def control_json(self, method, path, *, body=None):
            assert (method, path, body) == ('POST', '/v1/submissions/batch', {'idempotency_key': 'batch'})
            self.calls += 1
            if self.calls == 1:
                raise tkex_runner.CloudBatchApiError(
                    method,
                    path,
                    429,
                    {'error': 'environment_slots_limit_exceeded', 'retryable': True},
                )
            return {'execution_id': 'parent'}

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(tkex_runner.asyncio, 'sleep', fake_sleep)
    api = FakeApi()

    assert await tkex_runner._submit_batch_with_retry(api, {'idempotency_key': 'batch'}) == {'execution_id': 'parent'}
    assert api.calls == 2
    assert sleeps == [1.0]


async def test_submit_batch_does_not_retry_non_retryable_rejection(monkeypatch):
    class FakeApi:
        async def control_json(self, method, path, *, body=None):
            raise tkex_runner.CloudBatchApiError(
                method,
                path,
                429,
                {'error': 'environment_slots_limit_exceeded', 'retryable': False},
            )

    async def unexpected_sleep(_delay):
        raise AssertionError('non-retryable rejection must not sleep')

    monkeypatch.setattr(tkex_runner.asyncio, 'sleep', unexpected_sleep)

    with pytest.raises(tkex_runner.CloudBatchApiError):
        await tkex_runner._submit_batch_with_retry(FakeApi(), {'idempotency_key': 'batch'})


async def test_run_stage_records_peak_before_tkex_observer_catches_ready():
    observation_order = []
    profile = BatchLoadProfile(
        name='tiny',
        code_mode='config_only',
        item_count=2,
        execution_manager_count=2,
        slots_per_manager=1,
        max_inflight=2,
        minimum_peak=2,
        task_sleep_seconds=0,
    )

    class TrackingSummary(dict):
        def get(self, key, default=None):
            if key == 'child_execution_status_counts':
                observation_order.append('running-count')
            return super().get(key, default)

    class FakeApi:
        consecutive_gateway_5xx = 0

        def __init__(self):
            self.execution_reads = 0
            self.allocation_reads = 0

        async def control_json(self, method, path, *, body=None):
            if (method, path) == ('GET', '/v1/system/status'):
                return {
                    'artifacts': {
                        'uploads_by_status': {'completed': 10},
                        'failed_uploads': 0,
                        'checksum_failures': 0,
                    }
                }
            if (method, path) == ('POST', '/v1/submissions/batch'):
                assert body['idempotency_key'] == 'tiny'
                return {'execution_id': 'parent'}
            if (method, path) == ('GET', '/v1/executions/parent'):
                self.execution_reads += 1
                return {'execution_id': 'parent', 'status': 'running' if self.execution_reads < 3 else 'succeeded'}
            if method == 'GET' and path in {
                '/v1/executions/parent/batch',
                '/v1/executions/parent/batch?include_status_counts=true',
            }:
                running = self.execution_reads == 1
                return TrackingSummary(
                    {
                        'generation_status': 'complete',
                        'materialized_count': 2,
                        'submitted_items': 2,
                        'active_items': 2 if running else 0,
                        'terminal_items': 0 if running else 2,
                        'child_execution_status_counts': {'running': 2} if running else {'succeeded': 2},
                        'summary': {
                            'total': 2,
                            'succeeded': 2,
                            'failed': 0,
                            'killed': 0,
                            'lost': 0,
                            'expired': 0,
                            'cancelled': 0,
                        },
                    }
                )
            if (method, path) == ('GET', '/v1/executions/parent/allocations'):
                self.allocation_reads += 1
                if self.allocation_reads <= 3:
                    return {
                        'items': [
                            {
                                'allocation_id': 'parent-allocation',
                                'status': 'ready',
                                'desired_manager_count': 1,
                                'ready_manager_count': 1,
                            },
                            {
                                'allocation_id': 'child-allocation',
                                'status': 'ready',
                                'desired_manager_count': 2,
                                'ready_manager_count': 2,
                            },
                        ]
                    }
                return {
                    'items': [
                        {'allocation_id': 'parent-allocation', 'status': 'released'},
                        {'allocation_id': 'child-allocation', 'status': 'released'},
                    ]
                }
            raise AssertionError((method, path, body))

        async def list_all_batch_items(self, execution_id):
            assert execution_id == 'parent'
            status = 'running' if self.execution_reads < 3 else 'succeeded'
            return [
                {
                    'item_id': 'item-1',
                    'status': status,
                    'child_execution_status': status,
                    'current_child_execution_id': 'child-1',
                },
                {
                    'item_id': 'item-2',
                    'status': status,
                    'child_execution_status': status,
                    'current_child_execution_id': 'child-2',
                },
            ]

        async def execution_facts(self, execution_ids):
            assert execution_ids == {'child-1', 'child-2'}
            return {
                'child-1': {'execution_id': 'child-1', 'status': 'succeeded', 'result_artifact_id': 'result-1'},
                'child-2': {'execution_id': 'child-2', 'status': 'succeeded', 'result_artifact_id': 'result-2'},
            }

    class FakeObserver:
        def __init__(self):
            self.active_reads = 0

        async def observe(self, allocations):
            observation_order.append('tkex-observe')
            released = all(item['status'] == 'released' for item in allocations)
            if not released:
                self.active_reads += 1
                await asyncio.sleep(0.002)
            transient_startup_gap = not released and self.active_reads == 1
            return {
                'running_manager_containers': 0 if released else 3,
                'active_pod_count': 0 if released else 3,
                'workload_count': 0 if released else 2,
                'desired_replicas': 0 if released else 3,
                'ready_replicas': 0 if released or transient_startup_gap else 3,
                'unreachable_managers': 3 if transient_startup_gap else 0,
                'restart_count': 0,
                'oom_killed_container_count': 0,
                'services_present': [],
                'pod_ip_downward_api': True,
                'db_pool_sizes': ['1'],
            }

    stage = await tkex_runner._run_stage(
        FakeApi(),
        FakeObserver(),
        profile=profile,
        payload={'idempotency_key': 'tiny'},
        available_cpu_cores=10,
        available_memory_gib=10,
        database_metrics_reader=None,
        poll_interval=0,
        observation_interval=0,
    )

    assert stage['execution_id'] == 'parent'
    assert observation_order.index('running-count') < observation_order.index('tkex-observe')
    assert stage['peak_active_items'] == 2
    assert stage['peak_running_items'] == 2
    assert stage['concurrency_observations'][-1]['running_items'] == 0
    assert stage['scheduler_fill_seconds'] is not None
    assert stage['validation'] == {
        'item_count': 2,
        'child_execution_count': 2,
        'unique_result_artifact_count': 2,
    }


async def test_run_stage_samples_concurrency_while_tkex_observer_is_slow():
    profile = BatchLoadProfile(
        name='tiny-narrow-peak',
        code_mode='config_only',
        item_count=2,
        execution_manager_count=2,
        slots_per_manager=1,
        max_inflight=2,
        minimum_peak=2,
        task_sleep_seconds=0,
    )

    class FakeObserver:
        def __init__(self):
            self.observing = False

        async def observe(self, allocations):
            released = all(item['status'] == 'released' for item in allocations)
            if not released:
                self.observing = True
                await asyncio.sleep(0.02)
                self.observing = False
            return {
                'running_manager_containers': 0 if released else 3,
                'active_pod_count': 0 if released else 3,
                'workload_count': 0 if released else 2,
                'desired_replicas': 0 if released else 3,
                'ready_replicas': 0 if released else 3,
                'unreachable_managers': 0,
                'restart_count': 0,
                'oom_killed_container_count': 0,
                'services_present': [],
                'pod_ip_downward_api': True,
                'db_pool_sizes': ['1'],
            }

    observer = FakeObserver()

    class FakeApi:
        consecutive_gateway_5xx = 0

        def __init__(self):
            self.execution_reads = 0
            self.allocation_reads = 0

        async def control_json(self, method, path, *, body=None):
            if (method, path) == ('GET', '/v1/system/status'):
                return {
                    'artifacts': {
                        'uploads_by_status': {'completed': 10},
                        'failed_uploads': 0,
                        'checksum_failures': 0,
                    }
                }
            if (method, path) == ('POST', '/v1/submissions/batch'):
                return {'execution_id': 'narrow-parent'}
            if (method, path) == ('GET', '/v1/executions/narrow-parent'):
                self.execution_reads += 1
                return {
                    'execution_id': 'narrow-parent',
                    'status': 'running' if self.execution_reads == 1 else 'succeeded',
                }
            if method == 'GET' and path in {
                '/v1/executions/narrow-parent/batch',
                '/v1/executions/narrow-parent/batch?include_status_counts=true',
            }:
                running = observer.observing
                return {
                    'generation_status': 'complete',
                    'materialized_count': 2,
                    'submitted_items': 2,
                    'active_items': 2 if running else 0,
                    'terminal_items': 0 if running else 2,
                    'child_execution_status_counts': {'running': 2} if running else {'succeeded': 2},
                    'summary': {
                        'total': 2,
                        'succeeded': 2,
                        'failed': 0,
                        'killed': 0,
                        'lost': 0,
                        'expired': 0,
                        'cancelled': 0,
                    },
                }
            if (method, path) == ('GET', '/v1/executions/narrow-parent/allocations'):
                self.allocation_reads += 1
                status = 'ready' if self.allocation_reads <= 2 else 'released'
                ready = 1 if status == 'ready' else 0
                return {
                    'items': [
                        {
                            'allocation_id': 'narrow-parent-allocation',
                            'status': status,
                            'desired_manager_count': 1,
                            'ready_manager_count': ready,
                        },
                        {
                            'allocation_id': 'narrow-child-allocation',
                            'status': status,
                            'desired_manager_count': 2,
                            'ready_manager_count': 2 * ready,
                        },
                    ]
                }
            raise AssertionError((method, path, body))

        async def list_all_batch_items(self, execution_id):
            assert execution_id == 'narrow-parent'
            return [
                {
                    'item_id': 'item-1',
                    'status': 'succeeded',
                    'child_execution_status': 'succeeded',
                    'current_child_execution_id': 'child-1',
                },
                {
                    'item_id': 'item-2',
                    'status': 'succeeded',
                    'child_execution_status': 'succeeded',
                    'current_child_execution_id': 'child-2',
                },
            ]

        async def execution_facts(self, execution_ids):
            assert execution_ids == {'child-1', 'child-2'}
            return {
                'child-1': {'execution_id': 'child-1', 'status': 'succeeded', 'result_artifact_id': 'result-1'},
                'child-2': {'execution_id': 'child-2', 'status': 'succeeded', 'result_artifact_id': 'result-2'},
            }

    stage = await tkex_runner._run_stage(
        FakeApi(),
        observer,
        profile=profile,
        payload={'idempotency_key': 'tiny-narrow-peak'},
        available_cpu_cores=10,
        available_memory_gib=10,
        database_metrics_reader=None,
        poll_interval=0.001,
        observation_interval=0,
    )

    assert stage['peak_active_items'] == 2
    assert stage['peak_running_items'] == 2


async def test_run_stage_records_upload_guardrail_and_waits_for_release_on_failure():
    profile = BatchLoadProfile(
        name='tiny-failure',
        code_mode='config_only',
        item_count=100,
        execution_manager_count=2,
        slots_per_manager=1,
        max_inflight=2,
        minimum_peak=2,
        task_sleep_seconds=0,
    )

    class FakeApi:
        consecutive_gateway_5xx = 0

        def __init__(self):
            self.system_reads = 0
            self.stop_calls = 0

        async def control_json(self, method, path, *, body=None):
            if (method, path) == ('GET', '/v1/system/status'):
                self.system_reads += 1
                failed = 0 if self.system_reads == 1 else 2
                completed = 10 if self.system_reads == 1 else 108
                return {
                    'artifacts': {
                        'uploads_by_status': {'completed': completed},
                        'failed_uploads': failed,
                        'checksum_failures': 0,
                    }
                }
            if (method, path) == ('POST', '/v1/submissions/batch'):
                return {'execution_id': 'failed-parent'}
            if (method, path) == ('GET', '/v1/executions/failed-parent'):
                return {'execution_id': 'failed-parent', 'status': 'running'}
            if method == 'GET' and path in {
                '/v1/executions/failed-parent/batch',
                '/v1/executions/failed-parent/batch?include_status_counts=true',
            }:
                return {
                    'generation_status': 'complete',
                    'materialized_count': 100,
                    'submitted_items': 100,
                    'active_items': 2,
                    'terminal_items': 98,
                }
            if (method, path) == ('GET', '/v1/executions/failed-parent/allocations'):
                status = 'released' if self.stop_calls else 'ready'
                ready = 0 if self.stop_calls else 1
                return {
                    'items': [
                        {
                            'allocation_id': 'failed-parent-allocation',
                            'status': status,
                            'desired_manager_count': 1,
                            'ready_manager_count': ready,
                        },
                        {
                            'allocation_id': 'failed-child-allocation',
                            'status': status,
                            'desired_manager_count': 2,
                            'ready_manager_count': 0 if self.stop_calls else 2,
                        },
                    ]
                }
            if (method, path) == ('POST', '/v1/executions/failed-parent/stop'):
                self.stop_calls += 1
                return {'status': 'stopping'}
            raise AssertionError((method, path, body))

        async def list_all_batch_items(self, execution_id):
            assert execution_id == 'failed-parent'
            return [
                {'item_id': 'item-1', 'status': 'running', 'child_execution_status': 'running'},
                {'item_id': 'item-2', 'status': 'running', 'child_execution_status': 'running'},
            ]

    class FakeObserver:
        async def observe(self, allocations):
            released = all(item['status'] == 'released' for item in allocations)
            return {
                'running_manager_containers': 0 if released else 3,
                'active_pod_count': 0 if released else 3,
                'workload_count': 0 if released else 2,
                'unreachable_managers': 0,
                'restart_count': 0,
                'oom_killed_container_count': 0,
                'services_present': [],
                'pod_ip_downward_api': True,
                'db_pool_sizes': ['1'],
            }

    api = FakeApi()
    with pytest.raises(tkex_runner.StageRunError) as error:
        await tkex_runner._run_stage(
            api,
            FakeObserver(),
            profile=profile,
            payload={'idempotency_key': 'tiny-failure'},
            available_cpu_cores=10,
            available_memory_gib=10,
            database_metrics_reader=None,
            poll_interval=0,
            observation_interval=0,
        )

    assert api.stop_calls == 1
    assert error.value.stage['status'] == 'failed'
    assert error.value.stage['cleanup'] == {'status': 'released'}
    assert error.value.stage['artifact_observations'][-1]['result_upload_failure_ratio'] == pytest.approx(0.02)
    assert error.value.stage['guardrail_stop_reasons'] == ['result_upload_failure_ratio=0.020 exceeds 0.010']


async def test_execute_profiles_persists_failed_stage_before_raising(monkeypatch, tmp_path):
    profile = BatchLoadProfile(
        name='tiny-report',
        code_mode='config_only',
        item_count=2,
        execution_manager_count=1,
        slots_per_manager=2,
        max_inflight=2,
        minimum_peak=2,
        task_sleep_seconds=0,
    )
    token_file = tmp_path / 'submit-token'
    token_file.write_text('secret')
    token_file.chmod(0o600)
    report_path = tmp_path / 'report.json'
    args = argparse.Namespace(
        taihu_token_file=token_file,
        control_url='https://control.example',
        application_url='https://application.example',
        environment_id='tkex-stress',
        project_root=tmp_path,
        database_url_file=None,
        control_database_metrics=False,
        available_cpu_cores=10,
        available_memory_gib=10,
        poll_interval=0,
        observation_interval=0,
        tkex_cluster_name='cluster',
        tkex_namespace='namespace',
        universal_image='registry.example/universal@sha256:' + '1' * 64,
        workload_image='registry.example/workload@sha256:' + '2' * 64,
        builder_image='registry.example/builder@sha256:' + '3' * 64,
    )

    class FakeObserver:
        async def close(self):
            return None

    class FakeApi:
        maximum_consecutive_gateway_5xx = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def control_json(self, method, path):
            assert (method, path) == ('GET', '/v1/health')
            return {'status': 'ok'}

        async def application_json(self, method, path):
            assert (method, path) == ('GET', '/v1/health')
            return {'status': 'ok'}

        async def upload_artifact(self, path, *, kind, sha256, size_bytes, idempotency_key, environment_id):
            return {
                'artifact_id': f'artifact-{kind}',
                'kind': kind,
                'sha256': sha256,
                'size_bytes': size_bytes,
            }

    failed_stage = {
        'name': profile.name,
        'execution_id': 'parent',
        'status': 'failed',
        'guardrail_stop_reasons': ['result_upload_failure_ratio=0.020 exceeds 0.010'],
        'cleanup': {'status': 'released'},
    }

    async def fail_stage(*args, **kwargs):
        raise tkex_runner.StageRunError(failed_stage, RuntimeError('guardrail'))

    monkeypatch.setattr(tkex_runner, 'TkexObserver', lambda config: FakeObserver())
    monkeypatch.setattr(tkex_runner, 'CloudBatchApi', lambda **kwargs: FakeApi())
    monkeypatch.setattr(tkex_runner, '_tkex_observer_config', lambda args: object())
    monkeypatch.setattr(tkex_runner, '_run_stage', fail_stage)

    with pytest.raises(tkex_runner.StageRunError):
        await tkex_runner.execute_profiles(
            args,
            profiles=[profile],
            plan={'stages': [{'name': profile.name}]},
            report_path=report_path,
        )

    report = json.loads(report_path.read_text())
    assert report['status'] == 'failed'
    assert report['stages'] == [failed_stage]


async def test_read_control_database_metrics_requires_available_normalized_metrics():
    class FakeApi:
        async def control_json(self, method, path):
            assert (method, path) == ('GET', '/v1/system/status')
            return {
                'database': {
                    'available': True,
                    'dialect': 'mysql',
                    'max_connections': 2000,
                    'threads_connected': 140,
                    'threads_running': 5,
                    'max_used_connections': 350,
                    'slow_queries': 3,
                    'aborted_connects': 8,
                }
            }

    assert await tkex_runner.read_control_database_metrics(FakeApi()) == {
        'available': True,
        'dialect': 'mysql',
        'max_connections': 2000,
        'threads_connected': 140,
        'threads_running': 5,
        'max_used_connections': 350,
        'slow_queries': 3,
        'aborted_connects': 8,
    }

    class UnavailableApi:
        async def control_json(self, method, path):
            return {'database': {'available': False, 'dialect': 'mysql'}}

    with pytest.raises(RuntimeError, match='database metrics are unavailable'):
        await tkex_runner.read_control_database_metrics(UnavailableApi())


async def test_read_control_artifact_metrics_normalizes_completed_and_failed_uploads():
    class FakeApi:
        async def control_json(self, method, path):
            assert (method, path) == ('GET', '/v1/system/status')
            return {
                'artifacts': {
                    'uploads_by_status': {'pending': 3, 'completed': 120},
                    'failed_uploads': 2,
                    'checksum_failures': 1,
                }
            }

    assert await tkex_runner.read_control_artifact_metrics(FakeApi()) == {
        'completed_uploads': 120,
        'failed_uploads': 2,
        'checksum_failures': 1,
    }

    class EmptyApi:
        async def control_json(self, method, path):
            return {
                'artifacts': {
                    'uploads_by_status': {},
                    'failed_uploads': 0,
                    'checksum_failures': 0,
                }
            }

    assert await tkex_runner.read_control_artifact_metrics(EmptyApi()) == {
        'completed_uploads': 0,
        'failed_uploads': 0,
        'checksum_failures': 0,
    }

    class IncompleteApi:
        async def control_json(self, method, path):
            return {'artifacts': {'uploads_by_status': {}}}

    with pytest.raises(RuntimeError, match='artifact metrics are incomplete'):
        await tkex_runner.read_control_artifact_metrics(IncompleteApi())

    class MissingApi:
        async def control_json(self, method, path):
            return {}

    with pytest.raises(RuntimeError, match='artifact metrics are incomplete'):
        await tkex_runner.read_control_artifact_metrics(MissingApi())


def test_large_profiles_accept_control_database_metrics_instead_of_local_database_url(tmp_path):
    token_file = tmp_path / 'token'
    token_file.write_text('secret')
    token_file.chmod(0o600)
    common = [
        '--control-url',
        'https://control.example',
        '--application-url',
        'https://application.example',
        '--environment-id',
        'tkex-stress',
        '--taihu-token-file',
        str(token_file),
        '--profile',
        'wheelhouse-smoke',
        '--profile',
        'container-ceiling',
        '--tkex-base-url',
        'https://tkex.example',
        '--tkex-api-account',
        'account',
        '--tkex-api-token-file',
        str(token_file),
        '--tkex-cluster-name',
        'cluster',
        '--tkex-project-name',
        'project',
        '--tkex-namespace',
        'namespace',
        '--universal-image',
        'registry.example/universal@sha256:' + '1' * 64,
        '--workload-image',
        'registry.example/workload@sha256:' + '2' * 64,
        '--builder-image',
        'registry.example/builder@sha256:' + '3' * 64,
    ]
    profiles = tkex_runner.selected_profiles(['wheelhouse-smoke', 'container-ceiling'], allow_extreme=False)

    without_metrics = tkex_runner.parse_args(common)
    with pytest.raises(ValueError, match='database metrics'):
        tkex_runner._validate_execute_prerequisites(without_metrics, profiles)

    with_control_metrics = tkex_runner.parse_args([*common, '--control-database-metrics'])
    tkex_runner._validate_execute_prerequisites(with_control_metrics, profiles)

    manager_profiles = tkex_runner.selected_profiles(['manager-2000'], allow_extreme=True)
    tkex_runner._validate_execute_prerequisites(with_control_metrics, manager_profiles)

    source_profiles = tkex_runner.selected_profiles(['wheelhouse-fanout'], allow_extreme=False)
    with pytest.raises(ValueError, match='source-package.*wheelhouse-smoke'):
        tkex_runner._validate_execute_prerequisites(with_control_metrics, source_profiles)


def test_sandbox_profiles_require_a_private_gongfeng_token_file(tmp_path):
    token_file = tmp_path / 'token'
    token_file.write_text('secret')
    token_file.chmod(0o600)
    common = [
        '--control-url',
        'https://control.example',
        '--application-url',
        'https://application.example',
        '--environment-id',
        'tkex-stress',
        '--taihu-token-file',
        str(token_file),
        '--profile',
        'sandbox-agent-smoke',
        '--tkex-base-url',
        'https://tkex.example',
        '--tkex-api-account',
        'account',
        '--tkex-api-token-file',
        str(token_file),
        '--tkex-cluster-name',
        'cluster',
        '--tkex-project-name',
        'project',
        '--tkex-namespace',
        'namespace',
        '--universal-image',
        'registry.example/universal@sha256:' + '1' * 64,
        '--workload-image',
        'registry.example/workload@sha256:' + '2' * 64,
        '--builder-image',
        'registry.example/builder@sha256:' + '3' * 64,
    ]
    profiles = tkex_runner.selected_profiles(['sandbox-agent-smoke'], allow_extreme=False)

    with pytest.raises(ValueError, match='gongfeng-token-file'):
        tkex_runner._validate_execute_prerequisites(tkex_runner.parse_args(common), profiles)

    tkex_runner._validate_execute_prerequisites(
        tkex_runner.parse_args([*common, '--gongfeng-token-file', str(token_file)]),
        profiles,
    )

    crash_args = [
        *common,
        '--gongfeng-token-file',
        str(token_file),
        '--manager-crash',
        'batch',
    ]
    with pytest.raises(ValueError, match='confirm-manager-pod-delete'):
        tkex_runner._validate_execute_prerequisites(tkex_runner.parse_args(crash_args), profiles)
    tkex_runner._validate_execute_prerequisites(
        tkex_runner.parse_args([*crash_args, '--confirm-manager-pod-delete']),
        profiles,
    )
    tkex_runner._validate_execute_prerequisites(
        tkex_runner.parse_args(
            [
                *crash_args,
                '--manager-crash-approval-file',
                str(tmp_path / 'manager-crash-approved.json'),
            ]
        ),
        profiles,
    )


@pytest.mark.parametrize(
    'observed',
    [
        {},
        {'execution': {'pool_size': '1', 'max_overflow': '0'}},
        {
            'execution': {'pool_size': '2', 'max_overflow': '0'},
            'orchestration': {'pool_size': '2', 'max_overflow': '0'},
        },
    ],
)
def test_large_profiles_require_role_specific_manager_pools(observed):
    profile = tkex_runner.TKEX_LOAD_PROFILES['container-scale']

    with pytest.raises(RuntimeError, match='role-specific DB pools'):
        tkex_runner._validate_large_profile_db_pool(profile, observed)

    tkex_runner._validate_large_profile_db_pool(
        profile,
        {
            'execution': {'pool_size': '1', 'max_overflow': '0'},
            'orchestration': {'pool_size': '4', 'max_overflow': '2'},
        },
    )


def test_db_pool_observations_merge_roles_without_losing_earlier_workloads():
    merged = tkex_runner._merge_observed_db_pool_settings(
        {},
        [
            {
                'db_pool_settings': {
                    'execution': {'pool_size': '1', 'max_overflow': '0'},
                    'orchestration': {'pool_size': '4', 'max_overflow': '2'},
                }
            },
            {'db_pool_settings': {'execution': {'pool_size': '1', 'max_overflow': '0'}}},
        ],
    )

    assert merged == {
        'execution': {'pool_size': '1', 'max_overflow': '0'},
        'orchestration': {'pool_size': '4', 'max_overflow': '2'},
    }


def test_execute_requires_digest_pinned_report_images(tmp_path):
    token_file = tmp_path / 'token'
    token_file.write_text('secret')
    token_file.chmod(0o600)
    common = [
        '--control-url',
        'https://control.example',
        '--application-url',
        'https://application.example',
        '--environment-id',
        'tkex-stress',
        '--taihu-token-file',
        str(token_file),
        '--profile',
        'wheelhouse-smoke',
        '--tkex-base-url',
        'https://tkex.example',
        '--tkex-api-account',
        'account',
        '--tkex-api-token-file',
        str(token_file),
        '--tkex-cluster-name',
        'cluster',
        '--tkex-project-name',
        'project',
        '--tkex-namespace',
        'namespace',
    ]
    profiles = tkex_runner.selected_profiles(['wheelhouse-smoke'], allow_extreme=False)

    with pytest.raises(ValueError, match='image references are required'):
        tkex_runner._validate_execute_prerequisites(tkex_runner.parse_args(common), profiles)

    invalid = tkex_runner.parse_args(
        [
            *common,
            '--universal-image',
            'registry.example/universal:latest',
            '--workload-image',
            'registry.example/workload@sha256:' + '2' * 64,
            '--builder-image',
            'registry.example/builder@sha256:' + '3' * 64,
        ]
    )
    with pytest.raises(ValueError, match='must be digest pinned'):
        tkex_runner._validate_execute_prerequisites(invalid, profiles)
