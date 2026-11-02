"""Run real Manager pause/recovery acceptance against an EMPTY, disposable MySQL DB.

Set NEXUS_LIVENESS_TEST_DATABASE_URL and run inside the DevCloud test container.
Creates schema and test data; the caller must delete the disposable database/Pod.
No production service or database may be used. Logs go to /tmp/liveness-acceptance.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time

from pathlib import Path
from datetime import timedelta

from aiohttp import web
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from tests.unit.distributed.submission.test_typed_submission_repository import _seed_environment

from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner
from nexus.distributed.config import DistributedConfig
from nexus.distributed.gateways.application.app import create_application_app
from nexus.distributed.persistence.db import init_database
from nexus.distributed.persistence.orm import (
    DistributedConfigOverrideRow,
    EnvironmentAllocationRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
)
from nexus.distributed.repositories.reconciler import ReconcilerRepository
from nexus.distributed.repositories.scheduler import SchedulerRepository
from nexus.distributed.repositories.submission import SubmissionRepository
from nexus.distributed.submission.specs import SingleSubmission
from nexus.distributed.utils.clock import utcnow


async def main() -> None:
    url = (
        make_url(os.environ['NEXUS_LIVENESS_TEST_DATABASE_URL'])
        .set(database='liveness_runtime_acceptance')
        .render_as_string(hide_password=False)
    )
    config = DistributedConfig(
        database_url=url, config_store_required=False, environment_enabled_providers=['kubernetes']
    )
    await init_database(config)
    engine = create_async_engine(url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    output = Path('/tmp/liveness-acceptance')
    output.mkdir(exist_ok=True)
    application = web.AppRunner(create_application_app(session_factory=factory, config=config))
    await application.setup()
    await web.TCPSite(application, '127.0.0.1', 18081).start()
    signer = ArtifactContentTicketSigner(
        keys=config.artifact_ticket_keyring(),
        current_key_id=config.artifact_upload_ticket_current_key_id,
        ttl=timedelta(hours=1),
    )
    processes = []
    records = []
    logs = []
    try:
        async with factory() as session:
            await _seed_environment(
                session,
                environment_id='env-dynamic',
                allocation_type='dynamic',
                provider_type='kubernetes',
                environment_limits={},
                grant_limits={},
            )
            session.add(
                DistributedConfigOverrideRow(
                    role='instance-manager',
                    identity='*',
                    config_json={
                        'manager_self_fence_timeout': 30,
                        'manager_executor_hang_timeout': 180,
                        'manager_dead_ttl': 45,
                    },
                )
            )
            await session.commit()
        for index, name in enumerate(('recover', 'expire', 'self-fence')):
            submission = SingleSubmission.model_validate({
                'submission_version': 1,
                'idempotency_key': name,
                'environment_id': 'env-dynamic',
                'code': {'mode': 'config_only'},
                'resources': {
                    'managers': {
                        'count': 1,
                        'cpu_cores': 1,
                        'memory_gib': 1,
                        'slots': 1,
                        'liveness': {
                            'self_fence_timeout_seconds': 90,
                            'executor_hang_timeout_seconds': 180,
                            'dead_timeout_seconds': 120,
                        },
                    }
                },
                'run_input': {
                    'imports': ['tests.integration.distributed.fixtures'],
                    'task_name': 'distributed_sleep_echo',
                    'task_config': {'index': index, 'message': name, 'sleep_seconds': 80 if name == 'recover' else 600},
                },
            })
            async with factory() as session:
                result = await SubmissionRepository(session, config=config).create(
                    submission,
                    actor_principal_id='principal-submit',
                    owner_principal_id='principal-submit',
                    owner_tenant='tenant',
                    now=utcnow(),
                )
                execution = await session.get(ExecutionRow, result.execution.execution_id)
                allocation = await session.get(EnvironmentAllocationRow, execution.allocation_id)
                # The dedicated Pod runs bootstrap processes directly in place of the Kubernetes allocator.
                allocation.status = 'ready'
                allocation.ready_manager_count = 1
                execution.first_schedulable_at = utcnow()
                spec = allocation.allocation_spec_json
                await session.commit()
                records.append((name, execution.execution_id, allocation.allocation_id))
            env = {
                **os.environ,
                'NEXUS_DIST_DATABASE_URL': url,
                'NEXUS_DIST_CONFIG_STORE_REQUIRED': '0',
                'NEXUS_DIST_MANAGER_OWNER_PRINCIPAL_ID': 'principal-submit',
                'NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL': 'http://127.0.0.1:18081',
                'NEXUS_DIST_ARTIFACT_CONTENT_TICKET': signer.issue(
                    execution_id=execution.execution_id,
                    top_level_execution_id=execution.execution_id,
                    environment_id='env-dynamic',
                    allocation_id=allocation.allocation_id,
                    operations=['artifact:read', 'run_result:write'],
                    artifact_ids=[],
                    now=utcnow(),
                ),
                'NEXUS_DIST_AIOMONITOR_ENABLED': 'false',
                'NEXUS_DIST_ALLOCATION_SPEC_JSON': json.dumps(spec),
                'POD_NAME': f'{allocation.allocation_id}-0',
                'PYTHONPATH': str(Path.cwd()),
                'NEXUS_OTEL_ENABLED': 'false',
            }
            log = (output / f'{name}.log').open('w')
            logs.append(log)
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                '-m',
                'nexus.distributed.instance_manager.bootstrap',
                '--allocation-id',
                allocation.allocation_id,
                '--environment-id',
                'env-dynamic',
                '--pool-id',
                allocation.pool_id,
                '--workload',
                'execution',
                '--allocation-spec-hash',
                allocation.allocation_spec_hash,
                '--require-allocation-liveness-v1',
                '--cache-root',
                str(output / 'cache'),
                '--port',
                str(18082 + index),
                env=env,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
            processes.append(proc)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            async with factory() as session:
                await SchedulerRepository(
                    session, shard_count=config.shard_count, manager_healthy_ttl=config.manager_healthy_ttl
                ).claim_and_assign(shard_ids=list(range(config.shard_count)), now=utcnow())
                await session.commit()
                states = dict(
                    (await session.execute(select(ExecutionRow.execution_id, ExecutionRow.status))).tuples().all()
                )
            if all(states.get(execution_id) == 'running' for _, execution_id, _ in records):
                break
            assert all(proc.returncode is None for proc in processes), 'Manager exited before assignments started'
            await asyncio.sleep(0.5)
        else:
            raise AssertionError(f'Managers did not run all assignments: {states}')
        print('All three real Managers registered and tasks RUNNING', flush=True)
        for name, _, _ in records:
            content = (output / f'{name}.log').read_text()
            assert 'self_fence=90.0s executor_hang=180.0s dead=120.0s' in content, content[-3000:]
        for proc in processes:
            proc.send_signal(signal.SIGSTOP)
        paused = time.monotonic()
        resumed_recover = resumed_fence = False
        while time.monotonic() - paused < 123:
            elapsed = time.monotonic() - paused
            if elapsed >= 55 and not resumed_recover:
                async with factory() as session:
                    assert await ReconcilerRepository(session).reconcile_manager_dead(now=utcnow()) == 0
                    await session.commit()
                processes[0].send_signal(signal.SIGCONT)
                resumed_recover = True
                print('55s pause: default 45s passed, all tasks preserved; resuming recovery Manager', flush=True)
            if elapsed >= 95 and not resumed_fence:
                processes[2].send_signal(signal.SIGCONT)
                resumed_fence = True
                print('95s pause: resuming third Manager to verify irreversible self-fence', flush=True)
            await asyncio.sleep(0.5)
        async with factory() as session:
            changed = await ReconcilerRepository(session).reconcile_manager_dead(now=utcnow())
            await session.commit()
            states = dict(
                (await session.execute(select(ExecutionRow.execution_id, ExecutionRow.status))).tuples().all()
            )
            assert states[records[0][1]] == 'succeeded', states
            assert states[records[1][1]] == 'lost', states
            assert states[records[2][1]] == 'lost', states
            live = await session.get(InstanceManagerLivenessRow, f'{records[1][2]}-0')
            assert live.fenced_at is not None
        assert processes[2].returncode is not None, 'self-fenced Manager did not exit'
        processes[1].send_signal(signal.SIGCONT)
        await asyncio.wait_for(processes[1].wait(), 15)
        summary = {
            'states': {name: states[execution_id] for name, execution_id, _ in records},
            'effective_timeouts': [90, 180, 120],
            'pause_seconds': round(time.monotonic() - paused, 2),
            'reconciled': changed,
            'database_scope_overridden': True,
        }
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        for proc in processes:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
        for log in logs:
            log.close()
        await application.cleanup()
        await engine.dispose()


if __name__ == '__main__':
    asyncio.run(main())
