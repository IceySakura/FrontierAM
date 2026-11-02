"""Cloud Build regression exercised by the disposable DevCloud cluster harness."""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import io
import json
import re
import zipfile

from pathlib import Path
from typing import Any

import aiohttp

from packaging.version import Version

from nexus.cli.commands.submit import _build_wheel
from nexus.distributed.submission.source_package import build_source_package


# Historical, valid PyPI lock record: the source declaration permits upgrading it.
_OLD_FILELOCK_RECORD = """[[package]]
name = "filelock"
version = "3.29.1"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/packages/1f/f9/f38573ed5844586db374d085911740a501ccfa373b455fc9413f09f85237/filelock-3.29.1.tar.gz", hash = "sha256:d97e6b1b9757569626c58caa07dc4beb1613f4a2938b1e8cc81afca398906c9e", size = 59335 }
wheels = [
    { url = "https://files.pythonhosted.org/packages/4c/a0/614c5fe402fd88951df45f4dda2fa3b4e17a99ecd92340771929169b3b95/filelock-3.29.1-py3-none-any.whl", hash = "sha256:85199dfd706869641b72b2e8955d5416a4b2b7dc4b0e8e6d97b4cc1299a6983b", size = 40750 },
]

"""

_PROBE_SOURCE = """from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import re
import shutil
import sys
from pathlib import Path

from packaging.version import Version
from nexus.core import TaskRunInput, TaskRunOutput, task_registry


class DependencyProbeOutput(TaskRunOutput):
    marker: str
    versions: dict[str, str]
    python: str
    lock_sha256: str
    filelock_wheels: list[str]
    pip_check_exit: int


@task_registry.register("distributed_dependency_build_probe")
async def dependency_build_probe(_task_input: TaskRunInput) -> DependencyProbeOutput:
    versions = {name: importlib.metadata.version(name) for name in ("nexus-agent", "filelock", "harbor")}
    assert Version(versions["filelock"]) >= Version("3.29.4"), versions
    assert versions["harbor"] == "0.20.0", versions
    wheelhouse = Path(sys.prefix).parents[1] / "artifacts" / "wheelhouse"
    lock = wheelhouse / "locks" / "module-host.lock"
    locked = re.search(r"(?m)^filelock==([^\\s;]+)", lock.read_text())
    assert locked and locked[1] == versions["filelock"]
    wheels = sorted(path.name for path in (wheelhouse / "wheels").glob("filelock-*.whl"))
    assert len(wheels) == 1, wheels
    proc = await asyncio.create_subprocess_exec(
        shutil.which("uv") or "uv", "pip", "check", "--no-config", "--python", sys.executable,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await proc.communicate()
    assert proc.returncode == 0, output.decode(errors="replace")
    # Allow the client to capture the same execution in its running state.
    await asyncio.sleep(20)
    return DependencyProbeOutput(
        marker="cloud-dependency-build-ok", versions=versions, python=sys.executable,
        lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(), filelock_wheels=wheels,
        pip_check_exit=proc.returncode,
    )
"""


def stale_filelock_lock(text: str) -> str:
    result, count = re.subn(
        r'(?ms)^\[\[package\]\]\nname = "filelock"\n.*?(?=^\[\[package\]\]|\Z)',
        lambda _match: _OLD_FILELOCK_RECORD,
        text,
    )
    if count != 1:
        raise ValueError(f'expected one filelock lock record, found {count}')
    return result


def incompatible_nexus_wheel(source: Path, target: Path) -> None:
    """Create a valid negative fixture whose declared filelock bound conflicts with Harbor."""
    with zipfile.ZipFile(source) as archive:
        files = {info.filename: archive.read(info) for info in archive.infolist() if not info.is_dir()}
    metadata_path = next(name for name in files if name.endswith('.dist-info/METADATA'))
    metadata, count = re.subn(
        r'(?mi)^Requires-Dist: filelock[^\n]*$',
        'Requires-Dist: filelock<3.29.4',
        files[metadata_path].decode(),
    )
    if count != 1:
        raise ValueError(f'expected one Nexus filelock requirement, found {count}')
    files[metadata_path] = metadata.encode()
    record_path = metadata_path.removesuffix('METADATA') + 'RECORD'
    records = io.StringIO()
    writer = csv.writer(records, lineterminator='\n')
    for name, content in sorted(files.items()):
        if name == record_path:
            continue
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()
        writer.writerow((name, f'sha256={digest}', len(content)))
    writer.writerow((record_path, '', ''))
    files[record_path] = records.getvalue().encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


async def run_dependency_build_contract(
    control_url: str,
    app_url: str,
    *,
    root: Path,
    workspace: Path,
    uploaded_refs: list[dict[str, Any]],
    kubectl: str,
    namespace: str,
    release_name: str,
) -> dict[str, Any]:
    from tests.integration.distributed import run_k8s_dynamic_cluster as cluster

    workspace.mkdir(parents=True)
    nexus_wheel = await asyncio.to_thread(_build_wheel, root, workspace / 'wheel')
    lock_text = stale_filelock_lock((root / 'uv.lock').read_text())
    report: dict[str, Any] = {'input_filelock_version': '3.29.1', 'cases': {}}

    async def upload(path: Path, kind: str) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            ref = await cluster._upload_artifact(
                session,
                control_url,
                path,
                kind=kind,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            )
        uploaded_refs.append(ref)
        return ref

    for case in ('compatible', 'incompatible'):
        project = workspace / case
        project.mkdir()
        project_text = (root / 'pyproject.toml').read_text()
        wheel_path = nexus_wheel.path
        if case == 'incompatible':
            project_text, count = re.subn(r'(?m)^\s*"filelock[^"\n]*",$', '    "filelock<3.29.4",', project_text)
            if count != 1:
                raise ValueError(f'expected one source filelock requirement, found {count}')
            wheel_path = workspace / 'incompatible-wheel' / nexus_wheel.path.name
            await asyncio.to_thread(incompatible_nexus_wheel, nexus_wheel.path, wheel_path)
        (project / 'pyproject.toml').write_text(project_text)
        (project / 'uv.lock').write_text(lock_text)
        (project / 'README.md').write_text('Disposable DevCloud dependency-build regression fixture.\n')
        (project / 'dependency_probe.py').write_text(_PROBE_SOURCE)
        source = await asyncio.to_thread(build_source_package, project, workspace / f'{case}.tar.zst')
        source_ref = await upload(source.path, 'source_package')
        wheel_ref = await upload(wheel_path, 'nexus_wheel')
        lock_ref = await upload(project / 'uv.lock', 'uv_lock')
        payload = cluster._typed_single_live_submission(
            source_ref=source_ref,
            nexus_wheel_ref=wheel_ref,
            uv_lock_ref=lock_ref,
            run_input={
                'imports': ['nexus.extensions.harbor_bridge', 'dependency_probe'],
                'task_name': 'distributed_dependency_build_probe',
                'task_config': {},
            },
            live=False,
        )
        payload['provisioning_timeout_seconds'] = 900
        payload['execution_ttl_seconds'] = 2400
        submitted = await cluster.request_json('POST', f'{control_url}/v1/submissions/single', body=payload)
        execution_id = submitted['execution_id']
        print(f'dependency-build {case}: {execution_id}', flush=True)
        case_report: dict[str, Any] = {
            'execution_id': execution_id,
            'source': source_ref,
            'nexus_wheel': wheel_ref,
            'uv_lock': lock_ref,
        }
        report['cases'][case] = case_report
        if case == 'compatible':
            running = await cluster.poll_execution(
                control_url,
                execution_id,
                lambda item: item['status'] == 'running',
                timeout=1800,
            )
            case_report['running_status'] = running['status']
            finished = await cluster.poll_execution(
                control_url,
                execution_id,
                lambda item: item['status'] == 'succeeded',
                timeout=180,
            )
            result = await cluster.resolved_execution_result(app_url, finished)
            output = result['task_output']
            assert output['marker'] == 'cloud-dependency-build-ok', result
            assert Version(output['versions']['filelock']) >= Version('3.29.4'), output
            assert output['versions']['harbor'] == '0.20.0', output
            assert output['pip_check_exit'] == 0, output
            case_report['result'] = result
        else:
            finished = await cluster.poll_execution(
                control_url,
                execution_id,
                lambda item: item['status'] == 'failed',
                timeout=1800,
            )
        case_report['terminal_status'] = finished['status']
        allocations = await cluster.request_json('GET', f'{control_url}/v1/executions/{execution_id}/allocations')
        assert allocations['items'], allocations
        details = [
            await cluster.request_json('GET', f'{control_url}/v1/allocations/{item["allocation_id"]}')
            for item in allocations['items']
        ]
        builds = [item['dependency_build'] for item in details]
        if case == 'compatible':
            assert all(build['status'] == 'ready' for build in builds), builds
        else:
            assert all(build['status'] == 'failed' for build in builds), builds
            assert all(build['error_code'] == 'dependency_module_lock_failed' for build in builds), builds
            assert all('filelock' in build['error_message'] and 'harbor' in build['error_message'] for build in builds)
            assert all(item['ready_manager_count'] == 0 for item in details), details
        build_ids = sorted({build['dependency_build_id'] for build in builds})
        if any(re.fullmatch(r'bld_[0-9a-f]{32}', build_id) is None for build_id in build_ids):
            raise ValueError('unexpected dependency build ID in acceptance response')
        quoted_ids = ','.join(f"'{build_id}'" for build_id in build_ids)
        publication_sql = (
            "SELECT JSON_OBJECT('dependency_build_id', dependency_build_id, 'status', status, "
            "'wheelhouse_absent', COALESCE(JSON_TYPE(wheelhouse_ref_json), 'NULL') = 'NULL', "
            "'manifest_absent', COALESCE(JSON_TYPE(manifest_json), 'NULL') = 'NULL', "
            "'module_lock_sha256', JSON_UNQUOTE(JSON_EXTRACT(manifest_json, '$.module_lock_sha256'))) "
            'FROM nexus_dist.distributed_dependency_builds '
            f'WHERE dependency_build_id IN ({quoted_ids}) ORDER BY dependency_build_id;'
        )
        publication_output = await asyncio.to_thread(
            cluster._mysql_query, kubectl, namespace, release_name, publication_sql
        )
        publications = [json.loads(line) for line in publication_output.splitlines() if line.startswith('{')]
        assert [item['dependency_build_id'] for item in publications] == build_ids, publications
        if case == 'compatible':
            assert all(item['wheelhouse_absent'] == 0 and item['manifest_absent'] == 0 for item in publications)
            assert all(item['module_lock_sha256'] == output['lock_sha256'] for item in publications), publications
        else:
            assert all(item['wheelhouse_absent'] == 1 and item['manifest_absent'] == 1 for item in publications)
        case_report['publication'] = publications
        case_report['dependency_builds'] = builds
        await cluster._wait_allocations_released(control_url, execution_id, timeout=300)
        released = await cluster.request_json('GET', f'{control_url}/v1/executions/{execution_id}/allocations')
        case_report['released_allocations'] = [
            {'allocation_id': item['allocation_id'], 'status': item['status']} for item in released['items']
        ]
    return report
