"""Run the locked numerical evaluator in a cancellable, owned child process."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import uuid

from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

from .assets import IMAGE_ID, LockedAssets
from .local_calibration import qualified_baseline


OWNER_LABEL = 'nexus.frontieram.owner'


class OwnedCleanupError(RuntimeError):
    def __init__(self, errors: list[dict[str, str]]):
        self.errors = errors
        super().__init__('; '.join(error['category'] for error in errors))


def docker_command(config_path: Path, arguments: list[str]) -> int:
    """A trusted Docker CLI shim records only resources created by this grading run."""
    config = json.loads(config_path.read_text())
    command = arguments[2:] if arguments[:1] == ['--host'] else arguments
    ledger = Path(config['ledger'])

    def record(kind, value):
        with ledger.open('a') as out:
            out.write(json.dumps({'kind': kind, 'value': value}) + '\n')

    if command and command[0] in {'run', 'create'} and '--name' in command:
        record('container', command[command.index('--name') + 1])
        offset = 2 if arguments[:1] == ['--host'] else 0
        arguments = [
            *arguments[: offset + 1],
            '--label',
            f'{OWNER_LABEL}={config["owner"]}',
            *arguments[offset + 1 :],
        ]
    if command[:1] == ['commit']:
        result = subprocess.run([config['docker_path'], *arguments], stdout=subprocess.PIPE, check=False)
        if result.returncode == 0:
            image = result.stdout.decode().strip()
            if image.startswith('sha256:'):
                record('image', image)
        sys.stdout.buffer.write(result.stdout)
        return result.returncode
    os.execv(config['docker_path'], [config['docker_path'], *arguments])
    raise AssertionError('execv returned')


def worker(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    root = Path(config['output_dir'])
    assets = LockedAssets(config['source_root'])
    assets.verify_private_data()
    c = assets.controller
    wrapper = root / 'owned-docker'
    wrapper.write_text(
        f'#!{sys.executable}\n'
        'import sys\nfrom pathlib import Path\n'
        'from nexus.extensions.tasks.frontieram_offline.grading import docker_command\n'
        f'raise SystemExit(docker_command(Path({str(config_path)!r}), sys.argv[1:]))\n'
    )
    wrapper.chmod(0o700)
    docker = c.Docker(wrapper, config['docker_host'])
    try:
        build = c.offline_build(docker, assets.image_id, root)
        report = c.evaluation(
            SimpleNamespace(manifest=assets.manifest), docker, assets.image_id, root, Path(build['artifact'])
        )
        baseline = qualified_baseline(assets, docker)
        score = (
            assets.scoring.score_submission(
                json.loads(assets.manifest.read_text())['cases'], report['aggregates'], baseline['aggregates']
            )
            if baseline is not None
            else {'score': None, 'status': 'local_calibration_required'}
        )
        c.save(root / 'score.json', score)
        result = {
            'exit_status': 'finished',
            'score': score['score'],
            'score_status': score['status'],
            'valid_runs': sum(r['valid'] is True for r in report['runs']),
            'total_runs': len(report['runs']),
        }
    except c.SubmissionFailure as error:
        result = {'exit_status': 'finished', 'score': 0.0, 'reason': type(error).__name__}
    except Exception as error:
        result = {'exit_status': 'runtime_error_by_env', 'score': None, 'reason': type(error).__name__}
    c.save(root / 'grading_result.json', result)


async def cleanup_owned(config: dict) -> None:
    """Discover by owner as well as ledger, then attempt every owned resource."""
    ledger = Path(config['ledger'])
    errors: list[dict[str, str]] = []
    resources: dict[str, set[str]] = {'container': set(), 'image': set()}
    try:
        lines = ledger.read_text().splitlines() if ledger.exists() else []
    except OSError:
        lines = []
        errors.append({'category': 'owned_resource_ledger_read_failed'})
    for line in lines:
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('invalid ledger record')
            if record.get('kind') in resources:
                value = record.get('value')
                if not isinstance(value, str) or not value:
                    raise ValueError('invalid ledger value')
                resources[record['kind']].add(value)
        except (ValueError, TypeError):
            errors.append({'category': 'owned_resource_ledger_record_invalid'})

    # A commit can succeed just before its shim is killed, before recording its image ID.
    # The inherited owner label closes that window, even with no ledger at all.
    for kind in resources:
        try:
            code, stdout, _ = await _docker_capture(
                config,
                kind,
                'ls',
                '--all',
                '--quiet',
                '--no-trunc',
                '--filter',
                f'label={OWNER_LABEL}={config["owner"]}',
            )
            if code:
                raise RuntimeError('owned_resource_discovery_failed')
            resources[kind].update(stdout.decode().split())
        except Exception as error:
            errors.append({'category': _cleanup_category(error, 'discovery'), 'kind': kind})

    deleted: set[tuple[str, str]] = set()
    for kind, values in resources.items():
        for value in sorted(values):
            if value == IMAGE_ID:
                continue
            phase = 'inspection'
            try:
                code, stdout, stderr = await _docker_capture(config, 'inspect', '--type', kind, value)
                if code:
                    if b'no such' in stderr.lower():
                        continue
                    raise RuntimeError('owned_resource_inspection_failed')
                info = json.loads(stdout)[0]
                if (info.get('Config', {}).get('Labels') or {}).get(OWNER_LABEL) != config['owner']:
                    continue
                identity = info['Id']
                if not isinstance(identity, str) or not identity:
                    raise ValueError('invalid resource identity')
                if identity == IMAGE_ID or (kind, identity) in deleted:
                    continue
                phase = 'cleanup'
                arguments = ['rm', '--force', identity] if kind == 'container' else ['image', 'rm', identity]
                code, _, stderr = await _docker_capture(config, *arguments)
                if code and b'no such' not in stderr.lower():
                    raise RuntimeError('owned_resource_cleanup_failed')
                deleted.add((kind, identity))
            except Exception as error:
                errors.append({'category': _cleanup_category(error, phase), 'kind': kind, 'resource': value})
    if errors:
        raise OwnedCleanupError(errors)


def _cleanup_category(error: Exception, phase: str) -> str:
    if isinstance(error, TimeoutError):
        return f'owned_resource_{phase}_timeout'
    return f'owned_resource_{phase}_failed'


async def _docker_capture(config: dict, *arguments: str) -> tuple[int, bytes, bytes]:
    child = await asyncio.create_subprocess_exec(
        config['docker_path'],
        '--host',
        config['docker_host'],
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=15)
    except BaseException:
        if child.returncode is None:
            with suppress(ProcessLookupError):
                child.kill()
        await child.wait()
        raise
    assert child.returncode is not None
    return child.returncode, stdout, stderr


async def _stop_worker(child) -> None:
    if child is None or child.returncode is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(child.wait(), timeout=3)
    # Also stop descendants that outlive the worker after its SIGTERM handler exits.
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    await child.wait()


async def _finish_grading(child, spawn, config: dict) -> None:
    errors: list[dict[str, str]] = []
    if child is None and spawn is not None:
        # The original launch error is propagated by grade after resource cleanup.
        with suppress(Exception):
            child = await spawn
    try:
        await _stop_worker(child)
    except Exception:
        errors.append({'category': 'grading_worker_termination_failed'})
    try:
        await cleanup_owned(config)
    except OwnedCleanupError as error:
        errors.extend(error.errors)
    except Exception as error:
        errors.append({'category': 'owned_resource_cleanup_failed', 'error_type': type(error).__name__})
    if errors:
        raise OwnedCleanupError(errors)


async def grade(runtime, output_dir: Path, *, timeout: float = 900) -> dict:
    config = {
        'source_root': str(runtime.assets.source_root),
        'output_dir': str(output_dir),
        'docker_path': str(runtime.docker.prefix[0]),
        'docker_host': runtime.docker.prefix[2],
        'ledger': str(output_dir / 'grading_owned_resources.jsonl'),
        'owner': uuid.uuid4().hex,
    }
    config_path = output_dir / 'grading_config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    child = spawn = None
    failure: BaseException | None = None
    result = None
    with (output_dir / 'grading.log').open('wb') as log:
        try:
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    '-m',
                    'nexus.extensions.tasks.frontieram_offline.grading',
                    str(config_path),
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            )
            child = await asyncio.shield(spawn)
            await asyncio.wait_for(child.wait(), timeout=timeout)
            result_path = output_dir / 'grading_result.json'
            if child.returncode or not result_path.exists():
                result = {'exit_status': 'runtime_error_by_env', 'score': None, 'reason': 'grading_worker_failed'}
            else:
                result = json.loads(result_path.read_text())
        except BaseException as error:
            failure = error

        cleanup = asyncio.create_task(_finish_grading(child, spawn, config))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:
                failure = error
            except Exception:
                break
        try:
            cleanup.result()
        except OwnedCleanupError as error:
            try:
                (output_dir / 'grading_cleanup_error.json').write_text(
                    json.dumps({'owner': config['owner'], 'errors': error.errors}, indent=2) + '\n'
                )
            except OSError as write_error:
                error.add_note(f'Unable to write cleanup diagnostic: {type(write_error).__name__}')
            if not isinstance(failure, asyncio.CancelledError):
                failure = error
    if failure is not None:
        raise failure
    assert result is not None
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    worker(parser.parse_args().config)
