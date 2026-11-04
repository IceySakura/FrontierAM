"""Seal a host-local runtime without modifying the checked-in task packages."""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_EXP = 'sha256:f203391e0381bdca377548961998bbbcac9bfd2212cfd00ffb77c4b31ed230f2'
OLD_CONIC = 'sha256:a0d0870fae06d94f210cee1493b4a0c6cb5825c644ca9f81f28830be33008724'
OLD_CODEX = '58a91f6ea19bea477ae3a2ec48d619f0039afe75d3476260747e11d201fa7cea'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def resolve_codex(path):
    path = Path(path).resolve(strict=True)
    magic = {b'\x7fELF', b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xca\xfe\xba\xbe'}
    if path.read_bytes()[:4] in magic:
        return path
    # npm's codex entry is a JS wrapper; pin the native executable it packages.
    version = subprocess.check_output([str(path), '--version'], text=True).strip()
    package = path.parent.parent
    roots = [package, *package.parent.glob('codex-*')]
    matches = set()
    for root in roots:
        for candidate in root.rglob('codex'):
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                continue
            with candidate.open('rb') as stream:
                if stream.read(4) not in magic:
                    continue
            try:
                value = subprocess.check_output(
                    [str(candidate), '--version'], text=True, stderr=subprocess.DEVNULL, timeout=10
                ).strip()
            except (OSError, subprocess.SubprocessError):
                continue
            if value == version:
                matches.add(candidate.resolve())
    if len(matches) != 1:
        raise ValueError('Cannot uniquely locate native Codex; provide its executable path explicitly')
    return matches.pop()


def override_conic_manifest(task: Path, manifest: Path) -> dict:
    """Swap in a manifest that differs only in per-case time limits, re-chaining the hashes that name it.

    The original verifier checks the manifest against the score anchor and the package lock; both are updated in this
    sealed copy only. Instances, inputs, checks and code stay byte-identical to the original package.
    """
    target = task / 'tests/private/data/manifest.json'
    original, new = json.loads(target.read_text()), json.loads(manifest.read_text())
    keep = lambda m: [{k: v for k, v in c.items() if k != 'time_limit_seconds'} for c in m['instances']]  # noqa: E731
    if keep(original) != keep(new) or new.get('scoring_version') != original.get('scoring_version'):
        raise SystemExit('A conic manifest override may change time_limit_seconds only')
    if any(not 1 <= c['time_limit_seconds'] <= 300 for c in new['instances']):
        raise SystemExit('Time limits must stay within the original 1-300 second range')
    shutil.copyfile(manifest, target)
    anchor_path = task / 'tests/private/score-anchor.json'
    anchor = json.loads(anchor_path.read_text())
    anchor['evaluation_manifest_sha256'] = sha(target)
    write(anchor_path, anchor)
    lock_path = task / 'tests/private/package-lock.json'
    lock = json.loads(lock_path.read_text())
    for relative in ('private/data/manifest.json', 'private/score-anchor.json'):
        path = task / 'tests' / relative
        lock['files'][relative] = {'sha256': sha(path), 'size_bytes': path.stat().st_size}
    write(lock_path, lock)
    return {
        'source': str(manifest.resolve()),
        'manifest_sha256': sha(target),
        'original_manifest_sha256': sha(ROOT / 'tasks/conic/original/tests/private/data/manifest.json'),
        'changed_limits': sum(a['time_limit_seconds'] != b['time_limit_seconds']
                              for a, b in zip(original['instances'], new['instances'])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docker', default=shutil.which('docker'))
    parser.add_argument('--docker-host', default=os.environ.get('DOCKER_HOST', 'unix:///var/run/docker.sock'))
    parser.add_argument(
        '--machine',
        type=Path,
        default=Path(os.environ.get('FRONTIERAM_MACHINE', '/etc/frontieram/machine.json')),
        help='layout from ./bench host setup; images are copied by ID into every per-slot grading daemon',
    )
    parser.add_argument('--codex', required=True, help='Codex executable or its npm launcher')
    parser.add_argument('--exponential-image', default='frontieram-exponential:port-v1')
    parser.add_argument('--conic-image', default='frontieram-conic:success-v2-port-v1')
    parser.add_argument('--conic-manifest', type=Path, help='held-out manifest with recalibrated time limits only')
    args = parser.parse_args()
    if not args.docker_host.startswith('unix://'):
        parser.error('Use a local Unix Docker socket')
    if not args.docker:
        parser.error('Docker CLI was not found')
    docker = Path(args.docker).resolve(strict=True)
    codex = resolve_codex(args.codex)
    version = subprocess.check_output([str(codex), '--version'], text=True).strip()
    help_text = subprocess.check_output([str(codex), '--help'], text=True)
    exec_help = subprocess.check_output([str(codex), 'exec', '--help'], text=True)
    for option in ('--ignore-user-config', '--ignore-rules', '--strict-config', '--no-daemon'):
        if option not in help_text + exec_help:
            parser.error(f'This Codex build lacks required option {option}; run with a compatible pinned build')
    prefix = [str(docker), '--host', args.docker_host]
    images = {}
    for name in ('exponential', 'conic'):
        image = json.loads(subprocess.check_output(prefix + ['image', 'inspect', getattr(args, name + '_image')]))[0]
        if image['Os'] != 'linux' or image['Architecture'] not in ('amd64', 'arm64'):
            parser.error('Only native Linux amd64/arm64 numerical images are supported')
        images[name] = {'id': image['Id'], 'architecture': image['Architecture']}
    if images['exponential']['architecture'] != images['conic']['architecture']:
        parser.error('The two numerical images must use the same architecture')
    if not args.machine.exists():
        parser.error(f'{args.machine} is missing; run ./bench host setup first')
    grade_hosts = json.loads(args.machine.read_text())['docker_hosts']['grade']
    for host in grade_hosts:
        grade_prefix = [str(docker), '--host', host]
        for name in ('exponential', 'conic'):
            image_id = images[name]['id']
            if subprocess.run(grade_prefix + ['image', 'inspect', image_id], capture_output=True).returncode:
                save = subprocess.Popen(
                    prefix + ['image', 'save', getattr(args, name + '_image')], stdout=subprocess.PIPE
                )
                subprocess.run(grade_prefix + ['image', 'load', '--quiet'], stdin=save.stdout, check=True)
                if save.wait():
                    parser.error(f'Could not export the numerical image to {host}')
            loaded = json.loads(subprocess.check_output(grade_prefix + ['image', 'inspect', image_id]))[0]['Id']
            if loaded != image_id:
                parser.error(f'{host} holds a different image under the locked ID')
    server = json.loads(subprocess.check_output(prefix + ['info', '--format', '{{json .}}']))
    native_arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(server['Architecture'], server['Architecture'])
    if native_arch != images['exponential']['architecture']:
        parser.error('Numerical scoring requires native architecture; emulated image builds are not native validation')
    if server.get('CgroupVersion') != '2':
        parser.error('The verifier requires cgroup v2')
    if server.get('MemTotal', 0) < 10 * 2**30:
        parser.error('Allocate at least 10 GiB to the Docker host for the 8 GiB verifier')
    runtime = Path(os.environ.get('FRONTIERAM_RUNTIME') or ROOT / '.runtime').resolve()
    if runtime.exists():
        parser.error('.runtime already exists; preserve completed runs and rename it before preparing a new profile')
    runtime.mkdir()
    shutil.copytree(
        ROOT / 'nexus',
        runtime / 'nexus',
        ignore=shutil.ignore_patterns('.venv', '__pycache__', '.pytest_cache', '.ruff_cache', 'outputs', '.git'),
    )
    shutil.copytree(ROOT / 'tasks/exponential', runtime / 'tasks/exponential')
    source = runtime / 'tasks/exponential/docker_eval'
    binaries = source / 'runtime/bin'
    binaries.mkdir(parents=True)
    (binaries / 'docker').symlink_to(docker)
    (binaries / 'codex-frozen').symlink_to(codex)
    lock = source / 'image_lock.json'
    old_lock_hash = sha(lock)
    data = json.loads(lock.read_text())
    data.update(
        image_id=images['exponential']['id'],
        architecture=images['exponential']['architecture'],
        image_tag=args.exponential_image,
        port_version='portable-v1',
    )
    write(lock, data)
    extensions = runtime / 'nexus/nexus/extensions/tasks'
    asset_path = extensions / 'frontieram_offline/assets.py'
    code = asset_path.read_text().replace(OLD_EXP, images['exponential']['id'])
    code = code.replace(OLD_CODEX, sha(codex)).replace(old_lock_hash, sha(lock))
    asset_path.write_text(code)
    asset_path = extensions / 'frontieram_conic/assets.py'
    asset_path.write_text(asset_path.read_text().replace(OLD_CONIC, images['conic']['id']))
    # Sealed copies: editing the repository can never change files under a running or queued job.
    shutil.copytree(ROOT / 'tasks/conic/original', runtime / 'tasks/conic/original')
    shutil.copytree(ROOT / 'tasks/conic/reference_submission', runtime / 'tasks/conic/reference_submission')
    task = runtime / 'tasks/conic/original'
    manifest_override = override_conic_manifest(task, args.conic_manifest) if args.conic_manifest else None
    public_tar = runtime / 'conic-public.tar'
    architecture = images['conic']['architecture']
    port_note = (
        '\n\n## Host migration profile portable-v1\n'
        f'This run uses Linux {architecture}. This architecture statement overrides the original ARM64 wording. '
        'The offline time and memory limits, data and mathematical checks remain unchanged. '
        'Host Codex and numerical image hashes are sealed before trials. '
        'Agent CLIs are host-side only; the portable candidate image does not ship authenticated agent CLIs. '
        'Only the workspace execution tool is available. '
        'Several isolated attempts share this host, each on its own dedicated physical core: wherever the task '
        'says CPU 0 or CPU0, read "the one CPU this container is pinned to". Its index is assigned per run and may '
        'differ between development and verification; never hard-code a CPU number, and use the inherited affinity '
        '(for example os.sched_getaffinity(0)) if you need it.\n'
    )
    resources = json.loads((task / 'environment/resources.yaml').read_text())
    resources['resources'] = [r for r in resources['resources'] if r['id'] != 'agent-clis']
    with tarfile.open(public_tar, 'w', format=tarfile.PAX_FORMAT) as archive:

        def clean(member):
            member.uid = member.gid = 1000
            member.uname = member.gname = ''
            member.mtime = 0
            return member

        def add_text(name, value):
            payload = value.encode()
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(payload), 0o644
            archive.addfile(clean(member), io.BytesIO(payload))

        add_text('PILOT.md', (ROOT / 'tasks/conic/PILOT_BASE.md').read_text() + port_note)
        add_text('resources.yaml', json.dumps(resources, indent=2) + '\n')
        archive.add(task / 'instruction.md', arcname='INSTRUCTION.md', filter=clean)
        archive.add(task / 'environment/public-dev', arcname='public-dev', filter=clean)
    profile = json.loads((ROOT / 'tasks/conic/profile_template.json').read_text())
    profile.update(
        task_root=str(task),
        public_tar=str(public_tar),
        public_tar_sha256=sha(public_tar),
        image_id=images['conic']['id'],
        port_version='portable-v1',
    )
    if manifest_override:
        for relative in ('tests/private/data/manifest.json', 'tests/private/score-anchor.json',
                         'tests/private/package-lock.json'):
            if relative in profile['locked_files']:
                profile['locked_files'][relative] = sha(task / relative)
        profile['manifest_override'] = manifest_override
    write(runtime / 'conic-profile.json', profile)
    portable = {
        'schema_version': 1,
        'port_version': 'portable-v1',
        'source_root': str(source),
        'profile_path': str(runtime / 'conic-profile.json'),
        'profile_sha256': sha(runtime / 'conic-profile.json'),
        'docker_path': str(docker),
        'docker_host': args.docker_host,
        'grade_docker_hosts': grade_hosts,
        'conic_reference_root': str(runtime / 'tasks/conic/reference_submission'),
        'codex_path': str(codex),
        'codex_version': version,
        'codex_sha256': sha(codex),
        'images': images,
        'calibration_required': True,
        'architecture_override_prompt': port_note,
        'code_sha256': {
            str(p.relative_to(runtime)): sha(p)
            for directory in ('frontieram_offline', 'frontieram_conic')
            for p in (extensions / directory).glob('*.py')
        },
    }
    write(source / 'PORTABLE.json', {'profile': str(runtime / 'runtime-lock.json'), 'calibration_required': True})
    write(runtime / 'runtime-lock.json', portable)
    (runtime / 'READY').write_text('sealed\n')
    print(
        json.dumps(
            {
                'prepared': True,
                'architecture': architecture,
                'codex_version': version,
                'next': './bench smoke --task both --output outputs/first-smoke',
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
