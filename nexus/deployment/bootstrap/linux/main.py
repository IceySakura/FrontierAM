"""Build a Linux x86_64 Bootstrap bundle using the runtime's locked Nix closure."""

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

from pathlib import Path

from entrypoint import namespace_command


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def record_links(payload):
    """Archive link metadata separately; the launcher restores the verified links.

    Nix closures contain absolute /nix links, host /etc links, and intentionally
    dangling links. None can pass Bootstrap's release-contained link validation.
    """
    links = []
    for directory, dirs, files in os.walk(payload / 'nix', followlinks=False):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            if path.is_symlink():
                links.append({'path': path.relative_to(payload).as_posix(), 'target': os.readlink(path)})
    (payload / 'links.json').write_text(json.dumps(links, indent=2))


def pack_release(payload, output, version):
    archive = output / 'bundles' / f'{version}.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)

    def normalize(info):
        if info.issym() or '__pycache__' in Path(info.name).parts or info.name.endswith('.pyc'):
            return None
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ''
        info.pax_headers = {}
        return info

    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(payload.iterdir()):
                    tar.add(path, arcname=path.name, filter=normalize)
    archive_hash = digest(archive)
    manifest = {
        'schema_version': 1,
        'version': version,
        'os': 'linux',
        'arch': 'x86_64',
        'entrypoint': 'entrypoint.py',
        'entrypoint_sha256': digest(payload / 'entrypoint.py'),
        'files': [{'path': f'bundles/{version}.tar.gz', 'sha256': archive_hash, 'extract': True}],
        'args': [],
    }
    text = json.dumps(manifest, indent=2)
    (output / f'manifest-{version}.json').write_text(text)
    (output / 'manifest.json').write_text(text)
    return {
        'archive': str(archive),
        'archive_sha256': archive_hash,
        'archive_bytes': archive.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=['x86_64'], default='x86_64')
    parser.add_argument('--python-version', choices=['3.12', '3.13', '3.14'], default='3.13')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/bundle')
    args = parser.parse_args()
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        parser.error('Build on Linux x86_64')
    if not all(shutil.which(tool) for tool in ('nix', 'nix-store', 'bwrap', 'uv')):
        parser.error('Build requires nix, nix-store, bwrap and uv')
    output = args.output.resolve() / 'linux-x86_64'
    if output.exists():
        parser.error(f'Refusing to overwrite {output}; choose a new --output')
    output.mkdir(parents=True)
    log_path = output.parent / 'linux-x86_64.build.log'
    print(f'Build log: {log_path}', flush=True)

    def run(command, *, env=None):
        with log_path.open('ab') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Build command failed ({command[0]}); see {log_path}')

    # Match build-runtime's flake and lock, without building an OCI distribution.
    with tempfile.TemporaryDirectory(prefix='nexus-linux-build-') as temporary:
        temp = Path(temporary)
        profile = temp / 'result'
        run(['nix', 'build', '--no-update-lock-file', str(ROOT / 'deployment/runtime'), '--out-link', str(profile)])
        closure = subprocess.check_output(['nix-store', '-qR', str(profile)], text=True).splitlines()
        run(['uv', 'build', '--wheel', '--out-dir', str(temp / 'wheels')])
        wheels = list((temp / 'wheels').glob('nexus_agent-*.whl'))
        if len(wheels) != 1:
            raise RuntimeError('Expected exactly one Nexus wheel')
        wheel = wheels[0]
        version = wheel.name.split('-')[1]
        payload = temp / 'payload'
        runtime = payload / 'nix/runtime'
        (runtime / 'nexus').mkdir(parents=True)
        store = payload / 'nix/store'
        store.mkdir(parents=True)
        print(f'Copying {len(closure)} Nix store objects', flush=True)
        # Store objects may be regular files as well as directories. Preserve links/modes.
        run(['cp', '-a', '--reflink=auto', '--target-directory', str(store), *closure])
        (runtime / 'nix-env').symlink_to(str(profile.resolve()))
        # Copied Nix store directories are initially immutable; only edit this private copy.
        for directory, _, _ in os.walk(payload, followlinks=False):
            os.chmod(directory, Path(directory).stat().st_mode | 0o700)
        build_env = os.environ.copy()
        build_env.update(
            UV_CACHE_DIR=str(temp / 'uv-cache'),
            UV_PYTHON_DOWNLOADS='never',
            UV_INDEX_URL=os.environ.get('UV_INDEX_URL', 'https://mirrors.tencent.com/pypi/simple/'),
            PATH='/nix/runtime/nix-env/bin:' + os.environ['PATH'],
            LD_LIBRARY_PATH='/nix/runtime/nix-env/lib',
        )
        prefix = namespace_command(payload / 'nix', readonly=False)
        uv = '/nix/runtime/nix-env/bin/uv'
        venv = '/nix/runtime/nexus/.venv'
        constraints = temp / 'constraints.txt'
        run([
            'uv',
            'export',
            '--frozen',
            '--no-dev',
            '--extra',
            'runtime',
            '--extra',
            'otel',
            '--no-emit-project',
            '--no-hashes',
            '--output-file',
            str(constraints),
        ])
        run(
            prefix + [uv, 'venv', venv, '--python', f'/nix/runtime/nix-env/bin/python{args.python_version}'],
            env=build_env,
        )
        run(
            prefix + [uv, 'pip', 'install', '-p', venv, '-c', str(constraints), f'{wheel}[runtime,otel]'], env=build_env
        )
        build_env['VIRTUAL_ENV'] = venv
        run(prefix + [venv + '/bin/nexus', 'modules', 'install', '--all', '--runtime'], env=build_env)
        run(prefix + [uv, 'pip', 'check', '-p', venv], env=build_env)
        node = runtime / f'nexus/.venv/lib/python{args.python_version}/site-packages/playwright/driver/node'
        node.unlink()
        node.symlink_to('/nix/runtime/nix-env/bin/node')
        shutil.copytree(ROOT / 'deployment/runtime', runtime / 'nexus/deployment/runtime')
        shutil.copy2(HERE / 'server.py', runtime / 'server.py')
        shutil.copy2(HERE / 'entrypoint.py', payload / 'entrypoint.py')
        (payload / 'entrypoint.py').chmod(0o755)
        run(
            prefix
            + [
                venv + '/bin/python',
                '-I',
                '-c',
                'from nexus.runtime.server.app import asgi_app; print("RUNTIME_IMPORT_OK")',
            ],
            env=build_env,
        )
        record_links(payload)
        print('Packing release', flush=True)
        release = pack_release(payload, output, version)
        record = {
            **release,
            'version': version,
            'python_version': args.python_version,
            'nix_profile': str(profile.resolve()),
            'closure_paths': closure,
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_dirty': bool(
                subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
            ),
            'wheel_sha256': digest(wheel),
        }
        shutil.copy2(wheel, output / wheel.name)
        (output / 'build.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(release, indent=2))


if __name__ == '__main__':
    main()
