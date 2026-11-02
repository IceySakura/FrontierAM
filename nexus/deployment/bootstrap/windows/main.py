"""Build a standalone Nexus wheel bundle from the module-validated environment."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from nexus.core.module_meta import discover_all_modules
from nexus.utils.platform import is_windows


try:
    import tomllib
except ImportError:
    import tomli as tomllib


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD_LOG: Path | None = None


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def run(command, *, env=None, cwd=ROOT):
    assert BUILD_LOG is not None
    with BUILD_LOG.open('ab') as log:
        subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def dependency_distributions():
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    project = config['project']
    overrides = [Requirement(raw) for raw in config.get('tool', {}).get('uv', {}).get('override-dependencies', [])]
    requirements = project['dependencies'] + project['optional-dependencies']['runtime']
    requirements += [dep for module in discover_all_modules() for dep in module.base_deps + module.runtime_extras]
    pending = [(requirement, '') for requirement in requirements]
    checked = set()
    distributions = {}
    while pending:
        raw, extra = pending.pop()
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({'extra': extra}):
            continue
        dist = importlib.metadata.distribution(requirement.name)
        name = canonicalize_name(dist.metadata['Name'])
        if name == 'nexus-agent':
            continue
        direct_url = json.loads(dist.read_text('direct_url.json') or '{}')
        if direct_url.get('dir_info', {}).get('editable'):
            raise RuntimeError(f'Cannot bundle editable dependency: {name}; install a regular distribution first')
        if dist.files is None:
            raise RuntimeError(f'Installed file list is missing for dependency: {name}')
        # uv resolves these dependencies using the project's overrides, so validate
        # the installed version against the same effective requirements.
        effective = [
            override
            for override in overrides
            if canonicalize_name(override.name) == name
            and (not override.marker or override.marker.evaluate({'extra': extra}))
        ] or [requirement]
        for expected in effective:
            if expected.specifier and not expected.specifier.contains(dist.version, prereleases=True):
                raise RuntimeError(f'Installed version does not satisfy {expected}: {dist.version}')
        distributions[name] = dist
        for context in {''} | requirement.extras:
            if (name, context) in checked:
                continue
            checked.add((name, context))
            pending.extend((value, context) for value in dist.requires or [])
    return distributions


def ignore_python_files(directory, names):
    # Copy the interpreter, not any unrelated packages installed in its base environment.
    if Path(directory).name == 'site-packages':
        return names
    return shutil.ignore_patterns('__pycache__', '*.pyc')(directory, names)


def payload_files(root):
    # Include the root entrypoint, but not wheels or build metadata.
    paths = [root / 'entrypoint.py']
    paths += [path for folder in ('python', 'tools') for path in sorted((root / folder).rglob('*'))]
    return [
        {'path': path.relative_to(root).as_posix(), 'sha256': digest(path)}
        for path in paths
        if path.is_file() and path.suffix != '.pyc'
    ]


def pack_release(payload, output, *, version, commit, arch):
    """Publish one verified tar.gz and a Bootstrap manifest; exclude build artifacts."""
    if not re.fullmatch(r'[0-9A-Za-z][0-9A-Za-z.+_-]*', version) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Expected a safe version and a full Git commit SHA')
    files = payload_files(payload)
    archive_path = output / 'bundles' / f'{version}.tar.gz'
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(f'Refusing to overwrite archive: {archive_path}')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=archive_path.parent, suffix='.tar.gz.tmp', delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                    for item in files:
                        path = payload / item['path']
                        if path.is_symlink() or not path.resolve().is_relative_to(payload.resolve()):
                            raise ValueError(f'Not a regular in-bundle file: {item["path"]}')
                        info = archive.gettarinfo(str(path), arcname=item['path'])
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ''
                        info.pax_headers = {}
                        with path.open('rb') as stream:
                            archive.addfile(info, stream)
        temporary.rename(archive_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    archive_hash = digest(archive_path)
    manifest = {
        'schema_version': 1,
        'version': version,
        'os': 'windows',
        'arch': arch,
        'entrypoint': 'python/python.exe',
        'entrypoint_sha256': digest(payload / 'python/python.exe'),
        'files': [
            {
                'path': archive_path.relative_to(output).as_posix(),
                'sha256': archive_hash,
                'extract': True,
            }
        ],
        'args': ['-I', '-B', '-u', '${RELEASE_ROOT}/entrypoint.py'],
    }
    versioned_manifest = output / f'manifest-{version}.json'
    versioned_manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    manifest_path = output / 'manifest.json'
    temporary_manifest = output / 'manifest.json.tmp'
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temporary_manifest.replace(manifest_path)
    return {
        'manifest_sha256': digest(manifest_path),
        'archive': str(archive_path),
        'archive_sha256': archive_hash,
        'archive_bytes': archive_path.stat().st_size,
        'files': len(files),
    }


def main():
    global BUILD_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=['x86_64'], default='x86_64', help='Target architecture; native builds only')
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'build' / 'bundle',
        help='Output root; target platform directory must not exist',
    )
    args = parser.parse_args()
    if not is_windows() or platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('This builder currently supports Windows x86_64 only')
    output = args.output.resolve() / f'windows-{args.arch}'
    if output.exists():
        parser.error(f'Refusing to overwrite existing output: {output}; choose a new --output')
    output.mkdir(parents=True)
    BUILD_LOG = output.parent / f'{output.name}.build.log'
    BUILD_LOG.write_text('', encoding='utf-8')
    print(f'Build log: {BUILD_LOG}', flush=True)
    uv = shutil.which('uv')
    if not uv:
        raise RuntimeError('uv is required for bundle preparation')
    run([uv, 'run', '--no-sync', 'nexus', 'modules', 'install', '--all', '--runtime'])
    run([uv, 'build', '--wheel', '--out-dir', str(output / 'wheels')])
    version = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    wheel = output / 'wheels' / f'nexus_agent-{version}-py3-none-any.whl'
    distributions = dependency_distributions()
    print(f'Packing Nexus {version} and {len(distributions)} installed runtime dependencies', flush=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix='.payload-', dir=output) as temporary_payload:
        payload = Path(temporary_payload)
        # Reuse dependency versions validated by Nexus modules install; do not ship
        # the editable checkout or unrelated development packages from .venv.
        shutil.copytree(
            Path(sys.base_prefix),
            payload / 'python',
            ignore=ignore_python_files,
        )
        target_site = payload / 'python' / 'Lib' / 'site-packages'
        for dist in distributions.values():
            source_site = Path(dist.locate_file('')).resolve()
            for item in dist.files or []:
                source = Path(dist.locate_file(item)).resolve()
                if not source.is_relative_to(source_site) or not source.is_file():
                    continue
                if source.suffix == '.pyc' or source.name == 'direct_url.json':
                    continue
                destination = target_site / source.relative_to(source_site)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        (payload / 'tools').mkdir()
        shutil.copyfile(uv, payload / 'tools' / 'uv.exe')
        python = payload / 'python' / 'python.exe'
        # This is our private copy of uv's Python, not the managed host interpreter.
        run([
            uv,
            'pip',
            'install',
            '--python',
            str(python),
            '--break-system-packages',
            '--no-deps',
            '--no-config',
            '--reinstall-package',
            'nexus-agent',
            str(wheel),
        ])
        # The wheel is installed through uv. This is not zipimport: source-based
        # module discovery and package resources retain their supported semantics.
        env = {
            key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')
        }
        env.update(UV_PYTHON=str(python), UV_NO_SYNC='1', UV_BREAK_SYSTEM_PACKAGES='1', PYTHONUTF8='1')
        env['PATH'] = str(payload / 'python' / 'Scripts') + os.pathsep + env.get('PATH', '')
        run(
            [
                uv,
                'run',
                '--no-project',
                '--no-sync',
                '--python',
                str(python),
                'nexus',
                'modules',
                'install',
                '--all',
                '--runtime',
            ],
            env=env,
            cwd=payload,
        )
        run(
            [
                str(python),
                '-I',
                '-c',
                'import nexus; from nexus.runtime.server import app; print(nexus.__file__); print(app.__file__)',
            ],
            env=env,
            cwd=payload,
        )
        shutil.copyfile(HERE / 'entrypoint.py', payload / 'entrypoint.py')
        print('Compressing runtime payload', flush=True)
        release = pack_release(payload, output, version=version, commit=commit, arch=args.arch)
    record = {
        'status': 'BUILT',
        **release,
        'wheel': str(wheel),
        'wheel_sha256': digest(wheel),
        'bundle': str(output),
        'dependencies': {name: dist.version for name, dist in sorted(distributions.items())},
        'source_dirty': bool(
            subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT, text=True
            ).strip()
        ),
        'source_commit': commit,
        'python_version': sys.version,
        'load_mode': 'wheel-installed',
    }
    (output / 'build.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in record.items() if key != 'dependencies'}, indent=2))


if __name__ == '__main__':
    main()
