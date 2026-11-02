"""Stdlib-only Claude installation inspection and executable preparation.

Also executed by the transport in the Agent account's filesystem.
"""

from __future__ import annotations

import contextlib
import json
import logging
import platform
import re
import shutil
import stat
import struct
import sys

from pathlib import Path


logger = logging.getLogger(__name__)
_NATIVE_RELEASE_MIN_VERSION = (2, 1, 113)


def is_windows() -> bool:
    return sys.platform == 'win32'


def find_runtime_binary(name: str) -> str | None:
    found = shutil.which(name)
    fallback = Path('/nix/runtime/nix-env/bin') / name
    return found or (str(fallback) if fallback.is_file() else None)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def get_installed_version(install_dir: str) -> str | None:
    """Return the installed Claude Code version if present."""
    package_json = Path(install_dir) / 'node_modules' / '@anthropic-ai' / 'claude-code' / 'package.json'
    if not package_json.exists():
        return None
    try:
        data = json.loads(package_json.read_text())
    except Exception:
        return None
    version = data.get('version')
    return version if isinstance(version, str) else None


_NATIVE_SUBPKG_PREFIX = '@anthropic-ai/claude-code-'


def _parse_release_version(version: str) -> tuple[int, int, int] | None:
    match = re.match(r'^v?(\d+)\.(\d+)\.(\d+)', version.strip())
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _uses_native_subpackages(version: str) -> bool:
    """Return whether this Claude Code version is expected to publish native subpackages."""
    if version == 'latest':
        return True
    parsed = _parse_release_version(version)
    return parsed is not None and parsed >= _NATIVE_RELEASE_MIN_VERSION


def _rooted_path(root: Path, absolute_path: str) -> Path:
    return root / absolute_path.lstrip('/')


def _read_os_release(root: Path) -> dict[str, str]:
    os_release = _rooted_path(root, '/etc/os-release')
    if not os_release.exists():
        return {}
    try:
        lines = os_release.read_text(encoding='utf-8').splitlines()
    except OSError:
        return {}

    data: dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        data[key] = value.strip().strip('"').strip("'")
    return data


def _detect_linux_libc(root: str | Path = '/') -> str:
    """Detect the Linux libc family for the sandbox filesystem.

    We intentionally do not rely on Python's or Node's own libc detection:
    both interpreters can come from Nix and report glibc while the sandbox root
    is actually Alpine/musl. Loader files and ``/etc/os-release`` describe the
    target runtime more reliably.
    """
    root_path = Path(root)
    for pattern in ('lib/ld-musl-*.so.1', 'lib64/ld-musl-*.so.1', 'usr/lib/ld-musl-*.so.1'):
        if any(root_path.glob(pattern)):
            return 'musl'

    os_release = _read_os_release(root_path)
    os_id = os_release.get('ID', '').lower()
    id_like = {part.lower() for part in os_release.get('ID_LIKE', '').split()}
    if os_id == 'alpine' or 'alpine' in id_like:
        return 'musl'

    return 'glibc'


def _normalize_native_arch(machine: str | None = None) -> str | None:
    normalized = (machine or platform.machine()).lower()
    if normalized in {'x86_64', 'amd64'}:
        return 'x64'
    if normalized in {'aarch64', 'arm64'}:
        return 'arm64'
    return None


def _detect_claude_native_package(
    root: str | Path = '/',
    *,
    platform_name: str | None = None,
    machine: str | None = None,
) -> str | None:
    """Return the exact Claude Code native npm package for this sandbox."""
    arch = _normalize_native_arch(machine)
    if arch is None:
        return None

    current_platform = platform_name or sys.platform
    if current_platform.startswith('linux'):
        libc_suffix = '-musl' if _detect_linux_libc(root) == 'musl' else ''
        return f'{_NATIVE_SUBPKG_PREFIX}linux-{arch}{libc_suffix}'
    if current_platform == 'darwin':
        return f'{_NATIVE_SUBPKG_PREFIX}darwin-{arch}'
    if current_platform in {'win32', 'cygwin'}:
        return f'{_NATIVE_SUBPKG_PREFIX}win32-{arch}'
    return None


# Mirrors _detect_linux_libc for a filesystem this process cannot see: loader files first,
# then the os-release identity. Runs under plain `sh`, since the target may ship no bash.
LINUX_LIBC_PROBE = (
    'set -u; '
    'for candidate in /lib/ld-musl-*.so.1 /lib64/ld-musl-*.so.1 /usr/lib/ld-musl-*.so.1; do '
    'if [ -e "$candidate" ]; then printf "%s musl\\n" "$(uname -m)"; exit 0; fi; done; '
    'if [ -r /etc/os-release ]; then . /etc/os-release 2>/dev/null || true; '
    'case " ${ID:-} ${ID_LIKE:-} " in *alpine*) printf "%s musl\\n" "$(uname -m)"; exit 0;; esac; fi; '
    'printf "%s glibc\\n" "$(uname -m)"'
)


def claude_native_package_from_probe(probe_output: str) -> str | None:
    """Map ``LINUX_LIBC_PROBE`` output to the native subpackage that target can execute.

    Needed whenever the machine doing the installing is not the machine doing the running:
    a Nix-based host resolves its own libc, which says nothing about the container the
    binary will be exec'd in.
    """
    fields = (probe_output or '').split()
    if len(fields) != 2:
        raise ValueError(f'unexpected libc probe output: {probe_output!r}')
    machine, libc = fields
    arch = _normalize_native_arch(machine)
    if arch is None:
        return None
    return f'{_NATIVE_SUBPKG_PREFIX}linux-{arch}{"-musl" if libc == "musl" else ""}'


def _native_package_dir_name(package: str) -> str:
    return package.rsplit('/', 1)[-1]


def _native_package_npm_env(package: str) -> dict[str, str]:
    name = _native_package_dir_name(package)
    prefix = 'claude-code-'
    if not name.startswith(prefix):
        return {}

    parts = name[len(prefix) :].split('-')
    if len(parts) < 2:
        return {}

    os_name, cpu = parts[0], parts[1]
    env = {
        'NPM_CONFIG_OS': os_name,
        'npm_config_os': os_name,
        'NPM_CONFIG_CPU': cpu,
        'npm_config_cpu': cpu,
    }
    if os_name == 'linux':
        libc = parts[2] if len(parts) >= 3 else 'glibc'
        env.update({
            'NPM_CONFIG_LIBC': libc,
            'npm_config_libc': libc,
        })
    return env


def _alternate_linux_native_package(package: str) -> str | None:
    """Return the other libc variant after npm reports EBADPLATFORM."""
    marker = '-linux-'
    if marker not in package:
        return None
    if package.endswith('-musl'):
        return package.removesuffix('-musl')
    return f'{package}-musl'


def _read_elf_interpreter(binary: Path) -> str | None:
    """Return the PT_INTERP path from an ELF binary, or None for non-ELF/static files."""
    try:
        with binary.open('rb') as f:
            ident = f.read(16)
            if len(ident) < 16 or not ident.startswith(b'\x7fELF'):
                return None

            elf_class = ident[4]
            data_encoding = ident[5]
            if elf_class not in {1, 2} or data_encoding not in {1, 2}:
                return None

            endian = '<' if data_encoding == 1 else '>'
            if elf_class == 2:
                header = f.read(struct.calcsize(f'{endian}HHIQQQIHHHHHH'))
                if len(header) != struct.calcsize(f'{endian}HHIQQQIHHHHHH'):
                    return None
                fields = struct.unpack(f'{endian}HHIQQQIHHHHHH', header)
                e_phoff = fields[4]
                e_phentsize = fields[8]
                e_phnum = fields[9]
                ph_format = f'{endian}IIQQQQQQ'
                ph_offset_index = 2
                ph_filesz_index = 5
            else:
                header = f.read(struct.calcsize(f'{endian}HHIIIIIHHHHHH'))
                if len(header) != struct.calcsize(f'{endian}HHIIIIIHHHHHH'):
                    return None
                fields = struct.unpack(f'{endian}HHIIIIIHHHHHH', header)
                e_phoff = fields[4]
                e_phentsize = fields[8]
                e_phnum = fields[9]
                ph_format = f'{endian}IIIIIIII'
                ph_offset_index = 1
                ph_filesz_index = 4

            ph_size = struct.calcsize(ph_format)
            if e_phentsize < ph_size:
                return None

            for index in range(e_phnum):
                f.seek(e_phoff + index * e_phentsize)
                ph = f.read(e_phentsize)
                if len(ph) < ph_size:
                    return None
                values = struct.unpack(ph_format, ph[:ph_size])
                if values[0] != 3:  # PT_INTERP
                    continue
                offset = values[ph_offset_index]
                size = values[ph_filesz_index]
                if size <= 0 or size > 4096:
                    return None
                f.seek(offset)
                data = f.read(size)
                return data.split(b'\0', 1)[0].decode('utf-8', errors='replace')
    except (OSError, struct.error, ValueError):
        return None
    return None


def _is_candidate_launchable(candidate: Path, *, local_execution: bool = True) -> bool:
    """Whether *candidate* can run, judged against this machine's loaders.

    ``local_execution=False`` says the binary is destined for another filesystem, so this
    machine's loaders answer the wrong question -- a build for the target is expected to be
    unrunnable here, and rejecting it is how the caller ends up shipping the wrong one.
    """
    if not local_execution:
        return True
    interpreter = _read_elf_interpreter(candidate)
    if not interpreter or not interpreter.startswith('/'):
        return True
    if Path(interpreter).exists():
        return True
    logger.warning(f'Skipping Claude Code binary {candidate}: missing ELF interpreter {interpreter}')
    return False


def _candidate_executables(package_dir: Path) -> tuple[Path, Path]:
    return package_dir / 'claude', package_dir / 'claude.exe'


def _find_native_executable(package_dir: Path, *, local_execution: bool = True) -> Path | None:
    for candidate in _candidate_executables(package_dir):
        if candidate.is_file() and _is_candidate_launchable(candidate, local_execution=local_execution):
            return candidate
    return None


def _find_native_package_executable(
    install_dir: str,
    package: str,
    *,
    local_execution: bool = True,
) -> Path | None:
    package_dir = Path(install_dir) / 'node_modules' / '@anthropic-ai' / _native_package_dir_name(package)
    if not package_dir.is_dir():
        return None
    return _find_native_executable(package_dir, local_execution=local_execution)


def _resolve_claude_executable(
    install_dir: str,
    preferred_package: str | None = None,
    *,
    local_execution: bool = True,
) -> Path | None:
    """Return the actual Claude Code executable path.

    Releases from **2.1.113** onward ship as a native binary, distributed via
    ``@anthropic-ai/claude-code-<platform>`` optional dependencies — the main
    package's ``bin.claude`` points at ``bin/claude.exe``, which a postinstall
    script overwrites in place. The postinstall's platform detection can
    disagree with which subpackage npm actually reified (e.g. glibc-linked
    Node on a musl host), so we scan for any downloaded native subpackage and
    return its bundled binary directly. Releases **<=2.1.112** are JS bundles
    (``cli.js`` with a ``#!/usr/bin/env node`` shebang); for those we fall
    back to ``.bin/claude``, which still needs a real ``node`` interpreter —
    see :func:`resolve_claude_launch_argv`.
    """
    install_path = Path(install_dir)
    node_modules = install_path / 'node_modules'
    anthropic_dir = node_modules / '@anthropic-ai'
    if anthropic_dir.is_dir():
        entries: list[Path] = []
        if preferred_package is not None:
            preferred_entry = anthropic_dir / _native_package_dir_name(preferred_package)
            if preferred_entry.is_dir():
                entries.append(preferred_entry)

        seen = {entry.resolve() for entry in entries}
        for entry in sorted(anthropic_dir.iterdir()):
            if not entry.name.startswith('claude-code-'):
                continue
            resolved = entry.resolve()
            if resolved in seen:
                continue
            entries.append(entry)
            seen.add(resolved)

        for entry in entries:
            candidate = _find_native_executable(entry, local_execution=local_execution)
            if candidate is not None:
                return candidate

    legacy = node_modules / '.bin' / 'claude'
    if legacy.exists():
        return legacy
    return None


def _patch_shebang_to_node(script: Path, node: str) -> bool:
    """Rewrite *script*'s shebang to ``#!<node>``. Returns True if the file was changed."""
    data = script.read_bytes()
    if not data.startswith(b'#!'):
        return False
    newline = data.find(b'\n')
    if newline == -1:
        return False
    desired = f'#!{node}\n'.encode()
    if data[: newline + 1] == desired:
        return False
    mode = script.stat().st_mode
    tmp = script.with_name(script.name + '.shebang.tmp')
    tmp.write_bytes(desired + data[newline + 1 :])
    tmp.chmod(mode)
    tmp.replace(script)
    return True


def _make_self_executable(
    install_dir: str,
    preferred_package: str | None = None,
    *,
    local_execution: bool = True,
) -> Path | None:
    """Make the installed Claude Code binary directly exec'able.

    The sandbox strips ``/nix`` from the agent PATH, which breaks the
    ``#!/usr/bin/env node`` shebang on JS-bundle releases (<=2.1.112) and
    also means that Claude's own internal re-invocations of ``claude`` (e.g.
    ``claude mcp add``) would not find a node interpreter. Two fixes:

    1. If the binary is a JS wrapper, bake an absolute node path into its
       shebang — resolved via :func:`find_runtime_binary` every install so a
       nix GC between runs repairs itself.
    2. Create a stable top-level symlink at ``<install_dir>/claude`` pointing
       at the resolved executable. ``install_dir`` is prepended to the agent
       PATH by :func:`build_runtime_path`, so plain ``claude`` resolves here.
    """
    exe = _resolve_claude_executable(
        install_dir,
        preferred_package=preferred_package,
        local_execution=local_execution,
    )
    if exe is None:
        return None
    target = exe.resolve()
    if is_windows() and target.suffix.lower() == '.exe':
        # Launch the native binary directly; no Unix shebang or privileged symlink needed.
        return target

    try:
        head = target.open('rb').read(2)
    except OSError:
        head = b''
    if head == b'#!':
        node = find_runtime_binary('node')
        if node is None:
            raise RuntimeError(
                f'Claude Code at {target} is a Node.js wrapper but no ``node`` binary '
                'was found (checked PATH and /nix/runtime/nix-env/bin). Either install '
                'node in the runtime image, or pin a Claude Code version that ships a '
                'native binary (>=2.1.113).'
            )
        if _patch_shebang_to_node(target, node):
            logger.debug(f'Rewrote shebang of {target} to #!{node}')

    with contextlib.suppress(Exception):
        _make_executable(target)

    top_level = Path(install_dir) / 'claude'
    if top_level.is_symlink() or top_level.exists():
        with contextlib.suppress(FileNotFoundError):
            top_level.unlink()
    top_level.symlink_to(target)
    return top_level


def operate(request: dict):
    """Inspect or prepare an installation on this filesystem; no npm policy lives here."""
    operation = request['operation']
    if operation == 'detect':
        return _detect_claude_native_package()
    directory = request['install_dir']
    package = request.get('preferred_package')
    local_execution = request.get('local_execution', True)
    if operation == 'inspect':
        executable = _resolve_claude_executable(directory, preferred_package=package, local_execution=local_execution)
        if package and _find_native_package_executable(directory, package, local_execution=local_execution) is None:
            executable = None
        return {
            'version': get_installed_version(directory),
            'executable': str(executable) if executable is not None else None,
        }
    if operation == 'finalize':
        result = _make_self_executable(directory, preferred_package=package, local_execution=local_execution)
        return str(result) if result is not None else None
    raise ValueError(operation)


if __name__ == '__main__':
    print(json.dumps(operate(json.load(sys.stdin))))
