"""Align writable workspace directories with Syntara's sandbox identity."""

import os
import posixpath
import pwd
import stat

from pathlib import Path, PurePosixPath


PRIVATE_SERVICE_ROOTS = ('/initial_workspace', '/initial_data', '/data', '/tests', '/logs', '/judge_workspace')
_SYSTEM_ROOTS = (
    '/bin',
    '/boot',
    '/dev',
    '/etc',
    '/lib',
    '/lib64',
    '/proc',
    '/root',
    '/run',
    '/sbin',
    '/sys',
    '/usr',
    '/var',
)


def checked_workspace_root(workdir: str) -> str:
    """Normalize ``workdir`` and reject roots that cover system or private service data."""
    if not workdir.startswith('/'):
        raise ValueError(f'Invalid workspace root: {workdir!r}')
    root = PurePosixPath(posixpath.normpath('/' + workdir.lstrip('/')))
    for value in (*_SYSTEM_ROOTS, *PRIVATE_SERVICE_ROOTS):
        protected = PurePosixPath(value)
        if root == protected or root in protected.parents or protected in root.parents:
            raise ValueError(f'Workspace root {workdir!r} overlaps protected path {value}')
    return str(root)


def prepare_workspace_permissions(workdir: str) -> dict[str, int]:
    """Run inside the Runtime after uploads and workspace-link creation.

    Syntara's attachment writer drops privileges to SETUID/SETGID, even
    though the MCP proxy and Nexus worker run as root. Never follow symlinks
    while granting ownership. Regular workspace files are editable working copies;
    initial_workspace remains a separate, protected judge reference.
    """
    path = Path(workdir)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f'Invalid workspace root: {workdir!r}')
    root = Path(checked_workspace_root(str(path.resolve())))
    account = pwd.getpwnam('model')
    uid, gid = account.pw_uid, account.pw_gid
    # Read only the sandbox identity from the actual MCP service, whose env
    # can differ from the Nexus worker's environment.
    identities: set[tuple[int, int]] = set()
    for process in Path('/proc').glob('[0-9]*'):
        try:
            args = (process / 'cmdline').read_bytes().split(b'\0')
            if not any(Path(arg.decode(errors='replace')).name == 'mcp-proxy' for arg in args):
                continue
            env = dict(item.split(b'=', 1) for item in (process / 'environ').read_bytes().split(b'\0') if b'=' in item)
            if b'SETUID' in env and b'SETGID' in env:
                identities.add((int(env[b'SETUID']), int(env[b'SETGID'])))
        except (OSError, ValueError):
            continue
    if len(identities) > 1:
        raise RuntimeError('Conflicting MCP sandbox identities')
    if identities:
        uid, gid = identities.pop()
    root.mkdir(parents=True, exist_ok=True)
    for name in ('workspace', 'downloads'):
        child = root / name
        if child.is_symlink():
            raise ValueError(f'Workspace directory must not be a symlink: {child}')
        child.mkdir(exist_ok=True)
    count = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        mode = stat.S_IMODE(os.stat(directory, follow_symlinks=False).st_mode)
        os.chown(directory, uid, gid, follow_symlinks=False)
        os.chmod(directory, mode | stat.S_IRWXU)
        count += 1
        for name in files:
            path = Path(directory) / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                continue
            # Break hard links before changing permissions so pristine copies
            # elsewhere cannot inherit writable ownership.
            if info.st_nlink > 1:
                import shutil
                import tempfile

                fd, temporary = tempfile.mkstemp(dir=directory)
                os.close(fd)
                try:
                    shutil.copyfile(path, temporary)
                    os.chmod(temporary, stat.S_IMODE(info.st_mode))
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            os.chown(path, uid, gid, follow_symlinks=False)
            os.chmod(path, (stat.S_IMODE(info.st_mode) | stat.S_IRUSR | stat.S_IWUSR) & ~0o6000)
    return {'uid': uid, 'gid': gid, 'directories': count}


def protect_service_data(workdir: str, uid: int, gid: int) -> dict[str, list[str]]:
    """Hide service/seed/grading roots from the agent without chmod-ing service files.

    Service descendants retain their owners and modes. Root services can still
    traverse the boundary. Non-root service groups receive traverse-only ACLs;
    an explicit agent-user ACL takes precedence over any shared group.
    """
    import shutil
    import subprocess

    checked_workspace_root(str(Path(workdir).resolve()))
    protected = []
    for value in PRIVATE_SERVICE_ROOTS:
        path = Path(value)
        if path.is_symlink():
            raise RuntimeError(f'Private service root is a symlink: {value}')
        path.mkdir(exist_ok=True)
        # Include service account groups so daemons such as postgres retain
        # access to their own subdirectories; do not grant directory listing.
        groups = {p.pw_gid for p in pwd.getpwall() if p.pw_uid not in {0, uid}}
        if any(p.stat().st_uid == uid for p in path.iterdir() if not p.is_symlink()):
            raise RuntimeError(f'Service data under {value} shares Agent UID {uid}; separate service identity required')
        if shutil.which('setfacl'):
            os.chown(path, 0, 0)
            os.chmod(path, 0o700)
            entries = [f'u:{uid}:---', *(f'g:{g}:--x' for g in sorted(groups)), 'm::--x', 'o::---']
            subprocess.run(['setfacl', '-m', ','.join(entries), str(path)], check=True, capture_output=True)
        else:
            # Without ACL support a separate service sharing the agent UID
            # cannot be isolated by chmod. Refuse an unsafe permission change.
            owners = {p.stat().st_uid for p in path.iterdir() if not p.is_symlink()}
            if any(owner not in {0} for owner in owners):
                raise RuntimeError(f'setfacl is required to preserve non-root service access under {value}')
            os.chown(path, 0, 0)
            os.chmod(path, 0o700)
        protected.append(value)
    return {'protected_roots': protected}
