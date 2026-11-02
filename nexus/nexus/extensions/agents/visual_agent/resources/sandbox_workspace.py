"""Workspace archive extraction, invoked in the runtime before agent execution."""

import shutil
import stat
import zipfile

from pathlib import Path, PurePosixPath


def _check_destination(path: Path, *, directory: bool) -> None:
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f'Workspace destination contains a symlink: {parent}')
    if path.exists() and not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f'Workspace destination has the wrong file type: {path}')
    for parent in path.parents:
        if parent.exists() and not parent.is_dir():
            raise ValueError(f'Workspace destination parent is not a directory: {parent}')


def extract_workspace_archive(archive_path: str, target: str) -> None:
    """Merge ZIP contents into an exact directory root without stripping prefixes."""
    root = Path(target)
    if not root.is_absolute():
        raise ValueError('Workspace target must be absolute')
    _check_destination(root, directory=True)

    with zipfile.ZipFile(archive_path) as archive:
        entries: list[tuple[zipfile.ZipInfo, Path]] = []
        destinations: dict[Path, bool] = {root: True}
        for member in archive.infolist():
            name = member.filename
            relative = PurePosixPath(name)
            if name in ('.', './') and member.is_dir():
                continue
            if (
                not relative.parts
                or relative.is_absolute()
                or '..' in relative.parts
                or '\\' in name
                or ':' in relative.parts[0]
                or '\x00' in member.orig_filename
            ):
                raise ValueError(f'Unsafe workspace ZIP member: {name!r}')
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f'Workspace ZIP member is not a regular file or directory: {name!r}')
            is_dir = member.is_dir()
            if kind == stat.S_IFDIR and not is_dir:
                raise ValueError(f'Workspace ZIP directory must end with /: {name!r}')
            destination = root.joinpath(*relative.parts)
            if destination in destinations and not (is_dir and destinations[destination]):
                raise ValueError(f'Conflicting workspace ZIP member: {name!r}')
            destinations[destination] = is_dir
            for parent in destination.parents:
                if parent == root:
                    break
                if destinations.get(parent) is False:
                    raise ValueError(f'Conflicting workspace ZIP directory: {name!r}')
                destinations[parent] = True
            entries.append((member, destination))

        for destination, is_dir in destinations.items():
            _check_destination(destination, directory=is_dir)

        root.mkdir(parents=True, exist_ok=True)
        for member, destination in entries:
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open('wb') as output:
                shutil.copyfileobj(source, output)
            mode = (member.external_attr >> 16) & 0o777
            if member.create_system == 3 and mode:
                destination.chmod(mode)
