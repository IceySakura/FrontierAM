"""Root-owned judge inputs, never written through agent-controlled symlinks."""

import contextlib
import os


DELIVERABLE_DIR = '/tmp/agent_deliverable'


def _open_directory() -> int:
    parent = os.open('/tmp', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with contextlib.suppress(FileExistsError):
            os.mkdir('agent_deliverable', mode=0o700, dir_fd=parent)
        fd = os.open('agent_deliverable', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    info = os.fstat(fd)
    if info.st_uid != 0:
        os.close(fd)
        raise RuntimeError('Judge deliverable directory is not root-owned')
    os.fchmod(fd, 0o700)
    return fd


def prepare_deliverable_directory() -> None:
    fd = _open_directory()
    os.close(fd)


def write_deliverables(answer: str, transcript: str) -> None:
    directory = _open_directory()
    try:
        for name, content in [('answer.txt', answer), ('trajectory.txt', transcript)]:
            # Unlink the directory entry, not its target. A new inode also
            # avoids overwriting an attacker-provided hard link.
            with contextlib.suppress(FileNotFoundError):
                os.unlink(name, dir_fd=directory)
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            with os.fdopen(fd, 'wb') as stream:
                # Preserve invalid surrogate code points as visible escapes for the Judge.
                stream.write(content.encode('utf-8', errors='backslashreplace'))
    finally:
        os.close(directory)
