"""Stdlib-only file operations executed inside an installed Agent's target.

The transport sends this module's source to the target Python interpreter; it
must not import Nexus or depend on the Worker's virtualenv or filesystem.
"""

import hashlib
import json
import os
import stat
import sys
import tempfile
import urllib.error
import urllib.request

from pathlib import Path


def operate(request):
    operation = request['operation']
    path = Path(request['path'])
    if operation == 'mkdir':
        path.mkdir(parents=True, exist_ok=True)
    elif operation == 'prepare_home':
        path.mkdir(mode=0o700, exist_ok=True)
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise PermissionError(f'Agent directory must be private and owned by the execution user: {path}')
        return tempfile.mkdtemp(prefix='home-', dir=path)
    elif operation == 'exists':
        return path.is_dir() if request.get('directory') else path.exists()
    elif operation == 'read':
        return path.read_text(encoding='utf-8') if path.exists() else None
    elif operation == 'write':
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, request['mode'])
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(request['content'])
        path.chmod(request['mode'])
    elif operation == 'remove':
        path.unlink(missing_ok=True)
    elif operation == 'npm_entry':
        package = path / 'node_modules' / request['package']
        data = json.loads((package / 'package.json').read_text())
        entry = data.get('bin')
        entry = entry.get(request['name']) if isinstance(entry, dict) else entry
        if not isinstance(entry, str) or not (package / entry).is_file():
            raise RuntimeError('npm executable entry is missing')
        return str(package / entry)
    elif operation == 'download':
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix='.download-', dir=path.parent)
        temporary = Path(temporary_name)
        try:
            digest = hashlib.sha256()
            with os.fdopen(fd, 'wb') as stream:
                try:
                    with urllib.request.urlopen(request['url'], timeout=120) as response:
                        while chunk := response.read(65536):
                            stream.write(chunk)
                            digest.update(chunk)
                except urllib.error.HTTPError as exc:
                    raise RuntimeError(f'Binary download failed: HTTP {exc.code}') from None
                except (urllib.error.URLError, TimeoutError):
                    raise RuntimeError('Binary download failed') from None
            checksum = request.get('checksum')
            if checksum and digest.hexdigest().lower() != checksum.lower():
                raise RuntimeError('Binary SHA256 checksum verification failed')
            temporary.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return str(path)
    else:
        raise ValueError(f'Unknown target file operation: {operation}')
    return None


if __name__ == '__main__':
    print(json.dumps(operate(json.load(sys.stdin))))
