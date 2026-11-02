"""Windows Nexus payload entrypoint implementing the Bootstrap argument contract."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import sys

from pathlib import Path


def bundle_info():
    import nexus

    from nexus.runtime.server import app

    return {
        'pid': os.getpid(),
        'parent_pid': os.getppid(),
        'python': sys.executable,
        'nexus_version': importlib.metadata.version('nexus-agent'),
        'nexus_file': nexus.__file__,
        'server_file': app.__file__,
        'server_sha256': hashlib.sha256(Path(app.__file__).read_bytes()).hexdigest(),
        'release_id': os.environ.get('NEXUS_BOOTSTRAP_RELEASE_ID'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    for folder in (args.state_root, args.workspace, args.state_root / 'tmp'):
        folder.mkdir(parents=True, exist_ok=True)
    os.environ['NEXUS_HOME'] = str(args.state_root / 'nexus-home')
    os.environ['NEXUS_WORKSPACE'] = str(args.workspace)
    os.environ['NEXUS_AUTO_INSTALL'] = '0'
    os.environ['TEMP'] = os.environ['TMP'] = str(args.state_root / 'tmp')
    os.environ['PATH'] = os.pathsep.join([
        str(bundle / 'python'),
        str(bundle / 'python' / 'Scripts'),
        str(bundle / 'tools'),
        os.environ.get('PATH', ''),
    ])
    os.chdir(args.workspace)
    port = int(os.environ.get('NEXUS_RUNTIME_PORT', '18888'))
    if port == 0:
        with socket.socket() as reservation:
            reservation.bind(('0.0.0.0', 0))
            port = reservation.getsockname()[1]
    import uvicorn

    from nexus.runtime.server.app import asgi_app, configure

    configure(host='127.0.0.1', port=port)
    # Keep the local probe URL usable; 0.0.0.0 is a bind address, not a client destination.
    info = {**bundle_info(), 'url': f'http://127.0.0.1:{port}', 'bind_host': '0.0.0.0', 'state': 'starting'}
    (args.workspace / 'ready.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info), flush=True)
    uvicorn.run(asgi_app, host='0.0.0.0', port=port, ws_max_size=100 * 1024 * 1024)


if __name__ == '__main__':
    main()
