"""Run Nexus with a loopback callback address and a public listening socket."""

import argparse
import importlib.metadata
import json
import os
import sys

from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    import uvicorn

    from nexus.runtime.server.app import asgi_app, configure

    port = int(os.environ.get('NEXUS_RUNTIME_PORT', '18888'))
    configure(host='127.0.0.1', port=port)
    info = {
        'pid': os.getpid(),
        'python': sys.executable,
        'nexus_version': importlib.metadata.version('nexus-agent'),
        'release_id': os.environ.get('NEXUS_BOOTSTRAP_RELEASE_ID'),
        'url': f'http://127.0.0.1:{port}',
        'bind_host': '0.0.0.0',
    }
    (args.workspace / 'ready.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info), flush=True)
    uvicorn.run(asgi_app, host='0.0.0.0', port=port, ws_max_size=100 * 1024 * 1024)


if __name__ == '__main__':
    main()
