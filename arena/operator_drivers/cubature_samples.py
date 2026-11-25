"""Host-side oracle for the blackbox-cubature v3 samples during development (BlackboxCubatureV3.dev_service).

The workspace has the samples' metadata and reference values (samples.json) but not their functions: those stay here,
so the hidden families' formulas never enter the workspace. The development container reaches this service through a
Unix socket mounted at /run/cubature-samples/oracle.sock (workspace tools/oracle_client.py, tools/local_eval.py).

Requests and replies are JSON lines; one connection may send any number of requests:
    {"instance": ID, "points": [[...], ...]}  ->  {"values": [...]}     (at most 4096 rows, |z_i| <= 40)
    {"list": true}                            ->  {"instances": [{"id", "dimension"}, ...]}
Errors reply {"error": message}. There is no budget here; the local evaluator enforces the formal limits itself.

Usage: cubature_samples.py SOCKET_DIR INSTANCES_JSON   (prints "ready" once listening)
"""
from __future__ import annotations

import json
import os
import socketserver
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tasks/blackbox-cubature/versions/v3/generation'))
import oracle  # noqa: E402

MAX_ROWS = 4096
BOUND = 40.0


def main():
    directory, listing = Path(sys.argv[1]), Path(sys.argv[2])
    instances = {i['id']: i for i in json.loads(listing.read_text())['instances']}
    functions = {k: oracle.make(v['spec']) for k, v in instances.items()}

    def answer(request: dict) -> dict:
        if request.get('list'):
            return {'instances': [{'id': k, 'dimension': v['d']} for k, v in instances.items()]}
        name = request.get('instance')
        if name not in functions:
            return {'error': f'unknown instance {name!r}'}
        d = instances[name]['d']
        try:
            points = np.asarray(request.get('points'), dtype=float)
        except (TypeError, ValueError):
            return {'error': 'points must be a list of numeric rows'}
        if points.ndim != 2 or points.shape[1] != d or not 0 < len(points) <= MAX_ROWS:
            return {'error': f'expected 1 to {MAX_ROWS} rows of {d} coordinates'}
        if not np.all(np.isfinite(points)) or np.any(np.abs(points) > BOUND):
            return {'error': f'coordinates must be finite with |z_i| <= {BOUND:g}'}
        return {'values': np.asarray(functions[name](points), dtype=float).tolist()}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                if not line.strip():
                    continue
                try:
                    reply = answer(json.loads(line))
                except Exception as exc:  # malformed request: report it, keep serving
                    reply = {'error': f'{type(exc).__name__}: {exc}'}
                self.wfile.write((json.dumps(reply) + '\n').encode())
                self.wfile.flush()

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    path = directory / 'oracle.sock'
    server = Server(str(path), Handler)
    os.chmod(path, 0o666)
    print('ready', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
