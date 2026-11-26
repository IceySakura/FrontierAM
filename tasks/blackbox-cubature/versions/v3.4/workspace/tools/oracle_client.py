"""Query the sample functions during development (no budget): the functions run outside this container and answer over
/run/cubature-samples/oracle.sock.

    from oracle_client import SampleOracle, sample_ids
    f = SampleOracle('samples/H1-d8-k4-s0')
    values = f(points)          # points: (n, d) array, |z_i| <= 40; any n (sent in batches of 4096)

Only the sample instances listed in /workspace/samples/samples.json are served.
"""
from __future__ import annotations

import json
import os
import socket

import numpy as np

SOCKET = os.environ.get('CUBATURE_ORACLE_SOCKET', '/run/cubature-samples/oracle.sock')
BATCH = 4096


class _Connection:
    def __init__(self, path: str = SOCKET):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect(path)
        self.file = self.sock.makefile('rwb')

    def ask(self, request: dict) -> dict:
        self.file.write((json.dumps(request) + '\n').encode())
        self.file.flush()
        reply = json.loads(self.file.readline())
        if 'error' in reply:
            raise ValueError(reply['error'])
        return reply

    def close(self):
        self.file.close()
        self.sock.close()


def sample_ids() -> list[dict]:
    conn = _Connection()
    try:
        return conn.ask({'list': True})['instances']
    finally:
        conn.close()


class SampleOracle:
    def __init__(self, instance: str):
        self.instance = instance
        self.conn = _Connection()
        self.dimension = next(i['dimension'] for i in self.conn.ask({'list': True})['instances'] if i['id'] == instance)

    def __call__(self, points) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        out = [self.conn.ask({'instance': self.instance, 'points': points[i:i + BATCH].tolist()})['values']
               for i in range(0, len(points), BATCH)]
        return np.concatenate([np.asarray(v, dtype=float) for v in out])
