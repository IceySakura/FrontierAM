"""Minimal client for the blackbox-cubature v3 protocol (problem.md, section "Protocol"): JSON Lines on stdin/stdout.

    session = Session()                  # reads the init message
    values = session.query(points)       # points: (n, d) array; split into batches of at most max_batch
    session.result(estimate, error_bound=None, confidence=None)

Nothing but protocol messages may be written to stdout; use stderr for logs.
"""
from __future__ import annotations

import json
import sys

import numpy as np


class Session:
    def __init__(self, stdin=sys.stdin, stdout=sys.stdout):
        self.stdin, self.stdout = stdin, stdout
        init = json.loads(self.stdin.readline())
        if init.get('type') != 'init':
            raise RuntimeError(f'expected an init message, got {init.get("type")}')
        self.init = init
        self.d = int(init['dimension'])
        self.budget = int(init['budget'])
        self.max_batch = int(init['max_batch'])
        self.seed = int(init['seed'])
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def query(self, points) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != self.d:
            raise ValueError(f'points must have {self.d} columns')
        if len(points) > self.remaining:
            raise ValueError(f'{len(points)} points requested, {self.remaining} left in the budget')
        out = []
        for start in range(0, len(points), self.max_batch):
            batch = points[start:start + self.max_batch]
            self.stdout.write(json.dumps({'type': 'query', 'points': batch.tolist()}, allow_nan=False) + '\n')
            self.stdout.flush()
            self.used += len(batch)
            reply = json.loads(self.stdin.readline())
            if reply.get('type') != 'values':
                raise RuntimeError(f'expected values, got {reply.get("type")}: {reply}')
            values = np.asarray(reply['values'], dtype=float)
            if values.shape != (len(batch),):
                raise ValueError('wrong number of values in a batch')
            out.append(values)
        return np.concatenate(out) if out else np.zeros(0)

    def result(self, estimate: float, error_bound: float | None = None, confidence=None) -> None:
        message = {'type': 'result', 'estimate': float(estimate)}
        if error_bound is not None:
            message['error_bound'] = float(error_bound)
            message['confidence'] = confidence
        self.stdout.write(json.dumps(message, allow_nan=False) + '\n')
        self.stdout.flush()
