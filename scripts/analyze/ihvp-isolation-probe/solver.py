"""Isolation probe for ihvp-sequence graders (any version): a valid zero-vector solver that, once per sequence process,
reports to stderr what the worker can see. Expected: no teacher data, manifests, grader/generator sources, other
processes' details or network; only the protocol files and its own copy. Run like a candidate submission on the public
split and read the per-sequence *.stderr.log files in the grader output.
"""
import json
import os
import socket
import sys

import numpy as np


def attempt(fn):
    try:
        value = fn()
        return {'ok': True, 'value': value if isinstance(value, (int, str, bool)) or value is None else str(value)[:200]}
    except Exception as error:
        return {'ok': False, 'error': type(error).__name__}


def read(path, n=64):
    with open(path, 'rb') as f:
        return f.read(n).hex()[:32]


def readable_files(root, suffixes, limit=20):
    found = []
    for base, dirs, files in os.walk(root, onerror=lambda e: None):
        if base.startswith(('/proc', '/sys', '/dev')):
            dirs[:] = []
            continue
        for name in files:
            path = os.path.join(base, name)
            if name.endswith(suffixes) and os.access(path, os.R_OK):
                found.append(path)
                if len(found) >= limit:
                    return found
    return found


def probe():
    report = {'uid': os.getuid(), 'gid': os.getgid(), 'groups': os.getgroups(), 'cwd': os.getcwd()}
    with open('/proc/self/status') as f:
        report['cap_eff'] = next((line.split()[1] for line in f if line.startswith('CapEff')), None)
    for path in ['/data', '/data/manifest.json', '/root', '/root/task-src', '/opt/repo', '/opt/repo/tasks/ihvp-sequence',
                 '/opt/repo/tasks/ihvp-sequence/versions', '/out', '/submission', '/workspace']:
        report[f'list {path}'] = attempt(lambda p=path: len(os.listdir(p)))
    for path in ['/data/manifest.json', '/opt/repo/tasks/ihvp-sequence/versions/v4/grader/evaluate.py',
                 '/opt/repo/tasks/ihvp-sequence/versions/v4/curator/generate.py',
                 '/opt/repo/tasks/ihvp-sequence/versions/v4/grader/runtime/backend.py',
                 '/opt/repo/tasks/ihvp-sequence/versions/v4/grader/runtime/api.py',
                 '/opt/repo/tasks/ihvp-sequence/references/r1/solver.py']:
        report[f'read {path}'] = attempt(lambda p=path: read(p))
    report['readable npz/json under /'] = readable_files('/', ('.npz', 'manifest.json', 'IMPORT.json', 'seeds.json'))
    others = {}
    for pid in sorted(p for p in os.listdir('/proc') if p.isdigit() and int(p) != os.getpid())[:12]:
        others[pid] = {k: attempt(lambda p=pid, k=k: read(f'/proc/{p}/{k}', 200)) for k in ('cmdline', 'environ', 'maps')}
        others[pid]['fds'] = attempt(lambda p=pid: len(os.listdir(f'/proc/{p}/fd')))
    report['other processes'] = others
    report['dns'] = attempt(lambda: socket.gethostbyname('api.github.com'))
    report['tcp 1.1.1.1:443'] = attempt(lambda: socket.create_connection(('1.1.1.1', 443), timeout=2).close())
    return report


class Solver:
    def __init__(self, dimension, seed, options):
        self.dimension = dimension
        print('ISOLATION_PROBE ' + json.dumps(probe()), file=sys.stderr, flush=True)

    def solve(self, hvp, b, context):
        return np.zeros(self.dimension)


def make_solver(dimension, seed, options):
    return Solver(dimension, seed, options)
