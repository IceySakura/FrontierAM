#!/usr/bin/env python3
"""Run a submission on one sample through the evaluator's sample service (same protocol as the formal evaluation).

Usage: run_sample.py CASE [SUBMISSION]    CASE: smoke | sample_01 | sample_02; SUBMISSION defaults to /workspace/submission
"""
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVICE = Path('/run/ihvp-samples')
CASES = {'smoke': (4, 3), 'sample_01': (16384, 50), 'sample_02': (16384, 50)}
OPTIONS = {'rank': 16, 'maxiter_factor': 50}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CASES:
        sys.exit(__doc__)
    case = sys.argv[1]
    submission = Path(sys.argv[2] if len(sys.argv) > 2 else '/workspace/submission').resolve()
    dimension, steps = CASES[case]
    result = SERVICE / 'results' / f'{case}.json'
    before = result.stat().st_mtime_ns if result.exists() else 0
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(SERVICE / f'{case}.sock'))
    env = dict(os.environ, **{k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    worker = subprocess.Popen([sys.executable, str(HERE / 'support/worker.py'), '--submission', str(submission),
                               '--dimension', str(dimension), '--steps', str(steps), '--seed', '1',
                               '--options', json.dumps(OPTIONS)],
                              cwd=submission, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def upstream():
        while block := worker.stdout.read(65536):
            connection.sendall(block)
        connection.shutdown(socket.SHUT_WR)

    def downstream():
        try:
            while block := connection.recv(65536):
                worker.stdin.write(block)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                worker.stdin.close()
            except OSError:
                pass

    threads = [threading.Thread(target=upstream, daemon=True), threading.Thread(target=downstream, daemon=True)]
    for thread in threads:
        thread.start()
    worker.wait()
    threads[1].join(timeout=60)
    if not result.exists() or result.stat().st_mtime_ns == before:
        sys.exit('no result from the sample service')
    report = json.loads(result.read_text())
    print(json.dumps({k: report[k] for k in ('case', 'valid', 'termination_reason', 'wall_seconds', 'hvp_vectors',
                                              'mean_energy_loss')}, indent=1))
    print('per-step energy loss:', ' '.join('-' if s['energy_loss'] is None else f'{s["energy_loss"]:.3g}'
                                            for s in report['steps']))


if __name__ == '__main__':
    main()
