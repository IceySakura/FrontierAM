"""Host-side sample service for ihvp-sequence-v2: the samples' operators stay on the host.

Each sample case listens on SOCKET_DIR/<case>.sock. A client in the development container (samples/run_sample.py)
connects and relays the unchanged worker protocol (support/worker.py), so the solver sees exactly what the formal
evaluator gives it: the current right-hand side, the sequence deadline and HVP replies. After the sequence the
service writes SOCKET_DIR/results/<case>.json (validity, wall time, HVP count, per-step energy loss) and closes.
Teacher arrays, generator parameters and reference solutions never enter the container.

Runs as its own process under the graders' environment (NumPy/SciPy), started by IHVPSequence.dev_service:
python -m arena.operator_drivers.ihvp_oracle SOCKET_DIR CASES_JSON
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time

from pathlib import Path

import numpy as np

GRADER = Path(__file__).resolve().parents[2] / 'tasks/ihvp-sequence/versions/v1/grader'
WALL_SECONDS = 30.0


def _runtime():
    if str(GRADER) not in sys.path:
        sys.path.insert(0, str(GRADER))
    from runtime import api
    from runtime.backend import Case
    return api, Case


def serve_sequence(stream, case, wall_seconds: float = WALL_SECONDS) -> dict:
    """The host half of runtime/api.py run_case over an already connected stream (no process management)."""
    api, _ = _runtime()
    d, steps = int(case.dimension), int(case.steps)
    predictions = np.zeros((steps, d))
    records = [{'step': t, 'completed': False, 'hvp_vectors': 0} for t in range(steps)]
    start = time.monotonic()
    hard = start + wall_seconds - .3
    soft = max(start, hard - min(1.2, wall_seconds * .1))
    current, terminal = -1, None
    try:
        while True:
            header, array = api.receive_message(stream, hard, expected_dimension=d)
            kind = header.get('kind')
            if kind == 'ready':
                continue
            if kind == 'next':
                tick = header.get('step')
                if type(tick) is not int or tick != current + 1 or not 0 <= tick < steps:
                    raise api.ProtocolError('out-of-order step')
                if current >= 0 and not records[current]['completed']:
                    raise api.ProtocolError('previous step has no output')
                current = tick
                phase = int(case.phase(tick))
                context = {'deadline': soft, 'sequence_deadline': soft, 'hard_deadline': hard, 'step': tick,
                           'steps': steps, 'phase': phase, 'mu': float(case.mu), 'L': float(case.L),
                           'target_rtol': 1e-8, 'phase_changed': bool(tick == 0 or phase != int(case.phase(tick - 1)))}
                api.send_message(stream, 'step', case.rhs(tick), context=context)
            elif kind == 'hvp':
                if header.get('step') != current or current < 0 or records[current]['completed']:
                    raise api.ProtocolError('HVP for unavailable system')
                if array is None or not np.isfinite(array).all() or (array.ndim == 2 and array.shape[1] > api.MAX_BLOCK_COLUMNS):
                    raise api.ProtocolError('invalid HVP vector/block')
                if time.monotonic() >= soft:
                    api.send_message(stream, 'budget')
                    continue
                with np.errstate(over='ignore', invalid='ignore'):
                    product = np.asarray(case.hvp(current, array), dtype=np.float64)
                records[current]['hvp_vectors'] += 1 if array.ndim == 1 else array.shape[1]
                api.send_message(stream, 'product', product)
            elif kind == 'result':
                if header.get('step') != current or current < 0 or records[current]['completed']:
                    raise api.ProtocolError('output for unavailable system')
                if array is None or array.shape != (d,) or not np.isfinite(array).all():
                    raise api.ProtocolError('output must be a finite d-vector')
                predictions[current] = array
                records[current]['completed'] = True
            elif kind == 'done':
                if current != steps - 1 or not all(r['completed'] for r in records):
                    raise api.ProtocolError('incomplete sequence')
                break
            elif kind == 'fatal':
                terminal = 'submission_exception: ' + str(header.get('error'))
                break
            else:
                raise api.ProtocolError('unknown worker message')
    except api.BudgetExceeded:
        terminal = 'wall_limit'
    except (EOFError, BrokenPipeError, ConnectionError):
        terminal = 'unexpected_worker_exit'
    except Exception as error:
        terminal = f'protocol_failure: {type(error).__name__}: {error}'
    wall = time.monotonic() - start
    if wall > wall_seconds:
        terminal = 'wall_limit'
    losses = []
    for t, row in enumerate(records):
        try:
            loss = float(case.score(t, predictions[t])['energy_loss'])
        except Exception:
            loss = None
        row['energy_loss'] = loss
        losses.append(loss)
    valid = terminal is None and all(v is not None for v in losses)
    return {'valid': valid, 'termination_reason': terminal, 'wall_seconds': wall,
            'hvp_vectors': sum(r['hvp_vectors'] for r in records),
            'mean_energy_loss': float(np.mean(losses)) if valid else None, 'steps': records}


class SampleService:
    def __init__(self, socket_dir: Path, cases: dict[str, Path]):
        self.dir, self.cases = Path(socket_dir), cases
        self.sockets: list[socket.socket] = []
        self.stopped = threading.Event()

    def start(self) -> None:
        (self.dir / 'results').mkdir(parents=True, exist_ok=True)
        (self.dir / 'results').chmod(0o755)
        for name, path in self.cases.items():
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            target = self.dir / f'{name}.sock'
            server.bind(str(target))
            target.chmod(0o777)
            server.listen(4)
            server.settimeout(0.5)
            self.sockets.append(server)
            threading.Thread(target=self._accept, args=(server, name, path), daemon=True).start()

    def _accept(self, server: socket.socket, name: str, path: Path) -> None:
        _, Case = _runtime()
        Case(path)  # fail at start-up, not on the first connection
        while not self.stopped.is_set():
            try:
                conn, _ = server.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._handle, args=(conn, name, Case(path)), daemon=True).start()

    def _handle(self, conn: socket.socket, name: str, case) -> None:
        conn.settimeout(None)
        with conn, conn.makefile('rwb', buffering=0) as stream:
            result = serve_sequence(stream, case)
            out = self.dir / 'results' / f'{name}.json'
            temporary = out.with_suffix('.tmp')
            temporary.write_text(json.dumps({'case': name, **result}, indent=1) + '\n')
            temporary.chmod(0o644)
            temporary.replace(out)

    def stop(self) -> None:
        self.stopped.set()
        for server in self.sockets:
            server.close()


def main():
    import signal
    service = SampleService(Path(sys.argv[1]), {k: Path(v) for k, v in json.loads(sys.argv[2]).items()})
    service.start()
    signal.signal(signal.SIGTERM, lambda *_: service.stopped.set())
    print('ready', flush=True)
    service.stopped.wait()
    service.stop()


if __name__ == '__main__':
    main()
