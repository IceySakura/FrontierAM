"""End-to-end check of protocol v6.2 without a model: a shell script plays the agent in the real dev container.

Usage: v62-smoke.py TASK SETTING SECONDS OUT
  conic / exponential-v2, single / iterative. The script writes a trivial submission, runs one test submission on a
  small public subset (iterative), checks that the report arrived in /workspace/feedback, changes the submission and
  submits --final (iterative) or finishes its turn (single). Prints develop_result.json and a few assertions.
"""

import json
import sys

from pathlib import Path

from arena.cli import context, placement_for
from arena.native import develop
from arena.tasks import TASKS

CONIC_RUN = r'''#!/bin/sh
exec python3 -c "import sys,json,numpy as np; o=sys.argv[sys.argv.index('--output')+1]; np.savez(open(o,'wb'), metadata_json=np.array(json.dumps({'schema_version':1,'status':'unresolved'})))" "$@"
'''

EXP2_SOLVE = r'''from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdk.wire import ready, read_header, read_request, write_result
import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import expm_multiply


def main():
    ready()
    h = read_header()
    L = csr_matrix((h['data'], h['indices'], h['indptr']), shape=(h['n'], h['n']))
    for _ in range(h['requests']):
        b0, b1 = read_request(h)
        M = bmat([[-L, csr_matrix(b1.reshape(-1, 1))], [None, csr_matrix((1, 1))]], format='csr')
        y = expm_multiply(h['t'] * M, np.append(b0, 1.0))
        write_result(np.ascontiguousarray(y[:-1]))


if __name__ == '__main__':
    main()
'''


def script(task: str, setting: str) -> str:
    if task == 'conic':
        files = f"cat > /workspace/submission/run.sh <<'EOF'\n{CONIC_RUN}EOF\nchmod +x /workspace/submission/run.sh\n"
        subset = '--groups exponential'
    else:
        files = (f"mkdir -p /workspace/submission/sdk && cp /workspace/task/sdk/wire.py /workspace/submission/sdk/ && "
                 f"cat > /workspace/submission/solve.py <<'EOF'\n{EXP2_SOLVE}EOF\n"
                 "printf '#!/bin/sh\\nexec python3 \"$(dirname \"$0\")/solve.py\"\\n' > /workspace/submission/run.sh\n"
                 "chmod +x /workspace/submission/run.sh\n")
        subset = '--cases standard_DIMACS10_fe_4elt2_p4_k1,control_zero,control_scalar_stiff'
    lines = ['set -x', files, "echo 'smoke submission v1' > /workspace/submission/README.md"]
    if setting == 'iterative-cap':  # conic: every solve sleeps past its limit; a short cap must return partial results
        lines += ["printf '#!/bin/sh\\nsleep 400\\n' > /workspace/submission/run.sh",
                  '/opt/frontieram/bin/submit --status; echo "status-exit=$?"',
                  '/opt/frontieram/bin/submit --test --groups soc; echo "submit-exit=$?"',
                  'ls -la /workspace/feedback/test-01/ && head -12 /workspace/feedback/test-01/report.md',
                  '/opt/frontieram/bin/submit --final; echo "final-exit=$?"', 'sleep 120', 'echo NOT-REACHED']
    elif setting == 'iterative':
        lines += [f'/opt/frontieram/bin/submit --test {subset}; echo "submit-exit=$?"',
                  'ls -la /workspace/feedback/test-01/ && head -20 /workspace/feedback/test-01/report.md',
                  "echo 'smoke submission v2 (final)' > /workspace/submission/README.md",
                  '/opt/frontieram/bin/submit --final; echo "final-exit=$?"', 'sleep 120', 'echo NOT-REACHED']
    else:
        lines += ['ls /opt/frontieram 2>&1; ls /run/frontieram-eval 2>&1; echo single-done']
    return '\n'.join(lines) + '\n'


def main():
    task_name, setting, seconds, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), Path(sys.argv[4]).resolve()
    layout, broker, runtime = context()
    task = TASKS[task_name]
    out.mkdir(parents=True)
    with broker.acquire('dev', task.dev_memory, f'v6.2 smoke {out.name}', timeout=3600) as lease:
        spec = {'task': task_name, 'model': 'gpt-6.1-sol', 'placement': placement_for(lease, layout).__dict__,
                'output': str(out), 'protocol': 'v6.2', 'setting': setting.replace('-cap', ''),
                'development_seconds': seconds,
                'agent_script': script(task_name, setting)}
        result = develop(spec, runtime)
    print(json.dumps(result, indent=1))
    log = (out / 'cli.log').read_text(errors='replace')
    checks = {'frozen': result.get('source_frozen') is True,
              'final_readme_frozen': 'v2' in (out / 'task/source/README.md').read_text()
              if setting == 'iterative' else True}
    if setting == 'iterative-cap':
        test = next((r for r in result.get('evaluations', []) if r.get('kind') == 'test'), {})
        report = json.loads((out / 'evaluations/test-01/report/report.json').read_text())
        checks.update(stop_reason=result.get('agent_stop_reason') == 'formal_submission',
                      status_ok='Remaining budget' in log, capped=report.get('capped') is True,
                      partial=bool(report.get('not_evaluated')) and bool(report.get('cases')),
                      feedback_in_container='report.md' in log and 'NOT-REACHED' not in log)
    elif setting == 'iterative':
        checks.update(stop_reason=result.get('agent_stop_reason') == 'formal_submission',
                      test_done=any(r.get('kind') == 'test' and r.get('status') in ('ok', 'invalid_submission')
                                    for r in result.get('evaluations', [])),
                      feedback_in_container='report.md' in log and 'NOT-REACHED' not in log)
    else:
        checks.update(no_eval_mounts='No such file or directory' in log, stop_reason=result.get('agent_stop_reason')
                      == 'turn_completed')
    print(json.dumps(checks))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
