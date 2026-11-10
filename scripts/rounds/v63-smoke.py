"""End-to-end check of protocol v6.3 without a model: a shell script plays the agent in the real dev container.

Usage: v63-smoke.py TASK SETTING SECONDS OUT
  TASK: a v6.3 task version (conic with FRONTIERAM_RUNTIME=.runtime-conic-v3, exponential-v3, galaxy-gravity-v3,
  polarization-demosaicking-v2, species-tree-v3, trajectory-covariance-v2); SETTING: single / iterative.
  The script checks the workspace (statement and samples only), writes a trivial submission, and in the iterative
  setting submits a test, polls --status until the report arrives, checks it lists no case IDs, then submits --final.
  Prints develop_result.json and the checks; exits 1 if any check fails.
"""

import json
import sys

from pathlib import Path

import arena.tasks  # noqa: F401
from arena.cli import context, placement_for
from arena.native import develop
from arena.tasks import TASKS

# A trivial submission per task family: valid where that is one line, otherwise deliberately wrong.
SUBMISSIONS = {
    'conic': ("cat > run.sh <<'EOF'\n#!/bin/sh\nexec python3 -c \"import sys,json,numpy as np; "
              "o=sys.argv[sys.argv.index('--output')+1]; np.savez(open(o,'wb'), metadata_json=np.array(json.dumps("
              "{'schema_version':1,'status':'unresolved'})))\" \"$@\"\nEOF\nchmod +x run.sh"),
    'exponential': ("mkdir -p sdk && cp /workspace/task/sdk/wire.py sdk/ && cat > solve.py <<'EOF'\n"
                    "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\n"
                    "from sdk.wire import ready, read_header, read_request, write_result\nimport numpy as np\n"
                    "from scipy.sparse import csr_matrix, bmat\nfrom scipy.sparse.linalg import expm_multiply\n"
                    "ready()\nh = read_header()\nL = csr_matrix((h['data'], h['indices'], h['indptr']), shape=(h['n'], h['n']))\n"
                    "for _ in range(h['requests']):\n    b0, b1 = read_request(h)\n"
                    "    M = bmat([[-L, csr_matrix(b1.reshape(-1, 1))], [None, csr_matrix((1, 1))]], format='csr')\n"
                    "    write_result(np.ascontiguousarray(expm_multiply(h['t'] * M, np.append(b0, 1.0))[:-1]))\nEOF\n"
                    "printf '#!/bin/sh\\nexec python3 \"$(dirname \"$0\")/solve.py\"\\n' > run.sh && chmod +x run.sh"),
    'galaxy-gravity': ("cat > solve <<'EOF'\n#!/bin/sh\nexec python3 -c \"import os,sys; n=os.path.getsize(sys.argv[1])//32; "
                       "open(sys.argv[2],'wb').write(bytes(24*n))\" \"$@\"\nEOF\nchmod +x solve"),
    'polarization-demosaicking': ("cat > solution.py <<'EOF'\nimport numpy as np\n\ndef reconstruct(raw):\n"
                                  "    return np.stack([raw.astype(np.float32)] * 4)\nEOF"),
    'species-tree': ("cat > run.sh <<'EOF'\n#!/bin/sh\nexec python3 -c \"import re,sys; t=open(sys.argv[1]).readline(); "
                     "labels=sorted(set(re.findall(r'[(,]([^():,;]+)', t)), key=lambda s:(len(s),s)); "
                     "tree=labels[0]\nfor x in labels[1:]: tree='('+tree+','+x+')'\nopen(sys.argv[2],'w').write(tree+';')\" \"$@\"\nEOF\n"
                     "chmod +x run.sh"),
    'trajectory-covariance': ("cat > run.sh <<'EOF'\n#!/bin/sh\nexec python3 -c \"import numpy as np,sys; x=np.load(sys.argv[1]); "
                              "np.save(sys.argv[2], np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1]))\" \"$@\"\nEOF\n"
                              "chmod +x run.sh"),
}
FAMILY = {'conic': 'conic', 'exponential-v3': 'exponential', 'galaxy-gravity-v3': 'galaxy-gravity',
          'polarization-demosaicking-v2': 'polarization-demosaicking', 'species-tree-v3': 'species-tree',
          'trajectory-covariance-v2': 'trajectory-covariance'}


def script(task: str, setting: str) -> str:
    family = FAMILY[task]
    lines = ['set -x', 'echo WORKSPACE-BEGIN; find /workspace -maxdepth 3 -not -path "*/submission*" | sort; echo WORKSPACE-END',
             'cd /workspace/submission', SUBMISSIONS[family], "echo 'v6.3 smoke submission' > README.md", 'cd /workspace']
    if setting == 'iterative':
        lines += ['/opt/frontieram/bin/submit --test; echo "test-exit=$?"',
                  '/opt/frontieram/bin/submit --test; echo "second-test-exit=$?"',
                  'for i in $(seq 1 200); do [ -f /workspace/feedback/test-01/report.md ] && break; sleep 6; done',
                  '/opt/frontieram/bin/submit --status; echo "status-exit=$?"',
                  'echo REPORT-BEGIN; cat /workspace/feedback/test-01/report.md; echo REPORT-END',
                  "echo 'v6.3 smoke submission (final)' > /workspace/submission/README.md",
                  '/opt/frontieram/bin/submit --final; echo "final-exit=$?"', 'sleep 120', 'echo NOT-REACHED']
    else:
        lines += ['ls /opt/frontieram 2>&1; echo single-done']
    return '\n'.join(lines) + '\n'


def main():
    task_name, setting, seconds, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), Path(sys.argv[4]).resolve()
    layout, broker, runtime = context()
    task = TASKS[task_name]
    out.mkdir(parents=True)
    with broker.acquire('dev', task.dev_memory, f'v6.3 smoke {out.name}', timeout=3600) as lease:
        spec = {'task': task_name, 'model': 'gpt-6.1-sol', 'placement': placement_for(lease, layout).__dict__,
                'output': str(out), 'protocol': 'v6.3', 'setting': setting, 'development_seconds': seconds,
                'agent_script': script(task_name, setting)}
        result = develop(spec, runtime)
    print(json.dumps({k: v for k, v in result.items() if k != 'route'}, indent=1))
    log = (out / 'cli.log').read_text(errors='replace')
    workspace = log.split('WORKSPACE-BEGIN', 1)[-1].split('WORKSPACE-END', 1)[0]
    prompt = (out / 'prompt.txt').read_text()
    checks = {'frozen': result.get('source_frozen') is True,
              'no_public_dev': 'public-dev' not in workspace and 'dev_full' not in workspace,
              'samples_present': '/samples' in workspace,
              'limit_in_prompt': f'{int(seconds // 60) if seconds >= 60 else 0} minutes' in prompt or 'minutes' in prompt}
    if setting == 'iterative':
        report = log.split('REPORT-BEGIN', 1)[-1].split('REPORT-END', 1)[0]
        ids = json.loads((out / 'evaluations/test-01/report/report.json').read_text()) if \
            (out / 'evaluations/test-01/report/report.json').exists() else {}
        checks.update(stop_reason=result.get('agent_stop_reason') == 'formal_submission',
                      test_returned_at_once='is being evaluated in the background' in log,
                      one_test_at_a_time='still being evaluated' in log,
                      status_shown='Time left' in log,
                      report_delivered='Public-set score' in report or 'Invalid submission' in report or 'Not measured' in report,
                      no_case_ids_in_report='"id"' not in json.dumps(ids) and 'case_id' not in report,
                      final_frozen='(final)' in (out / 'task/source/README.md').read_text(),
                      tests_counted=result.get('test_submissions', 0) >= 1)
    else:
        checks.update(no_submit_client='No such file or directory' in log,
                      stop_reason=result.get('agent_stop_reason') in ('turn_completed', 'development_deadline'))
    print(json.dumps(checks))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
