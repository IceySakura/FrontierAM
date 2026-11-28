"""End-to-end check of protocol v6.3 without a model: a shell script plays the agent in the real dev container.

Usage: v63-smoke.py TASK SETTING SECONDS OUT
  TASK: a v6.3 task version (conic with FRONTIERAM_RUNTIME=.runtime-conic-v3, exponential-v3, galaxy-gravity-v3,
  polarization-demosaicking-v2, species-tree-v3, trajectory-covariance-v2, ihvp-sequence); SETTING: single / iterative.
  The script checks the workspace (statement and samples only), writes a trivial submission, and in the iterative
  setting submits a test, polls --status until the report arrives, checks it lists no case IDs, then submits --final.
  Prints develop_result.json and the checks; exits 1 if any check fails.
"""

import json
import re
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
    'ihvp-sequence': ("cat > solver.py <<'EOF'\nimport numpy as np\n\nclass Zero:\n    def __init__(self, d):\n        self.d = d\n\n"
                      "    def solve(self, hvp, b, context):\n        return np.zeros(self.d)\n\n\n"
                      "def make_solver(d, seed, options):\n    return Zero(d)\nEOF"),
    'blackbox-cubature': 'cp /workspace/examples/reference_submission/* . && chmod +x run.sh',
    'blackbox-cubature-v3': ('cp /workspace/examples/mc_submission/* . && chmod +x run.sh && mkdir -p theory && '
                             "echo '# smoke theory document' > theory/THEORY.md"),
    'theory-only': "mkdir -p theory && echo '# smoke theory document' > theory/THEORY.md",
    'fluorescence-deconvolution': ("cat > solution.py <<'EOF'\nimport sys\nimport numpy as np\n"
                                   "d = np.load(sys.argv[1])\nnp.save(sys.argv[2], np.maximum(d['observed'], 0).astype(np.float32))\nEOF"),
    'continuum-estimation': ("cat > run.sh <<'EOF'\n#!/bin/sh\nexec python3 -c \"import json,os,sys; a=sys.argv[1:]; "
                             "i=a[0] if a else '/workspace/inputs/input.json'; o=a[1] if len(a)>1 else '/workspace/result/output.json'; "
                             "d=json.load(open(i)); k=min(range(len(d['x'])), key=lambda j: d['x'][j]); "
                             "os.makedirs(os.path.dirname(o), exist_ok=True); json.dump({'estimate': float(d['y'][k])}, open(o, 'w'))\" \"$@\"\n"
                             "EOF\nchmod +x run.sh"),
}
FAMILY = {'conic': 'conic', 'exponential-v3': 'exponential', 'galaxy-gravity-v3': 'galaxy-gravity',
          'polarization-demosaicking-v2': 'polarization-demosaicking', 'species-tree-v3': 'species-tree',
          'trajectory-covariance-v2': 'trajectory-covariance', 'ihvp-sequence': 'ihvp-sequence',
          'ihvp-sequence-v2': 'ihvp-sequence', 'ihvp-sequence-v3': 'ihvp-sequence', 'ihvp-sequence-v4': 'ihvp-sequence', 'ihvp-sequence-v5': 'ihvp-sequence',
          'polarization-demosaicking-v3': 'polarization-demosaicking', 'blackbox-cubature-v2': 'blackbox-cubature',
          'blackbox-cubature-v3': 'blackbox-cubature-v3', 'blackbox-cubature-v3.1': 'blackbox-cubature-v3',
          'blackbox-cubature-v3.2': 'blackbox-cubature-v3', 'blackbox-cubature-v3.3': 'blackbox-cubature-v3',
          'blackbox-cubature-v3.4': 'blackbox-cubature-v3', 'blackbox-cubature-v3.5': 'blackbox-cubature-v3',
          'blackbox-cubature-v3.6': 'blackbox-cubature-v3',
          'blackbox-cubature-theory-v1': 'theory-only', 'decentralized-sum-of-nonconvex-theory-v1': 'theory-only',
          'blackbox-cubature-theory-v2': 'theory-only', 'decentralized-sum-of-nonconvex-theory-v2': 'theory-only',
          'continuum-estimation': 'continuum-estimation', 'fluorescence-deconvolution': 'fluorescence-deconvolution'}
V5_FILES = ('README.md', 'statement.md', 'interface.md', 'solver.py', 'smoke.json')  # the MODEL_ONLY package, nothing else
# Versions that keep the author's workspace layout: their public-dev material is the samples; these files must be there.
AUTHOR_LAYOUT = {'polarization-demosaicking-v3': ['/workspace/public-dev/check.py', '/workspace/public-dev/wen5'],
                 'blackbox-cubature-v2': ['/workspace/environment/public-dev/pilot.py', '/workspace/tests/grade.py',
                                          '/workspace/examples/reference_submission/run.sh'],
                 'continuum-estimation': ['/workspace/problem.md', '/workspace/public/check.py', '/workspace/runtime'],
                 'fluorescence-deconvolution': ['/workspace/problem.md', '/workspace/public-dev/check.py'],
                 'ihvp-sequence-v5': [f'/workspace/{name}' for name in V5_FILES],
                 'blackbox-cubature-v3': ['/workspace/problem.md', '/workspace/THEORY_APPENDIX.md',
                                          '/workspace/samples/samples.json', '/workspace/tools/local_eval.py',
                                          '/workspace/examples/mc_submission/run.sh']}
AUTHOR_LAYOUT['blackbox-cubature-v3.1'] = AUTHOR_LAYOUT['blackbox-cubature-v3']
AUTHOR_LAYOUT['blackbox-cubature-v3.2'] = AUTHOR_LAYOUT['blackbox-cubature-v3'] + [
    '/workspace/examples/lean_project/lakefile.toml', '/workspace/tools/lean_check.py']
AUTHOR_LAYOUT['blackbox-cubature-v3.3'] = AUTHOR_LAYOUT['blackbox-cubature-v3.2']
AUTHOR_LAYOUT['blackbox-cubature-v3.4'] = AUTHOR_LAYOUT['blackbox-cubature-v3.2']
AUTHOR_LAYOUT['blackbox-cubature-v3.5'] = AUTHOR_LAYOUT['blackbox-cubature-v3.2']
AUTHOR_LAYOUT['blackbox-cubature-v3.6'] = AUTHOR_LAYOUT['blackbox-cubature-v3.2']
AUTHOR_LAYOUT['blackbox-cubature-theory-v1'] = ['/workspace/problem.md', '/workspace/THEORY_APPENDIX.md',
                                                '/workspace/examples/lean_project/lakefile.toml', '/workspace/tools/lean_check.py']
AUTHOR_LAYOUT['decentralized-sum-of-nonconvex-theory-v1'] = [
    '/workspace/problem.md', '/workspace/PROFILE.md', '/workspace/MODEL_NOTES.md',
    '/workspace/examples/lean_project/lakefile.toml', '/workspace/tools/lean_check.py']
AUTHOR_LAYOUT['blackbox-cubature-theory-v2'] = AUTHOR_LAYOUT['blackbox-cubature-theory-v1']
AUTHOR_LAYOUT['decentralized-sum-of-nonconvex-theory-v2'] = AUTHOR_LAYOUT['decentralized-sum-of-nonconvex-theory-v1']
THEORY_ONLY = ('blackbox-cubature-theory-v1', 'decentralized-sum-of-nonconvex-theory-v1', 'blackbox-cubature-theory-v2',
               'decentralized-sum-of-nonconvex-theory-v2')
LEAN_TASKS = ('blackbox-cubature-v3.2', 'blackbox-cubature-v3.3', 'blackbox-cubature-v3.4', 'blackbox-cubature-v3.5',
              'blackbox-cubature-v3.6',
              *THEORY_ONLY)
NETWORK_TASKS = ('blackbox-cubature-v3.4', 'blackbox-cubature-v3.5', 'blackbox-cubature-v3.6', *THEORY_ONLY)
NETWORK_PROBES = {'arxiv': 'https://export.arxiv.org/api/query?search_query=all:cubature&max_results=1',
                  'github': 'https://github.com/', 'pypi': 'https://pypi.org/simple/numpy/',
                  'openai': 'https://api.openai.com/v1/models', 'intranet': 'https://git.woa.com/',
                  'optbound': 'https://optbound.reaslab.io/'}


def script(task: str, setting: str) -> str:
    family = FAMILY[task]
    lines = ['set -x', 'echo WORKSPACE-BEGIN; find /workspace -maxdepth 3 -not -path "*/submission*" | sort; echo WORKSPACE-END',
             'cd /workspace/submission', SUBMISSIONS[family], "echo 'v6.3 smoke submission' > README.md", 'cd /workspace']
    if task.startswith('blackbox-cubature-v3'):  # samples answered by the host-side service through the mounted socket
        lines += ['python3 /workspace/tools/local_eval.py --only H1-d8 asian 2>&1 | tail -4']
    if task in NETWORK_TASKS:  # through the proxy: literature reachable, the refused destinations not
        lines += [f'echo "NET-{name}=$(curl -sS -m 20 -o /dev/null -w %{{http_code}} {url} 2>/dev/null)"'
                  for name, url in NETWORK_PROBES.items()]
    if task in LEAN_TASKS:  # Lean in the container: the template passes the local formal check
        lines += ['echo "memory.max=$(cat /sys/fs/cgroup/memory.max)"; command -v lean lake',
                  'cp -r /workspace/examples/lean_project /workspace/submission/theory/formal && '
                  'chmod -R u+w /workspace/submission/theory/formal',
                  'python3 /workspace/tools/lean_check.py 2>&1 | tail -8']
    if task in ('ihvp-sequence-v2', 'ihvp-sequence-v3', 'ihvp-sequence-v4'):  # samples run through the host-side service
        lines += ['python3 /workspace/samples/run_sample.py smoke', 'python3 /workspace/samples/run_sample.py sample_02']
    if setting == 'iterative':
        lines += ['/opt/frontieram/bin/submit --test; echo "test-exit=$?"',
                  '/opt/frontieram/bin/submit --test; echo "second-test-exit=$?"',
                  'for i in $(seq 1 200); do [ -f /workspace/feedback/test-01/report.md ] && break; sleep 6; done',
                  '/opt/frontieram/bin/submit --status; echo "status-exit=$?"',
                  'echo REPORT-BEGIN; cat /workspace/feedback/test-01/report.md; echo REPORT-END',
                  "echo 'v6.3 smoke submission (final)' > /workspace/submission/README.md",
                  '/opt/frontieram/bin/submit --final; echo "final-exit=$?"', 'sleep 120', 'echo NOT-REACHED']
    elif setting == 'single-deadline':  # the agent never ends its turn: the limit must freeze it
        lines += ['ls /opt/frontieram/bin/submit 2>&1; echo single-done', 'sleep 3600', 'echo NOT-REACHED']
    else:
        lines += ['ls /opt/frontieram/bin/submit 2>&1; echo single-done']
    return '\n'.join(lines) + '\n'


def main():
    task_name, setting, seconds, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), Path(sys.argv[4]).resolve()
    layout, broker, runtime = context()
    task = TASKS[task_name]
    out.mkdir(parents=True)
    with broker.acquire('dev', task.dev_memory, f'v6.3 smoke {out.name}', timeout=3600) as lease:
        spec = {'task': task_name, 'model': 'gpt-6.1-sol', 'placement': placement_for(lease, layout).__dict__,
                'output': str(out), 'protocol': 'v6.3', 'setting': setting.replace('-deadline', ''),
                'development_seconds': seconds,
                'agent_script': script(task_name, setting)}
        result = develop(spec, runtime)
    print(json.dumps({k: v for k, v in result.items() if k != 'route'}, indent=1))
    log = (out / 'cli.log').read_text(errors='replace')
    listing = re.search(r'^WORKSPACE-BEGIN\n(.*?)^WORKSPACE-END', log, re.M | re.S)  # the output, not the xtrace lines
    workspace = listing.group(1) if listing else ''
    prompt = (out / 'prompt.txt').read_text()
    layout = AUTHOR_LAYOUT.get(task_name)
    checks = {'frozen': result.get('source_frozen') is True,
              **({'author_layout': all(path in workspace for path in layout)} if layout else
                 {'no_public_dev': 'public-dev' not in workspace and 'dev_full' not in workspace,
                  'samples_present': '/samples' in workspace}),
              'limit_in_prompt': f'{int(seconds // 60) if seconds >= 60 else 0} minutes' in prompt or 'minutes' in prompt}
    if task_name.startswith('blackbox-cubature-v3'):  # the local evaluator reached the sample service and scored
        checks.update(sample_service=' e=' in log and 'FAILED' not in log.split('WORKSPACE-END', 1)[-1])
    if task.spec.get('theory_only'):  # no samples and no sample service
        checks.update(no_samples='/workspace/samples' not in workspace and not (out / 'sample-service.log').exists())
    if task_name in NETWORK_TASKS:
        checks.update(network_literature='NET-arxiv=200' in log,
                      network_refused=all(f'NET-{name}=000' in log for name in ('github', 'pypi', 'openai', 'intranet', 'optbound')),
                      network_logged=(out / 'netproxy.jsonl').exists() and 'deny:github_or_mirror' in (out / 'netproxy.jsonl').read_text())
    if task_name in LEAN_TASKS:
        checks.update(memory_8gib='memory.max=8589934592' in log,
                      lean_check='LEAN-CHECK: 1/1 proved results accepted' in log)
    if setting == 'iterative':
        report = log.rsplit('REPORT-BEGIN\n', 1)[-1].split('REPORT-END', 1)[0]
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
        if task_name in ('ihvp-sequence-v4', 'ihvp-sequence-v5'):  # aggregate-only feedback: the total row and nothing per group or case
            rows = [l for l in report.splitlines() if l.startswith('| ') and not l.startswith(('| category', '| ---'))]
            checks.update(aggregate_only=not ids.get('categories') and bool(rows) and all('**total**' in l for l in rows))
    if task_name in ('ihvp-sequence-v4', 'ihvp-sequence-v5'):
        files = sorted(l for l in workspace.splitlines() if l.startswith('/workspace/') and '.' in l.rsplit('/', 1)[-1]
                       and not l.startswith(('/workspace/submission', '/workspace/feedback')))
        checks.update(workspace_files=files == ([f'/workspace/{name}' for name in sorted(V5_FILES)] if task_name == 'ihvp-sequence-v5' else ['/workspace/problem.md', '/workspace/samples/README.md', '/workspace/samples/run_sample.py',
                                          '/workspace/samples/smoke.json', '/workspace/samples/support/api.py',
                                          '/workspace/samples/support/worker.py']))
    else:
        if setting != 'iterative':
            expected = 'development_deadline' if setting == 'single-deadline' else 'turn_completed'
            checks.update(no_submit_client='No such file or directory' in log,
                          stop_reason=result.get('agent_stop_reason') == expected)
        checks.update(within_limit=result.get('development_seconds', 1e9) <= seconds + 15, never_reached='NOT-REACHED' not in log)
    print(json.dumps(checks))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
