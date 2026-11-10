import sys, json, subprocess, time, os
sys.path.insert(0, '/workspace/public-dev/io')
from model import load_problem
from check import check_solution
import numpy as np
man = json.load(open('/workspace/public-dev/manifest.json'))
inst = {i['id']: i for i in man['instances']}
def run(path, tl=None, verbose=False):
    out = '/tmp/t/out_%d.npz' % os.getpid()
    if os.path.exists(out): os.remove(out)
    env = dict(os.environ)
    if verbose: env['CSOLVER_VERBOSE'] = '1'
    t = time.time()
    try:
        p = subprocess.run(['/workspace/submission/run.sh', '--input', path, '--output', out], env=env, capture_output=True, text=True, timeout=tl)
        el = time.time() - t
        err = p.stderr
    except subprocess.TimeoutExpired:
        return 'timeout', time.time() - t, '', ''
    if verbose: print(err[-6000:])
    try:
        prob = load_problem(path)
        with np.load(out, allow_pickle=False) as d:
            meta = json.loads(str(d['metadata_json']))
            r = {k: meta[k] for k in ('status', 'objective_primal', 'objective_dual') if k in meta}
            r.update({k: d[k] for k in ('x', 's', 'z') if k in d})
        rep = check_solution(prob, r, tol=1e-8)
        return ('PASS' if rep['passed'] else 'FAIL'), el, r.get('status'), rep.get('reason')
    except Exception as e:
        return 'ERR', el, str(e)[:100], ''
if __name__ == '__main__':
    args = sys.argv[1:]
    verbose = False
    if args and args[0] == '-v': verbose = True; args = args[1:]
    for a in args:
        if a in inst:
            path = '/workspace/public-dev/' + inst[a]['path']; tl = inst[a]['time_limit_seconds']; g = inst[a]['group']
        else:
            path = a; tl = None; g = ''
        res, el, st, why = run(path, tl=min(30, 2*tl+1) if tl else 60, verbose=verbose)
        lim = (' lim %.2f' % tl) if tl else ''
        print('%-22s %-12s %-5s %6.3fs%s %s %s' % (os.path.basename(a), g, res, el, lim, st, why), flush=True)
