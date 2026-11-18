#!/usr/bin/env python3
"""ihvp-sequence v5 (real training trajectories, neutral statement): run one submission over a sealed pool.

Example: python evaluate.py --smoke --submission DIR --out var/outputs/ihvp-v5-smoke
Sealed pool: --manifest DATA/manifest.json --submission DIR --out DIR
The pool score comes from the platform's weak/strong references measured on the same pool (Warm CG = 0, Cold CG = 50);
this CLI reports raw losses and resource records only.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
VERSION=HERE.parent
REPO=VERSION.parents[3]
sys.path.insert(0,str(HERE))
from runtime.api import run_case,clean,InvalidPrediction
from runtime.backend import RealCase


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):digest.update(part)
    return digest.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def source_sha(root):
    digest=hashlib.sha256()
    for path in sorted(Path(root).rglob('*')):
        if not path.is_file() or any(part in ('__pycache__','.git','.venv') for part in path.parts):continue
        digest.update(str(path.relative_to(root)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def identity():
    import scipy,psutil
    from threadpoolctl import threadpool_info
    cpu=platform.machine()
    if Path('/proc/cpuinfo').exists():
        cpu=next((line.split(':',1)[-1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')),cpu)
    hashes={str(path.relative_to(VERSION)):file_sha(path) for path in sorted(HERE.rglob('*.py')) if '__pycache__' not in path.parts}
    runtime={'platform':platform.platform(),'machine':platform.machine(),'cpu_model':cpu,'python':sys.version,
             'numpy':np.__version__,'scipy':scipy.__version__,'psutil':psutil.__version__,
             'blas':threadpool_info(),'grader_sha256':hashes}
    return runtime,hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()


class SmokeCase:
    """The 4-dimensional interface example of the model package (dense H per step); infrastructure checks only."""
    def __init__(self,path):
        data=json.loads(Path(path).read_text())
        self.dimension,self.steps=int(data['dimension']),int(data['steps'])
        self.mu,self.L=float(data['mu']),float(data['L'])
        self.H=[np.asarray(s['H'],dtype=float) for s in data['systems']]
        self.b=[np.asarray(s['b'],dtype=float) for s in data['systems']]
        self.phases=[int(s['phase']) for s in data['systems']]
        self.x=[np.linalg.solve(H,b) for H,b in zip(self.H,self.b)]
    def rhs(self,t):return self.b[t].copy()
    def phase(self,t):return self.phases[t]
    def hvp(self,t,v):return self.H[t]@np.asarray(v,dtype=float)
    def score(self,t,x):
        err=x-self.x[t];energy=float(err@self.H[t]@err);denominator=float(self.x[t]@self.H[t]@self.x[t])
        if not np.isfinite(energy):raise InvalidPrediction('numerical overflow')
        return {'energy_loss':max(0.,energy)/denominator,
                'relative_residual':float(np.linalg.norm(self.H[t]@x-self.b[t])/np.linalg.norm(self.b[t]))}


def normalize(value):
    if value['valid']:status='ok'
    elif value.get('infrastructure_failure') is True:status='infrastructure_failure'
    else:status='submission_failure'
    return {'status':status,'raw_error':value.get('mean_energy_loss'),
            'reason':value.get('termination_reason') or (str(value.get('solver_errors')) if value.get('solver_errors') else None),**value}


def aggregate(rows,expected,repeats=1):
    if len(rows)!=expected*repeats or any(row['status']=='infrastructure_failure' for row in rows):status='infrastructure_failure'
    elif any(row['status']!='ok' for row in rows):status='submission_failure'
    else:status='ok'
    losses=[row.get('raw_error') for row in rows]
    def stat(key,how):
        values=[row[key] for row in rows if row.get(key) is not None]
        return float(how(values)) if values else None
    return {'status':status,'raw_error':float(np.mean(losses)) if rows and all(v is not None for v in losses) else None,
            'sequences':len(rows),'expected_sequences':expected*repeats,'compliant_sequences':sum(row['status']=='ok' for row in rows),
            'mean_wall_seconds':stat('wall_seconds',np.mean),'max_wall_seconds':stat('wall_seconds',max),
            'mean_peak_rss_mib':stat('peak_rss_mib',np.mean),'max_peak_rss_mib':stat('peak_rss_mib',max),
            'mean_hvp_calls':stat('hvp_calls',np.mean),'max_hvp_calls':stat('hvp_calls',max),
            'repeats':repeats,'repeat_raw_errors':[float(np.mean([r['raw_error'] for r in rows if r['repeat']==k]))
                                                   if all(r.get('raw_error') is not None for r in rows if r['repeat']==k) else None
                                                   for k in range(repeats)],
            'reason':next((r.get('reason') for r in rows if r.get('reason')),None)}


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    runtime,fingerprint=identity()
    submission=args.submission.resolve()
    if args.smoke:
        sample=VERSION/'workspace'/'smoke.json'
        entries=[{'id':'smoke','path':str(sample)}];root=sample.parent;data_fingerprint=file_sha(sample);repeats=1
    else:
        manifest=json.loads(args.manifest.read_text());root=args.manifest.resolve().parent
        entries=manifest['cases'];data_fingerprint=file_sha(args.manifest)
        repeats=int(manifest.get('repeats',1))  # every sequence runs this many times; the pool loss is the mean over all runs
        if len(entries)!=manifest['expected_cases']:raise ValueError('manifest case count differs from its declaration')
    expected=len(entries)
    if args.case_limit:entries=entries[:args.case_limit]
    if not (submission/'solver.py').is_file():
        result={'status':'submission_failure','raw_error':None,'reason':'solver.py missing','sequences':0,'expected_sequences':expected}
        write_json(out/'result.json',{'result':result,'runtime_fingerprint':fingerprint});return result
    if out.is_relative_to(submission):raise ValueError('output must be outside submission source')
    write_json(out/'run_manifest.json',{'runtime_identity':runtime,'runtime_fingerprint':fingerprint,'data_fingerprint':data_fingerprint,
                                        'submission_sha256':source_sha(submission),'wall_seconds':args.wall_seconds,'memory_mib':256,
                                        'cases':len(entries),'full_cases':expected,'repeats':repeats,'scope':'smoke' if args.smoke else 'pool'})
    rows=[]
    for entry in entries:
        case_id=str(entry['id'])
        if not case_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in case_id):raise ValueError('unsafe case id')
        if args.smoke:case=SmokeCase(entry['path'])
        else:
            folder=(root/entry['path']).resolve()
            if not folder.is_relative_to(root):raise ValueError('case escaped data root')
            if file_sha(folder/'COMPLETE.json')!=entry['sha256']:raise ValueError(f'{case_id}: data differs from the manifest')
            case=RealCase(folder)
            if case.dimension!=16384 or case.steps!=50:raise ValueError('pool cases require d=16384, T=50')
        for repeat in range(repeats):
            measured=run_case(submission,case,wall_seconds=args.wall_seconds,memory_mib=256.,seed=1,options={},
                              log_path=out/f'{case_id}__candidate__r{repeat}.stderr.log')
            measured.pop('prediction_array')
            row=normalize(measured);row.update(case=case_id,repeat=repeat,dataset=entry.get('dataset'),profile=entry.get('profile'))
            write_json(out/f'{case_id}__candidate__r{repeat}.json',row);rows.append(row)
            print(json.dumps({k:row.get(k) for k in ('case','repeat','status','raw_error','wall_seconds','peak_rss_mib','hvp_calls')}),flush=True)
        del case
    result=aggregate(rows,expected,repeats)
    if args.smoke or len(entries)!=expected:result['scope']='diagnostic (smoke or partial pool): not a pool result'
    write_json(out/'result.json',{'result':result,'runtime_fingerprint':fingerprint,'data_fingerprint':data_fingerprint})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--submission',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--case-limit',type=int)
    parser.add_argument('--wall-seconds',type=float,default=30)
    args=parser.parse_args()
    if args.smoke and args.wall_seconds==30:args.wall_seconds=3
    if (not args.smoke and args.wall_seconds!=30) or not 1<=args.wall_seconds<=30:
        parser.error('pool sequences use 30 s; a shorter budget is smoke-only')
    if not args.smoke and (args.manifest is None or not args.manifest.is_file()):parser.error('--manifest is required')
    try:result=run(args)
    except Exception as error:
        result={'status':'infrastructure_failure','raw_error':None,'reason':f'{type(error).__name__}: {error}'}
        write_json(args.out/'result.json',{'result':result})
    print(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
