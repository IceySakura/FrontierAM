#!/usr/bin/env python3
"""Draft v4 IHVP local smoke/recalibration CLI. No formal isolation is claimed.

Example: python evaluate.py --smoke --out var/outputs/ihvp-smoke
Target-machine measurement: --manifest ... --recalibrate --repeats 2 --out ...
Missing data, calibration, environment evidence or release gates is unmeasured.
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
import random
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
VERSION=HERE.parent
TASK=VERSION.parents[1]
REPO=VERSION.parents[3]
sys.path.insert(0,str(HERE))
from runtime.api import run_case,clean
from runtime.backend import Case
from scoring import score_result,RELEASE_GATES


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
    cpu=platform.processor() or platform.machine()
    if sys.platform=='darwin':
        try:cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True,timeout=2,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.SubprocessError):pass
    elif Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith(('model name','Hardware')):cpu=line.split(':',1)[-1].strip();break
    files=(list(HERE.rglob('*.py'))+list((TASK/'references'/'r1').glob('*.py'))
           +list((VERSION/'curator').glob('*.py'))
           +[VERSION/'requirements.lock',VERSION/'Dockerfile'])
    hashes={str(path.relative_to(REPO)):file_sha(path) for path in sorted(files) if '__pycache__' not in path.parts}
    runtime={'platform':platform.platform(),'machine':platform.machine(),'cpu_model':cpu,
             'python':sys.version,'python_executable_sha256':file_sha(sys.executable),
             'numpy':np.__version__,'scipy':scipy.__version__,'psutil':psutil.__version__,
             'blas':threadpool_info(),'source_sha256':hashes,
             'execution':'local child process, protocol isolation only; not adversarial worker isolation'}
    fingerprint=hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()
    return runtime,fingerprint


def normalize_case_result(value):
    if value['valid']:status='ok'
    elif value.get('infrastructure_failure') is True:status='infrastructure_failure'
    else:status='submission_failure'
    return {'status':status,'raw_error':value.get('mean_energy_loss'),
            'reason':value.get('termination_reason') or (str(value.get('solver_errors')) if value.get('solver_errors') else None),**value}


def aggregate(rows):
    if any(row['status']=='infrastructure_failure' for row in rows):status='infrastructure_failure'
    elif any(row['status']!='ok' for row in rows):status='submission_failure'
    else:status='ok'
    values=[row.get('raw_error') for row in rows]
    return {'status':status,'raw_error':float(np.mean(values)) if values and all(v is not None for v in values) else None,
            'sequences':len(rows),'max_wall_seconds':max((r['wall_seconds'] for r in rows),default=None),
            'mean_wall_seconds':float(np.mean([r['wall_seconds'] for r in rows])) if rows else None,
            'max_peak_rss_mib':max((r['peak_rss_mib'] for r in rows),default=None),
            'reason':next((r.get('reason') for r in rows if r.get('reason')),None)}


def run(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    runtime,fingerprint=identity()
    weak=TASK/'versions'/'v1'/'grader'/'calibration_references'/'cold_cg';strong=TASK/'references'/'r1'
    if args.smoke:
        sample=VERSION/'samples'/'smoke.json'
        entries=[{'id':'smoke','path':str(sample),'sha256':file_sha(sample)}]
        manifest_root=VERSION
        data_fingerprint=file_sha(sample)
    else:
        if args.manifest is None or not args.manifest.is_file():
            result={'status':'missing_data','raw_error':None,'reason':'maintainer manifest is missing'}
            scored=score_result(result,None,runtime_fingerprint=fingerprint,data_fingerprint='missing')
            payload={'result':result,'score':scored,'runtime_fingerprint':fingerprint}
            write_json(out/'result.json',payload);return payload
        manifest=json.loads(args.manifest.read_text());manifest_root=args.manifest.resolve().parent
        entries=manifest['cases'];data_fingerprint=file_sha(args.manifest)
    if not entries:raise ValueError('empty data manifest')
    complete_count=len(entries)
    if args.case_limit:entries=entries[:args.case_limit]
    methods={'cold_cg':weak,'lbfgs_pcg':strong} if args.recalibrate else {'candidate':(args.submission or strong).resolve()}
    for method,source in methods.items():
        if not (source/'solver.py').is_file():
            if args.recalibrate or args.submission is None:
                raise FileNotFoundError(f'maintainer reference {method}: solver.py missing')
            result={'status':'submission_failure','raw_error':None,'reason':'candidate solver.py missing'}
            calibration=json.loads(args.calibration.read_text()) if args.calibration and args.calibration.exists() else None
            if args.smoke or len(entries)!=complete_count:
                scored={'status':'unmeasured','score':None,'native_score':None,'reason':'sample/partial diagnostic only','raw_error':None}
            else:
                scored=score_result(result,calibration,runtime_fingerprint=fingerprint,data_fingerprint=data_fingerprint)
            payload={'result':result,'score':scored,'runtime_fingerprint':fingerprint,'data_fingerprint':data_fingerprint}
            write_json(out/'result.json',payload);return payload
        if out.is_relative_to(source):raise ValueError('output must be outside submission source')
    repeats=args.repeats if args.recalibrate else 1
    if repeats<1:raise ValueError('repeats must be positive')
    protocol={'dimension':16384,'steps':50,'wall_seconds':args.wall_seconds,'memory_mib':256,
              'single_thread':True,'hvp_limit':None,'max_block_columns':64,
              'oracle_allocation':'sequence deadline; per-step allocation chosen by solver',
              'seed':1,'local_only':True}
    frozen={'runtime_identity':runtime,'runtime_fingerprint':fingerprint,'data_fingerprint':data_fingerprint,
            'protocol':protocol,'sources':{k:source_sha(v) for k,v in methods.items()},'options':json.loads(args.options_json),
            'methods':list(methods),'repeats':repeats,'cases':len(entries),'full_cases':complete_count,
            'scope':'smoke' if args.smoke else 'maintainer_local_measurement'}
    manifest_path=out/'run_manifest.json'
    if manifest_path.exists():
        if not args.resume:raise FileExistsError('run exists; use --resume')
        if json.loads(manifest_path.read_text())!=frozen:raise ValueError('resume fingerprint/protocol/source mismatch')
    else:write_json(manifest_path,frozen)
    all_rows={name:[] for name in methods}
    for repeat in range(repeats):
        for index,entry in enumerate(entries):
            case_id=str(entry.get('id',entry.get('case','')))
            if not case_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in case_id):raise ValueError('unsafe case id')
            path=Path(entry['path']);path=path if path.is_absolute() else manifest_root/path
            path=path.resolve()
            if not args.smoke and not path.is_relative_to(manifest_root):raise ValueError('case escaped data root')
            if 'sha256' in entry and file_sha(path)!=entry['sha256']:raise ValueError('teacher bytes differ from manifest')
            case=Case(path)
            if not args.smoke and (case.dimension!=16384 or case.steps!=50):raise ValueError('release cases require d=16384,T=50')
            order=list(methods);random.Random(20261110+index+100*repeat).shuffle(order)
            for method in order:
                target=out/f'{case_id}__{method}__r{repeat}.json'
                if target.exists():
                    if not args.resume:raise FileExistsError(target)
                    row=json.loads(target.read_text())
                    if row.get('runtime_fingerprint')!=fingerprint or row.get('data_fingerprint')!=data_fingerprint:
                        raise ValueError('record freeze mismatch')
                else:
                    measured=run_case(methods[method],case,wall_seconds=args.wall_seconds,memory_mib=256.,seed=1,
                                      options={'rank':16,'maxiter_factor':50,**json.loads(args.options_json)},log_path=out/f'{case_id}__{method}__r{repeat}.stderr.log')
                    prediction=measured.pop('prediction_array')
                    np.save(target.with_suffix('.npy'),prediction,allow_pickle=False)
                    row=normalize_case_result(measured)
                    row.update(case=case_id,method=method,repeat=repeat,runtime_fingerprint=fingerprint,data_fingerprint=data_fingerprint)
                    write_json(target,row)
                all_rows[method].append(row)
                print(json.dumps({k:row[k] for k in ('case','method','repeat','status','raw_error','wall_seconds','peak_rss_mib')}),flush=True)
            del case
    results={name:aggregate(rows) for name,rows in all_rows.items()}
    if args.recalibrate:
        E0,E5=results['cold_cg']['raw_error'],results['lbfgs_pcg']['raw_error']
        ordered=E0 is not None and E5 is not None and E0>E5>0
        valid=all(value['status']=='ok' for value in results.values())
        repeat_losses={method:[aggregate([row for row in rows if row['repeat']==r])['raw_error'] for r in range(repeats)] for method,rows in all_rows.items()}
        stable=repeats>=2 and valid and all(max(v)-min(v)<=.05*max(max(v),1e-30) for v in repeat_losses.values())
        gates={name:False for name in RELEASE_GATES}
        calibration={'status':'measured_pending_activation' if valid and ordered and not args.smoke and len(entries)==complete_count else 'measurement_incomplete',
                     'E0':E0,'E5':E5,'runtime_fingerprint':fingerprint,'data_fingerprint':data_fingerprint,
                     'runtime_identity':runtime,'protocol':protocol,'release_gates':gates,
                     'diagnostic_probes':{'reference_headroom_observed':bool(valid and results['lbfgs_pcg']['max_wall_seconds']<=.75*30),
                                          'raw_loss_spread_within_5pct':bool(stable)},
                     'repeat_losses':repeat_losses,'results':results,
                     'headroom_definition':'r1 max wall <= 22.5s (25% of the 30s deadline)',
                     'stability_probe_definition':'5% raw-loss spread is diagnostic only, not the repository two-run score-consistency release gate; maintainer must declare and verify its score tolerance',
                     'activation':'maintainer must verify amd64 offline image, real worker isolation, non-saturation and all gates; this CLI never activates formal scoring'}
        write_json(args.calibration or out/'calibration.json',calibration)
        payload={'status':calibration['status'],'results':results,'calibration':calibration}
    else:
        calibration=json.loads(args.calibration.read_text()) if args.calibration and args.calibration.exists() else None
        result=results['candidate']
        if args.smoke or len(entries)!=complete_count:
            score={'status':'unmeasured','score':None,'native_score':None,'reason':'sample/partial diagnostic only','raw_error':result['raw_error']}
        else:score=score_result(result,calibration,runtime_fingerprint=fingerprint,data_fingerprint=data_fingerprint)
        payload={'result':result,'score':score,'runtime_fingerprint':fingerprint,'data_fingerprint':data_fingerprint}
    write_json(out/'result.json',payload);return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--submission',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--recalibrate',action='store_true')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--case-limit',type=int)
    parser.add_argument('--options-json',default='{}',help='Frozen solver options as a JSON object')
    parser.add_argument('--wall-seconds',type=float,default=30)
    args=parser.parse_args()
    if args.smoke and args.wall_seconds==30:args.wall_seconds=3
    if (not args.smoke and args.wall_seconds!=30) or not 1<=args.wall_seconds<=30:
        parser.error('release sequences use 30s; shortened budget is smoke-only')
    try:payload=run(args)
    except Exception as error:
        payload={'result':{'status':'infrastructure_failure','raw_error':None,'reason':f'{type(error).__name__}: {error}'},
                 'score':{'status':'unmeasured','score':None,'native_score':None,'reason':'grader/platform setup failure'}}
        write_json(args.out/'result.json',payload)
    print(json.dumps(clean(payload),ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
