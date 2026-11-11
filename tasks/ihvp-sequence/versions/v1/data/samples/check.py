#!/usr/bin/env python3
"""Standalone samples-only oracle check. It never issues a formal score.

python check.py --case smoke.json --submission /workspace/submission --out /tmp/check.json
python check.py --verify-package  # offline image-build integrity assertion
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
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'support'))
from api import run_case,clean,InvalidPrediction


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


class SampleCase:
    """Teacher data remain only in this development evaluator process."""
    def __init__(self,path):
        path=Path(path).resolve();self.smoke=path.suffix.lower()=='.json'
        if self.smoke:
            self.data=json.loads(path.read_text())
            if self.data.get('kind')!='dense_smoke':raise ValueError('JSON case must be dense_smoke')
            self.meta=self.data
            for key in ('H','b','phase'):self.data[key]=np.asarray(self.data[key])
            self.data['z_star']=np.asarray(self.data.get('z_star',self.data.get('x_ref')))
        else:
            import model
            self.model=model;self.data=model.build_operator_data(path);self.meta=self.data['metadata']
        self.dimension=int(self.meta['dimension']);self.steps=int(self.meta['steps'])
        self.mu=float(self.meta['mu']);self.L=float(self.meta['L'])
        if not 0<self.mu<=self.L:raise ValueError('invalid sample spectral bounds')
        if self.data['b'].shape!=(self.steps,self.dimension) or self.data['z_star'].shape!=(self.steps,self.dimension):
            raise ValueError('invalid sample array dimensions')
        if not np.isfinite(self.data['b']).all() or not np.isfinite(self.data['z_star']).all():
            raise ValueError('nonfinite sample reference')
    def rhs(self,t):return np.asarray(self.data['b'][t],dtype=float)
    def phase(self,t):return int(self.data['phase'][t])
    def hvp(self,t,v):return self.data['H'][t]@v if self.smoke else self.model.case_matvec(self.data,t,v)
    def score(self,t,x):
        z=self.data['z_star'][t];b=self.rhs(t)
        with np.errstate(over='ignore',invalid='ignore'):
            denominator=float(z@b);b_norm=float(np.linalg.norm(b))
        if not np.isfinite(denominator) or denominator<=0 or not np.isfinite(b_norm) or b_norm<=0:
            raise ArithmeticError('invalid sample reference denominator/RHS')
        error=x-z;numerator=float(error@self.hvp(t,error))
        residual=float(np.linalg.norm(self.hvp(t,x)-b)/b_norm)
        if not np.isfinite(numerator) or not np.isfinite(residual):raise InvalidPrediction('numerical overflow')
        if numerator < -1e-8*max(1.,denominator):raise InvalidPrediction('invalid numerical energy')
        return {'energy_loss':max(0.,numerator)/denominator,'relative_residual':residual}


def verify_package(manifest_path):
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent
    manifest=json.loads(manifest_path.read_text());entries=manifest.get('cases',[])
    if manifest.get('split')!='samples':raise ValueError('expected a samples manifest')
    if len(entries)!=3:raise ValueError('image must contain one smoke and exactly two full samples')
    identifiers=[];full=0;smoke=0
    for entry in entries:
        identifiers.append(str(entry['id']))
        path=(root/entry['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('sample body is missing or outside package')
        if file_sha(path)!=entry.get('sha256'):raise ValueError('sample body digest mismatch')
        if entry.get('kind')=='dense_smoke':
            data=json.loads(path.read_text())
            if data.get('kind')!='dense_smoke' or int(data['dimension'])!=4 or int(data['steps'])!=3:
                raise ValueError('invalid small interface fixture')
            smoke+=1
        elif entry.get('kind')=='signed_dct':
            with np.load(path,allow_pickle=False) as archive:
                metadata=json.loads(str(archive['metadata_json'].item()))
                if (int(metadata['dimension']),int(metadata['steps']))!=(16384,50):
                    raise ValueError('full sample must have d=16384,T=50')
                if archive['b'].shape!=(50,16384) or archive['z_star'].shape!=(50,16384):
                    raise ValueError('full sample RHS/reference array shape mismatch')
            full+=1
        else:raise ValueError('unknown sample kind')
    if len(set(identifiers))!=3 or full!=2 or smoke!=1:raise ValueError('sample package is incomplete')
    for relative in ('check.py','support/api.py','support/worker.py','support/model.py','support/common.py'):
        if not (root/relative).is_file():raise ValueError('standalone support file missing: '+relative)
    return {'status':'samples_package_verified','full_size_sequences':full,'smoke_sequences':smoke,
            'scope':'samples only; integrity/shape check, no timing benchmark or formal score',
            'score':None,'native_score':None,'manifest_sha256':file_sha(manifest_path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,default=HERE/'smoke.json')
    parser.add_argument('--submission',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--wall-seconds',type=float,default=30.)
    parser.add_argument('--verify-package',action='store_true')
    parser.add_argument('--manifest',type=Path,default=HERE/'manifest.json')
    args=parser.parse_args()
    if not 1<=args.wall_seconds<=30:parser.error('diagnostic wall budget must be in [1,30] seconds')
    if not args.verify_package and args.submission is None:parser.error('--submission is required for a solver check')
    try:
        if args.verify_package:
            result=verify_package(args.manifest);status=0
        else:
            result=run_case(args.submission,SampleCase(args.case),wall_seconds=args.wall_seconds,
                            memory_mib=256.,seed=1,options={})
            result.pop('prediction_array')
            result.update(scope='samples_development_diagnostic_only',score=None,native_score=None,
                          scoring_status='never_formal',
                          timing='includes worker startup, imports, protocol, HVPs, solve and outputs')
            status=0 if result['valid'] else 1
    except Exception as error:
        result={'status':'infrastructure_failure','reason':f'{type(error).__name__}: {error}',
                'scope':'samples_development_diagnostic_only','score':None,'native_score':None}
        status=2
    text=json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
    print(text,end='');return status

if __name__=='__main__':raise SystemExit(main())
