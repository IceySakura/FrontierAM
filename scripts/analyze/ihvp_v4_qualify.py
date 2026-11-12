#!/usr/bin/env python3
"""Host-only public qualification: audit teachers and replay the frozen v3 basis learner.

No private algorithm screening. The basis program is unmodified; a Python
return trace observes its rejection residual without changing control flow.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analyze'))
from ihvp_routed_diagnostics import IndependentHardCase,sha256,galerkin_projection


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def audit(manifest_path,history=0,basis_source=None):
    manifest=json.loads(manifest_path.read_text())
    if basis_source is not None and manifest['split'] != 'public':
        raise ValueError('basis attack regression is public only')
    if history and manifest['split'] != 'public':
        raise ValueError('history qualification is public only; private audit checks mathematics only')
    model=load(ROOT/'tasks/ihvp-sequence/versions/v4/curator/model.py','v4_audit_model')
    basis=load(basis_source/'basis.py','frozen_v3_basis') if basis_source else None
    rows=[]
    for entry in manifest['cases']:
        path=(manifest_path.parent/entry['path']).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()) or sha256(path)!=entry['sha256']:
            raise ValueError('teacher identity mismatch')
        independent=IndependentHardCase(path);production=model.HardCase(path)
        residuals=[];factor_errors=[];reference_errors=[];cross_errors=[];history_losses=[];basis_trials=[]
        rng=np.random.default_rng(1919810)
        for t in range(independent.steps):
            z=independent.reference(t);b=independent.rhs(t)
            residuals.append(float(np.linalg.norm(independent.hvp(t,z)-b)/np.linalg.norm(b)))
            factor_errors.append(independent.reference_factor_loss(t))
            reference_errors.append(float(np.linalg.norm(independent.independent_reference(t)-z)/np.linalg.norm(z)))
            if independent.score(t,z)['energy_loss'] != 0 or abs(independent.score(t,np.zeros_like(z))['energy_loss']-1)>1e-12:
                raise ArithmeticError('energy endpoints failed')
            if t in (0,independent.steps//2,independent.steps-1):
                probes=rng.normal(size=(independent.dimension,3));expected=independent.hvp(t,probes)
                cross_errors.append(float(np.linalg.norm(production.hvp(t,probes)-expected)/np.linalg.norm(expected)))
            if history and t:
                past=np.column_stack([independent.reference(j) for j in range(max(0,t-history),t)])
                history_losses.append(galerkin_projection(z,past,lambda v:independent.hvp(t,v))['relative_H_energy_error_squared'])
            if basis is not None and (t==0 or independent.phase(t)!=independent.phase(t-1)):
                learner=basis.LearnedBasis(independent.dimension,1)
                observed={}
                def trace(frame,event,arg):
                    if frame.f_code is basis.LearnedBasis.fit.__code__:
                        if event=='return':
                            observed.update({k:float(frame.f_locals[k]) for k in ('residual','initial','level') if k in frame.f_locals and frame.f_locals[k] is not None})
                        return trace
                    return None
                start=time.monotonic();prior=sys.gettrace()
                try:
                    sys.settrace(trace)
                    accepted=learner.fit(lambda v:production.hvp(t,v),production.mu,start+30)
                finally:sys.settrace(prior)
                basis_trials.append({'step':t,'accepted':bool(accepted),'depth':int(learner.depth),
                    'seconds':time.monotonic()-start,'observed':observed})
        row={'id':entry['id'],'teacher_sha256':entry['sha256'],
             'max_reference_relative_residual':max(residuals),'max_factor_energy_loss':max(factor_errors),
             'max_reconstructed_reference_l2_difference':max(reference_errors),
             'max_independent_backend_relative_difference':max(cross_errors),
             'factor_spectral_certificate':independent.factor_spectral_certificate,
             'history_size':history,'history_projection_mean_loss':float(np.mean(history_losses)) if history_losses else None,
             'basis_trials':basis_trials}
        if row['max_reference_relative_residual']>=1e-7 or row['max_factor_energy_loss']>=1e-16 or row['max_reconstructed_reference_l2_difference']>=1e-9 or row['max_independent_backend_relative_difference']>=1e-10:
            raise ArithmeticError('independent audit failed')
        rows.append(row)
        print(json.dumps({'id':entry['id'],'status':'PASS','basis_accepted':sum(x['accepted'] for x in basis_trials)}),flush=True)
    result={'status':'PASS','split':manifest['split'],'sequence_count':len(rows),
        'system_count':sum(int(e['steps']) for e in manifest['cases']),
        'manifest_sha256':sha256(manifest_path),'audit_source_sha256':sha256(__file__),
        'independent_source_sha256':sha256(ROOT/'scripts/analyze/ihvp_routed_diagnostics.py'),
        'model_sha256':sha256(ROOT/'tasks/ihvp-sequence/versions/v4/curator/model.py'),
        'cases':rows,'max_reference_relative_residual':max(r['max_reference_relative_residual'] for r in rows),
        'max_independent_backend_relative_difference':max(r['max_independent_backend_relative_difference'] for r in rows),
        'basis_trials':sum(len(r['basis_trials']) for r in rows),
        'basis_accepted':sum(x['accepted'] for r in rows for x in r['basis_trials']),
        'scope':'Mathematical audit; optional public-only frozen attack regression; no formal score or unlearnability claim'}
    if basis_source:
        result['basis_source_sha256']={p.name:sha256(p) for p in sorted(basis_source.iterdir()) if p.is_file() and p.suffix in ('.py','.c','.sh','.so')}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--history',type=int,default=0)
    p.add_argument('--basis-source',type=Path)
    a=p.parse_args();result=audit(a.manifest.resolve(),a.history,a.basis_source)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')

if __name__=='__main__':main()
