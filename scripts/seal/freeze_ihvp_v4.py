#!/usr/bin/env python3
"""Freeze public-qualified v4 evidence and methods before any private generation."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
CURATOR=ROOT/'tasks/ihvp-sequence/versions/v4/curator'
sys.path.insert(0,str(CURATOR))
from generate import design_hash,digest,local_only


def freeze(a):
    qualification=json.loads(a.qualification.read_text());bench=json.loads(a.benchmark.read_text())
    run=json.loads(a.benchmark.with_name('run_manifest.json').read_text())
    required={'cold_cg','lbfgs_pcg','frozen_v3_iterative'}
    if qualification.get('status')!='PASS' or qualification.get('split')!='public' or qualification.get('sequence_count')!=15 or qualification.get('basis_trials',0)<15 or qualification.get('basis_accepted')!=0:
        raise ValueError('complete public mathematical / frozen-attack qualification is missing')
    trials=[trial for row in qualification['cases'] for trial in row['basis_trials']]
    if any(t.get('accepted') or t.get('depth') != 0 or t.get('observed',{}).get('residual',0) <= .7 for t in trials):
        raise ValueError('known fixed-pair attack must be rejected by its structural residual, not merely a time limit')
    for field,path in (('audit_source_sha256',ROOT/'scripts/analyze/ihvp_v4_qualify.py'),
                       ('independent_source_sha256',ROOT/'scripts/analyze/ihvp_routed_diagnostics.py')):
        if qualification[field] != digest(path):raise ValueError('audit source changed after qualification')
    if run['harness_sha256'] != digest(ROOT/'scripts/references/ihvp_routed_benchmark.py'):
        raise ValueError('benchmark harness changed after measurement')
    if not bench.get('complete_full_manifest') or bench.get('completed_cases')!=15 or set(bench['methods'])!=required:
        raise ValueError('complete three-method public benchmark required')
    if qualification['manifest_sha256']!=run['manifest_sha256']:
        raise ValueError('qualification/benchmark use different public data')
    if qualification['model_sha256']!=digest(CURATOR/'model.py'):
        raise ValueError('model changed after qualification')
    for name,row in bench['methods'].items():
        if row.get('status')!='ok' or row.get('valid_sequences')!=15 or row.get('resource_compliant_sequences')!=15:
            raise ValueError('invalid or incomplete benchmark method')
    if not bench['methods']['cold_cg']['mean_energy_loss'] > bench['methods']['lbfgs_pcg']['mean_energy_loss'] > 0:
        raise ValueError('public anchor order invalid')
    # Bind actual snapshots and current runtime source to the recorded identity.
    for relative,expected in run['runtime_identity']['source_sha256'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('runtime changed after public measurements')
    for method,configuration in run['methods'].items():
        for relative,expected in configuration['source_sha256'].items():
            if digest(ROOT/relative)!=expected:raise ValueError('solver source changed after public measurement')
    receipt={'status':'FROZEN','version':'v4','created_utc':datetime.now(timezone.utc).isoformat(),
             'design_sha256':design_hash(),'model_sha256':digest(CURATOR/'model.py'),
             'generator_sha256':digest(CURATOR/'generate.py'),
             'public_qualification_sha256':digest(a.qualification),'benchmark_sha256':digest(a.benchmark),
             'benchmark_run_manifest_sha256':digest(a.benchmark.with_name('run_manifest.json')),
             'public_manifest_sha256':run['manifest_sha256'],'methods':run['methods'],
             'policy':'Fresh private seeds after this receipt; no private algorithm tuning; retain every generated case'}
    out=local_only(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    return {'status':'FROZEN','public_sequences':15,'private_to_generate':30,'receipt_sha256':digest(out)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    print(json.dumps(freeze(p.parse_args())))
