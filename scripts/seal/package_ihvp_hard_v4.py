#!/usr/bin/env python3
"""Create host-only v4 handoff ZIP; exact separate six-file learner ZIP required.

No old data, benchmark predictions, secrets outside teachers, or repository
history is copied. NPZ bodies remain host-only and are stored without recompression.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import os
import zipfile
from prepare_ihvp_hard_v4 import ROOT,load_split,samples,sha256,require


def package(a):
    data=a.data_root.resolve()
    public_path,public,ph,ps=load_split(data,'public')
    private_path,private,qh,qs=load_split(data,'private')
    require(not ph & qh and not ps & qs,'public/private overlap')
    evidence=a.evidence.resolve()
    freeze=json.loads((evidence/'design-freeze.json').read_text())
    qualification=json.loads((evidence/'public-qualification.json').read_text())
    private_audit=json.loads((evidence/'private-mathematical-audit.json').read_text())
    bench=json.loads(a.benchmark.read_text())
    require(freeze['status']=='FROZEN' and freeze['design_sha256']==public['design_sha256']==private['design_sha256'],'design mismatch')
    require(freeze['public_qualification_sha256']==sha256(evidence/'public-qualification.json'),'public evidence changed')
    require(freeze['benchmark_sha256']==sha256(a.benchmark),'benchmark changed after freeze')
    require(freeze['public_manifest_sha256']==sha256(public_path),'public pool changed')
    require(private_audit['status']=='PASS' and private_audit['sequence_count']==30 and private_audit['manifest_sha256']==sha256(private_path),'private audit missing')
    require(qualification['sequence_count']==15 and qualification['basis_accepted']==0,'public qualification missing')
    require(bench.get('complete_full_manifest') and bench['completed_cases']==15,'public benchmark incomplete')
    version=ROOT/'tasks/ihvp-sequence/versions/v4'
    require(sha256(version/'curator/model.py')==freeze['model_sha256']==public['model_sha256']==private['model_sha256'],'model mismatch')
    require(sha256(version/'curator/generate.py')==freeze['generator_sha256']==public['generator_sha256']==private['generator_sha256'],'generator mismatch')
    with zipfile.ZipFile(a.learner_zip) as archive:
        allowed={'problem.md','samples/README.md','samples/run_sample.py','samples/smoke.json','samples/support/api.py','samples/support/worker.py'}
        require(set(archive.namelist())==allowed,'learner archive whitelist mismatch')
        require(archive.read('problem.md')==(version/'statement.md').read_bytes(),'stale learner statement')
        for name in allowed-{'problem.md'}:
            require(archive.read(name)==(version/name).read_bytes(),'stale learner support')
    files={}
    def add(name,path):
        path=Path(path)
        require(path.is_file() and not path.is_symlink(),'not a regular delivery file')
        require(name not in files,'duplicate archive name')
        files[name]=path
    for split,manifest in (('public',public),('private',private)):
        add(f'data/{split}/manifest.json',data/split/'manifest.json')
        for entry in manifest['cases']:add(f'data/{split}/{entry["path"]}',data/split/entry['path'])
    # Full-scale samples are references into public, not additional teacher copies.
    sample_bytes=(json.dumps(samples(public),indent=2)+'\n').encode()
    for path in version.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            add('source/'+str(path.relative_to(ROOT)),path)
    scripts=(
        'scripts/seal/prepare_ihvp_hard_v4.py','scripts/seal/package_ihvp_hard_v4.py','scripts/seal/seal_ihvp_v4.py',
        'scripts/seal/build_ihvp_v4_model_visible.py','scripts/seal/freeze_ihvp_v4.py',
        'scripts/analyze/ihvp_routed_diagnostics.py','scripts/analyze/ihvp_v4_qualify.py',
        'scripts/references/ihvp_routed_benchmark.py','scripts/ops/serve_ihvp_v4_samples.py',
        'arena/public_v63.py','arena/operator_tasks.py','tasks/ihvp-sequence/meta.json',
        'arena/operator_drivers/ihvp_oracle.py','arena/operator_drivers/ihvp_hard_v4.py',
        'arena/operator_drivers/ihvp_sequence.py',
        'arena/tests/test_ihvp_routed_v4.py','arena/tests/test_ihvp_routed_integration.py',
        'tasks/ihvp-sequence/analysis/HARD_V4.md')
    for relative in scripts:add('source/'+relative,ROOT/relative)
    for relative in ('tasks/ihvp-sequence/references/r1',
                     'tasks/ihvp-sequence/versions/v1/grader/calibration_references/cold_cg',
                     'tasks/ihvp-sequence/references/candidates/galerkin_cg',
                     'tasks/ihvp-sequence/solutions/v6.3/v3/gpt-6.1-sol-iterative-1/source'):
        for path in (ROOT/relative).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.c','.sh') and '__pycache__' not in path.parts:
                add('source/'+str(path.relative_to(ROOT)),path)
    for name in ('design-freeze.json','public-qualification.json','private-mathematical-audit.json'):
        add('validation/'+name,evidence/name)
    add('validation/summary.json',ROOT/'tasks/ihvp-sequence/analysis/hard_v4_validation_summary.json')
    add('validation/sample-protocol-test.json',evidence/'sample-protocol-test.json')
    add('validation/public-benchmark/aggregate.json',a.benchmark)
    add('validation/public-benchmark/run_manifest.json',a.benchmark.with_name('run_manifest.json'))
    add('README-MAINTAINER.zh.md',version/'USAGE_GUIDE.zh.md')
    for path in (a.learner_zip,a.learner_zip.with_suffix('.manifest.json'),a.learner_zip.with_suffix('.zip.sha256')):
        add('model-visible/'+path.name,path)
    out=a.out.resolve();require(not out.exists(),'refusing to overwrite delivery')
    out.parent.mkdir(parents=True,exist_ok=True)
    records={name:{'sha256':sha256(path),'bytes':path.stat().st_size} for name,path in files.items()}
    records['data/samples/manifest.json']={'sha256':hashlib.sha256(sample_bytes).hexdigest(),'bytes':len(sample_bytes)}
    delivery={'schema':'frontieram/ihvp-v4-host-handoff-1','version':'v4','public':15,'private':30,
              'legacy_scoring_weight':0,'model_visible':'model-visible/'+a.learner_zip.name,
              'status':'draft_target_amd64_acceptance_pending','files':records}
    descriptor=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'wb') as stream,zipfile.ZipFile(stream,'w',allowZip64=True) as archive:
        for name,path in sorted(files.items()):
            archive.write(path,name,compress_type=zipfile.ZIP_STORED if path.suffix=='.npz' else zipfile.ZIP_DEFLATED)
        archive.writestr('data/samples/manifest.json',sample_bytes)
        archive.writestr('DELIVERY.json',json.dumps(delivery,indent=2)+'\n')
        archive.writestr('SHA256SUMS',''.join(f'{r["sha256"]}  {name}\n' for name,r in sorted(records.items())))
    with zipfile.ZipFile(out) as archive:
        require(archive.testzip() is None,'archive CRC failure')
        for name,record in records.items():
            h=hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
            require(h.hexdigest()==record['sha256'],'archive source bytes mismatch')
    require(out.stat().st_size < 2*1024**3,'exceeds single GitHub release asset size')
    checksum=sha256(out);out.with_suffix('.zip.sha256').write_text(f'{checksum}  {out.name}\n')
    return {'status':'PASS','version':'v4','public':15,'private':30,'bytes':out.stat().st_size,'sha256':checksum,'archive_members':len(records)+2}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--learner-zip',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    print(json.dumps(package(p.parse_args())))
