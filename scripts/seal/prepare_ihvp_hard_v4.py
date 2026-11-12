#!/usr/bin/env python3
"""Import a curator's fresh v4 cohort, preserving bytes and creating public sample references.

Host only. Never pass this source, manifests, or output data to a candidate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'frontieram/ihvp-routed-factored-spd-4'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def contained(root, path):
    root, path = Path(root).resolve(), Path(path).absolute()
    require(path.is_relative_to(root), 'path escapes data root')
    current = root
    for part in path.relative_to(root).parts:
        require(part not in ('.','..'), 'unsafe path component')
        current = current / part
        require(not current.is_symlink(), 'symlink in data path')
    require(path.resolve().is_relative_to(root), 'resolved path escapes root')
    return path


def load_split(data, split):
    root = contained(data, Path(data)/split)
    manifest_path = contained(root, root/'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get('cases', [])
    require(manifest.get('split') == split, 'split mismatch')
    require(len(records) == {'public':15,'private':30}[split], 'v4 requires fresh 15:30')
    require(manifest.get('design_sha256') and manifest.get('model_sha256'), 'missing source binding')
    ids, hashes, seeds = set(),set(),set()
    for record in records:
        require(record['id'] not in ids, 'duplicate ID'); ids.add(record['id'])
        rel = Path(record['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe teacher path')
        path = contained(root,root/rel)
        require(path.is_file() and sha256(path) == record['sha256'], 'teacher SHA mismatch')
        require(record['sha256'] not in hashes,'duplicate body'); hashes.add(record['sha256'])
        with np.load(path,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata_json'].item()))
        require(meta['schema'] == SCHEMA and meta['split'] == split and meta['id'] == record['id'], 'teacher identity mismatch')
        require(meta['dimension'] == record['dimension'] == 16384 and meta['steps'] == record['steps'] == 50, 'wrong full-scale dimensions')
        require(meta['design_sha256'] == manifest['design_sha256'], 'mixed design')
        require(meta['seed'] not in seeds,'duplicate seed'); seeds.add(meta['seed'])
    return manifest_path, manifest, hashes, seeds


def samples(public):
    return {'schema':'frontieram/ihvp-v4-sample-references-1','version':'v4','split':'samples',
            'cases':[{'id':f'sample_{i:02d}','source_public_feedback_id':record['id'],
                      'source_teacher_sha256':record['sha256'],'sha256':record['sha256'],
                      'dimension':16384,'steps':50} for i,record in enumerate(
                          (public['cases'][0],public['cases'][-1]),1)],
            'selection':'First/last public case, fixed before solver qualification; no copied teachers'}


def import_bundle(bundle, out):
    data = Path(bundle).resolve()/'data'
    public_path,public,ph,ps=load_split(data,'public')
    private_path,private,qh,qs=load_split(data,'private')
    require(not ph & qh and not ps & qs,'public/private overlap')
    require(public['design_sha256'] == private['design_sha256'],'split design mismatch')
    out=Path(out).resolve()
    require(not out.exists(),'refuse to replace an existing data root')
    if out.is_relative_to(ROOT):
        require(subprocess.run(['git','check-ignore','-q','--',str(out)],cwd=ROOT).returncode==0,'output must be ignored')
    out.mkdir(parents=True,mode=0o700)
    try:
        for split,manifest in (('public',public),('private',private)):
            for record in manifest['cases']:
                source=contained(data,data/split/record['path']);dest=out/split/record['path']
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,dest)
                require(sha256(dest)==record['sha256'],'copied teacher differs')
            manifest={**manifest,'version':'v4'}
            (out/split/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (out/split).chmod(0o700)
        (out/'samples').mkdir(mode=0o700)
        (out/'samples/manifest.json').write_text(json.dumps(samples(public),indent=2)+'\n')
        (out/'IMPORT.json').write_text(json.dumps({'version':'v4','public':15,'private':30,
            'legacy_scoring_weight':0,'source_public_manifest_sha256':sha256(public_path),
            'source_private_manifest_sha256':sha256(private_path)},indent=2)+'\n')
    except BaseException:
        # Preserve a failed import for inspection; never silently bless partial data.
        (out/'IMPORT_FAILED').touch()
        raise
    return {'status':'PASS','version':'v4','public':15,'private':30,'legacy_scoring_weight':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'var/raw/ihvp-sequence-v4-import/data')
    args=p.parse_args();print(json.dumps(import_bundle(args.bundle_root,args.out)))

if __name__=='__main__':main()
