#!/usr/bin/env python3
"""Seal an already imported v4 cohort on its Linux amd64 target (does not activate scoring)."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from prepare_ihvp_hard_v4 import ROOT,load_split,samples,sha256,require
from build_ihvp_v4_model_visible import ALLOWLIST,VERSION


def public_material(destination):
    with tarfile.open(destination,'x') as tar:
        for name,relative in sorted(ALLOWLIST.items()):
            body=(VERSION/relative).read_bytes();info=tarfile.TarInfo(name)
            info.size,info.mode,info.mtime=len(body),0o444,0
            tar.addfile(info,io.BytesIO(body))
    with tarfile.open(destination) as tar:
        require(set(tar.getnames())==set(ALLOWLIST),'public material is not the exact whitelist')


def seal(args):
    data=args.data_root.resolve();require(data.is_relative_to(ROOT),'sealed data root must be inside ignored repository storage')
    p,public,ph,ps=load_split(data,'public');q,private,qh,qs=load_split(data,'private')
    require(not ph&qh and not ps&qs,'split overlap')
    require(public.get('version')==private.get('version')=='v4','run the v4 importer first')
    expected=samples(public);manifest=data/'samples/manifest.json'
    require(json.loads(manifest.read_text())==expected,'sample references changed')
    spec_path=VERSION/'task.json';spec=json.loads(spec_path.read_text());tag=args.image or spec['images']['task']['tag']
    info=json.loads(subprocess.run(['docker','image','inspect',tag],check=True,capture_output=True,text=True).stdout)[0]
    require(info['Architecture']=='amd64' and info['Os']=='linux','v4 target image must be Linux amd64')
    sealed=ROOT/'.runtime-ops'/spec['name'];require(not sealed.exists(),'refusing to replace an existing seal')
    sealed.mkdir(parents=True)
    public_material(sealed/'public.tar')
    relative=data.relative_to(ROOT)
    for split in ('public','private'): (data/split).chmod(0o700)
    by_id={e['id']:e for e in public['cases']}
    service={'smoke':'tasks/ihvp-sequence/versions/v4/samples/smoke.json'}
    service.update({e['id']:str(relative/'public'/by_id[e['source_public_feedback_id']]['path']) for e in expected['cases']})
    lock={'name':spec['name'],'status':spec['status'],'sealed_utc':datetime.now(timezone.utc).isoformat(),
          'spec_sha256':sha256(spec_path),'image':{'tag':tag,'id':info['Id'],'replaces':[]},
          'public_tar_sha256':sha256(sealed/'public.tar'),'samples_manifest_sha256':sha256(manifest),
          'data':{split:str(relative/split) for split in ('public','private')},
          'data_manifest_sha256':{'public':sha256(p),'private':sha256(q)},
          'data_present':{'public':True,'private':True},'roots':{},'sample_service':service,
          'activation':'This seal preserves task status; calibration and isolation acceptance are separate.'}
    (sealed/'lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    return {'status':'SEALED_NOT_ACTIVATED','task':spec['name'],'public':15,'private':30,
            'learner_files':6,'image_id':info['Id']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--image')
    print(json.dumps(seal(p.parse_args())))
