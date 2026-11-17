"""Curator-only split overlays; source case IDs and operator bytes stay unchanged."""
from collections import Counter
from pathlib import Path
import hashlib,json,re

REVISION='balanced-split-20261117'
CALIBRATION_REVISION=REVISION+'-cg50-warm0'
DATASETS={'cifar10','cifar100','fashion_mnist'}
PROFILES={'continuous_fixed','continuous_resampled','medium_resets','short_resets','rapid_resampled'}
SOURCE_IDS={f'{pool}_{i:02d}' for pool,n in [('public',15),('private',30)] for i in range(n)}

class SplitManifest:
    def __init__(self,path):
        self.path=Path(path);raw=self.path.read_bytes();payload=json.loads(raw)
        if payload.get('revision')!=REVISION:raise ValueError('unsupported split revision')
        rows=payload.get('cases')
        if not isinstance(rows,list) or len(rows)!=45:raise ValueError('split manifest must contain 45 cases')
        self.revision=payload['revision'];self.sha256=hashlib.sha256(raw).hexdigest();self.cases={}
        strata=Counter();pools=Counter();hashes=set()
        for row in rows:
            if not isinstance(row,dict):raise ValueError('invalid split entry')
            case=row.get('case');original=row.get('original_split');pool=row.get('split');data_id=row.get('data_id')
            if case not in SOURCE_IDS or case in self.cases:raise ValueError('unknown or duplicate source case')
            if original!=case.split('_',1)[0] or pool not in {'public','private'}:raise ValueError('invalid original/effective split')
            if not isinstance(data_id,str) or not re.fullmatch('[0-9a-f]{64}',data_id) or data_id in hashes:
                raise ValueError('invalid or duplicate source data fingerprint')
            dataset=row.get('dataset');profile=row.get('profile')
            if dataset not in DATASETS or profile not in PROFILES:raise ValueError('unknown dataset/profile')
            self.cases[case]=dict(row);hashes.add(data_id);pools[pool]+=1;strata[dataset,profile,pool]+=1
        if set(self.cases)!=SOURCE_IDS or pools!=Counter(public=15,private=30):raise ValueError('incomplete source coverage or wrong pool counts')
        for dataset in DATASETS:
            for profile in PROFILES:
                if strata[dataset,profile,'public']!=1 or strata[dataset,profile,'private']!=2:
                    raise ValueError('each dataset/profile must have one public and two private cases')

    def check_source(self,case,data_id,original_split,dataset,profile):
        entry=self.cases.get(case)
        if entry is None:raise ValueError('source case absent from split manifest')
        actual=(data_id,original_split,dataset,profile)
        expected=tuple(entry[k] for k in ('data_id','original_split','dataset','profile'))
        if actual!=expected:raise ValueError('source fingerprint/metadata differs from split manifest: '+case)
        return entry

    def validate_data_root(self,root):
        complete={p.parent.name:p for p in Path(root).glob('*/COMPLETE.json')}
        if set(complete)!=set(self.cases):raise ValueError('data root does not cover the manifest exactly')
        for case,path in complete.items():
            meta=json.loads((path.parent/'meta.json').read_text())
            self.check_source(case,hashlib.sha256(path.read_bytes()).hexdigest(),meta['split'],meta['dataset'],meta['profile'])

    def overlay_result(self,row,*,require_overlay=False):
        provenance=('original_split','split_revision','split_manifest_sha256')
        tagged=any(key in row for key in provenance)
        if require_overlay and not tagged:raise ValueError('old result cannot be reused under a split overlay')
        if tagged and not all(key in row for key in provenance):raise ValueError('incomplete split provenance')
        original=row['original_split'] if tagged else row['split']
        entry=self.check_source(row['case'],row['data_id'],original,row['dataset'],row['profile'])
        if tagged:
            if row['split_revision']!=self.revision or row['split_manifest_sha256']!=self.sha256 or row['split']!=entry['split']:
                raise ValueError('result belongs to a different split overlay')
        elif row['split']!=entry['original_split']:raise ValueError('legacy result has a contradictory original split')
        return dict(row,original_split=entry['original_split'],split=entry['split'],split_revision=self.revision,split_manifest_sha256=self.sha256)
