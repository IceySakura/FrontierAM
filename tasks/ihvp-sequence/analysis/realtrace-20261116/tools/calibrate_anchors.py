"""Warm CG=0 / Cold CG=50, pool-level post-hoc calibration of frozen losses."""
from pathlib import Path
import argparse,csv,hashlib,json,math,re
ROOT=Path(__file__).resolve().parents[1]
REVISION='cg50-warm0-20261117'
ZERO='warm_cg'
FIFTY='cold_cg'
COUNTS={'public':15,'private':30}

def score_loss(loss,e0,e50):
    if not all(math.isfinite(x) for x in (loss,e0,e50)) or loss<0 or not e0>e50>0:
        raise ValueError('need finite nonnegative loss and E0 > E50 > 0')
    if loss>=e0:return 0.
    if loss>=e50:return 50.*(e0-loss)/(e0-e50)
    return 100.-50.*loss/e50

def collect_results(roots):
    rows={};data_ids={};groups={}
    for root in roots:
        for path in sorted(root.rglob('*.json')):
            if not re.fullmatch(r'(public|private)_\d+__.+\.json',path.name):continue
            row=json.loads(path.read_text());key=(row['split'],row['case'],row['method'])
            if key in rows:raise ValueError('duplicate result: '+str(key))
            if row['dimension']!=16384 or row['steps']!=50:raise ValueError('wrong case scale: '+str(path))
            case=(row['split'],row['case'])
            if case in data_ids and data_ids[case]!=row['data_id']:raise ValueError('mixed data versions')
            data_ids[case]=row['data_id'];rows[key]=row
            groups.setdefault((row['split'],row['method']),[]).append(row)
    summary={pool:{} for pool in COUNTS}
    for (pool,method),records in groups.items():
        if pool not in summary:raise ValueError('unknown pool')
        good=[r for r in records if r['valid']]
        for r in good:
            if r['completed_steps']!=50 or r['wall_seconds']>30 or r['peak_rss_mib']>256:
                raise ValueError('valid flag contradicts resource/output record')
        summary[pool][method]={'n':len(records),'valid':len(good),'mean_loss':sum(r['mean_energy_loss'] for r in good)/len(good) if good else None,'case_ids':sorted(r['case'] for r in records)}
    return summary

def calibrate(summary):
    anchors={};scores=[]
    for pool,count in COUNTS.items():
        methods=summary.get(pool,{})
        for method in (ZERO,FIFTY):
            row=methods.get(method,{})
            if row.get('n')!=count or row.get('valid')!=count:
                raise ValueError(f'{pool}: both anchors must have all {count} valid cases')
        e0=methods[ZERO]['mean_loss'];e50=methods[FIFTY]['mean_loss']
        score_loss(0.,e0,e50)
        reference_ids=methods[ZERO].get('case_ids')
        if reference_ids is not None and reference_ids!=methods[FIFTY].get('case_ids'):
            raise ValueError('anchors cover different cases')
        anchors[pool]={'cases':count,'zero_method':ZERO,'E0':e0,'fifty_method':FIFTY,'E50':e50}
        for method,row in methods.items():
            loss=row.get('mean_loss');value=None
            if row['n']!=count:status='unmeasured'
            elif reference_ids is not None and row.get('case_ids')!=reference_ids:raise ValueError('candidate covers different cases')
            elif row['valid']!=count:status='invalid_submission';value=0.
            elif loss is None:status='unmeasured'
            else:status='scored';value=score_loss(loss,e0,e50)
            scores.append({'pool':pool,'method':method,'mean_loss':loss,'score':value,'status':status,'cases':row['n'],'valid':row['valid'],'zero_method':ZERO,'E0':e0,'fifty_method':FIFTY,'E50':e50})
    return {'revision':REVISION,'selection_basis':'user-requested post-hoc choice from already-observed private mean losses','ranking_pool':'private','post_hoc':True,'unseen_holdout_claim':False,'scope':'replay calibration of 2026-11-16 candidate data; historical formal task scores unchanged','zero_method':ZERO,'fifty_method':FIFTY,'formula':'0 if E>=E0; 50*(E0-E)/(E0-E50) if E50<=E<E0; 100-50*E/E50 if 0<=E<E50','pool_anchors':anchors,'scores':scores}

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path);p.add_argument('--results',nargs='+',type=Path);p.add_argument('--out',type=Path,default=ROOT/'reports');a=p.parse_args()
    if a.summary and a.results:p.error('choose --summary or --results')
    source=a.summary or ROOT/'reports/results_summary.json'
    if a.results:
        summary=collect_results(a.results);fingerprint=json.dumps(summary,sort_keys=True).encode();origin='aggregated frozen per-case records'
    else:
        fingerprint=source.read_bytes();summary=json.loads(fingerprint);origin=str(source.name)
    result=calibrate(summary);result['source_sha256']=hashlib.sha256(fingerprint).hexdigest();result['source']=origin
    if not a.results:result['data_archive_sha256']=json.loads((ROOT/'reports/REMOTE_DATA_PACKAGE.json').read_text())['sha256']
    a.out.mkdir(parents=True,exist_ok=True);dest=a.out/f'anchor_calibration_{REVISION}.json';payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if dest.exists() and dest.read_text()!=payload:raise ValueError('calibration output differs; use a new output directory/version')
    dest.write_text(payload)
    with (a.out/f'scores_{REVISION}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result['scores'][0]),lineterminator='\n');w.writeheader();w.writerows(result['scores'])
    print(json.dumps({'revision':REVISION,'pool_anchors':result['pool_anchors'],'private_scores':{r['method']:r['score'] for r in result['scores'] if r['pool']=='private'}},indent=2))
if __name__=='__main__':main()
