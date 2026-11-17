"""Check source-preserving split overlays and pool-level result aggregation."""
from pathlib import Path
import copy,hashlib,json,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from split_manifest import SplitManifest,REVISION,CALIBRATION_REVISION
from calibrate_anchors import collect_results,calibrate

def rejected(call):
    try:call()
    except ValueError:return
    raise AssertionError('invalid overlay accepted')

with tempfile.TemporaryDirectory(prefix='ihvp-split-check-') as temp:
    root=Path(temp);data=root/'data';results=root/'results';results.mkdir();entries=[]
    for i in range(15):
        dataset=['cifar10','cifar100','fashion_mnist'][i//5]
        profile=['continuous_fixed','continuous_resampled','medium_resets','short_resets','rapid_resampled'][i%5]
        for case,pool in [(f'public_{i:02d}','private'),(f'private_{2*i:02d}','public'),(f'private_{2*i+1:02d}','private')]:
            original=case.split('_',1)[0];folder=data/case;folder.mkdir(parents=True)
            complete=json.dumps({'source':case}).encode();(folder/'COMPLETE.json').write_bytes(complete)
            (folder/'meta.json').write_text(json.dumps({'split':original,'dataset':dataset,'profile':profile}))
            entry={'case':case,'data_id':hashlib.sha256(complete).hexdigest(),'original_split':original,'split':pool,'dataset':dataset,'profile':profile};entries.append(entry)
            for method,loss in [('warm_cg',.2),('cold_cg',.1),('candidate',.3 if original=='public' else .05)]:
                row={'case':case,'split':original,'method':method,'data_id':entry['data_id'],'dataset':dataset,'profile':profile,'dimension':16384,'steps':50,'completed_steps':50,'valid':True,'wall_seconds':1.,'peak_rss_mib':20.,'mean_energy_loss':loss}
                (results/f'{case}__{method}.json').write_text(json.dumps(row))
    path=root/'split.json';payload={'revision':REVISION,'cases':entries}
    path.write_text(json.dumps(payload));manifest=SplitManifest(path);manifest.validate_data_root(data)
    summary=collect_results([results],manifest);scored=calibrate(summary,CALIBRATION_REVISION)
    by={(row['pool'],row['method']):row for row in scored['scores']}
    assert scored['revision']==CALIBRATION_REVISION
    assert all(case.startswith('private_') for case in summary['public']['cold_cg']['case_ids'])
    assert abs(by['public','candidate']['score']-75.)<1e-10
    assert abs(by['private','candidate']['score']-12.5)<1e-10
    out=root/'calibration';out.mkdir();historical=out/'anchor_calibration_cg50-warm0-20261117.json';historical.write_text('historical result')
    subprocess.run([sys.executable,str(ROOT/'tools/calibrate_anchors.py'),'--results',str(results),'--split-manifest',str(path),'--out',str(out)],check=True,capture_output=True,text=True)
    saved=json.loads((out/f'anchor_calibration_{CALIBRATION_REVISION}.json').read_text())
    assert saved['revision']==CALIBRATION_REVISION and saved['split_manifest_sha256']==manifest.sha256
    assert historical.read_text()=='historical result'
    rejected(lambda:collect_results([results,results],manifest))
    rawpath=results/'public_00__candidate.json';raw=json.loads(rawpath.read_text())
    overlaid=manifest.overlay_result(raw)
    assert raw['split']=='public' and overlaid['split']=='private' and overlaid['case']==raw['case']
    assert manifest.overlay_result(overlaid,require_overlay=True)==overlaid
    rejected(lambda:manifest.overlay_result(raw,require_overlay=True))
    bad=copy.deepcopy(raw);bad['split']='private';rejected(lambda:manifest.overlay_result(bad))
    bad=copy.deepcopy(overlaid);bad['split_manifest_sha256']='0'*64;rejected(lambda:manifest.overlay_result(bad))
    bad=copy.deepcopy(overlaid);bad.pop('original_split');rejected(lambda:manifest.overlay_result(bad))
    mixed=copy.deepcopy(raw);mixed['data_id']='0'*64;rawpath.write_text(json.dumps(mixed))
    rejected(lambda:collect_results([results],manifest));rawpath.write_text(json.dumps(raw))
    # A duplicate source cannot be admitted under a different pool even without an overlay.
    extra=root/'extra';extra.mkdir();duplicate=copy.deepcopy(raw);duplicate['split']='private'
    (extra/rawpath.name).write_text(json.dumps(duplicate))
    rejected(lambda:collect_results([results,extra]))
    changed=data/'public_00'/'COMPLETE.json';old=changed.read_bytes();changed.write_bytes(b'different source')
    rejected(lambda:manifest.validate_data_root(data));changed.write_bytes(old)
    for alter in ['missing','duplicate','fingerprint','stratum']:
        bad=copy.deepcopy(payload)
        if alter=='missing':bad['cases'].pop()
        elif alter=='duplicate':bad['cases'][-1]=bad['cases'][0]
        elif alter=='fingerprint':bad['cases'][-1]['data_id']=bad['cases'][0]['data_id']
        else:
            bad['cases'][1]['split']='private';bad['cases'][3]['split']='public'
        path.write_text(json.dumps(bad));rejected(lambda:SplitManifest(path))
print('PASS: balanced coverage, source fingerprints, reverse source pools, immutable originals, overlay provenance, cross-pool duplicates and aggregate-before-score')
