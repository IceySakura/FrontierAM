"""Boundary, monotonicity, pool separation and recorded-result checks."""
from pathlib import Path
import copy,json,math
from calibrate_anchors import score_loss,calibrate
ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'reports/results_summary.json').read_text());result=calibrate(s)
for pool,a in result['pool_anchors'].items():
    e0,e50=a['E0'],a['E50']
    assert e0>e50>0
    assert score_loss(e0,e0,e50)==0 and score_loss(e50,e0,e50)==50 and score_loss(0,e0,e50)==100
    assert abs(score_loss((e0+e50)/2,e0,e50)-25)<1e-10
    values=[score_loss(e0*1.1*i/1000,e0,e50) for i in range(1001)]
    assert all(0<=v<=100 for v in values) and all(x>=y for x,y in zip(values,values[1:]))
    assert abs(score_loss(e50*(1-1e-10),e0,e50)-50)<1e-6
    assert abs(score_loss(e50*(1+1e-10),e0,e50)-50)<1e-6
for value in [math.nan,math.inf,-1]:
    try:score_loss(value,.2,.1)
    except ValueError:pass
    else:raise AssertionError('invalid metric accepted')
bad=copy.deepcopy(s);bad['private']['cold_cg']['mean_loss']=bad['private']['warm_cg']['mean_loss']
try:calibrate(bad)
except ValueError:pass
else:raise AssertionError('inverted/equal anchors accepted')
incomplete=copy.deepcopy(s);incomplete['private']['cold_cg']['valid']=29
try:calibrate(incomplete)
except ValueError:pass
else:raise AssertionError('incomplete anchor accepted')
by={(r['pool'],r['method']):r for r in result['scores']}
assert by['private','cold_cg']['score']==50 and by['private','warm_cg']['score']==0
assert by['private','agent_sol_iter']['score']==0
for pool in ['public','private']:
    for method,row in s[pool].items():assert by[pool,method]['mean_loss']==row['mean_loss']
print('PASS: score endpoints, monotonicity, continuity, invalid/missing anchors, separate pools and unchanged raw losses')

# Nonlinear scoring must follow pool aggregation, and mixed/duplicate records must fail.
import tempfile
from calibrate_anchors import collect_results
with tempfile.TemporaryDirectory(prefix='ihvp-score-check-') as temp:
    folder=Path(temp)
    for pool,n in [('public',15),('private',30)]:
        for i in range(n):
            case=f'{pool}_{i:02d}'
            for method,loss in [('warm_cg',.2),('cold_cg',.1),('candidate',0. if i%2==0 else .3)]:
                row={'case':case,'split':pool,'method':method,'data_id':case,'dimension':16384,'steps':50,'completed_steps':50,'valid':True,'wall_seconds':1.,'peak_rss_mib':20.,'mean_energy_loss':loss}
                (folder/f'{case}__{method}.json').write_text(json.dumps(row))
    aggregated=collect_results([folder]);actual=calibrate(aggregated)
    private=next(r for r in actual['scores'] if r['pool']=='private' and r['method']=='candidate')
    assert abs(private['score']-25.)<1e-10
    try:collect_results([folder,folder])
    except ValueError:pass
    else:raise AssertionError('duplicate records accepted')
    path=folder/'private_00__candidate.json';row=json.loads(path.read_text());row['data_id']='different-data-version';path.write_text(json.dumps(row))
    try:collect_results([folder])
    except ValueError:pass
    else:raise AssertionError('mixed data versions accepted')
print('PASS: aggregate-before-scoring, duplicate rejection and data-version checks')
