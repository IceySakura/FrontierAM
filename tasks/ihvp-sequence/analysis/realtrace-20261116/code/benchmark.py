"""Serial, immutable-source comparison; no selection/tuning from private results."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from runtime.backend import RealCase
from runtime.api import run_case
from utils import savejson
from split_manifest import SplitManifest
METHODS=[('cold_cg','cold_cg',{}),('warm_cg','warm_cg',{}),('lbfgs_pcg','lbfgs_pcg',{}),('ritz_cg','literature',{'method':'ritz','rank':8,'window':64}),('pod_cg','literature',{'method':'pod','rank':8,'window':32}),('scaled_pcg','literature',{'method':'scaled','rank':4,'window':32,'which':'small'}),('agent_sol_single','agent_sol_single',{}),('agent_sol_iter','agent_sol_iter',{}),('agent_opus_single','agent_opus_single',{}),('agent_opus_iter','agent_opus_iter',{})]
def treehash(folder):
 h=hashlib.sha256()
 for p in sorted(Path(folder).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:
   h.update(str(p.relative_to(folder)).encode());h.update(p.read_bytes())
 return h.hexdigest()
def main(a):
 manifest=SplitManifest(a.split_manifest) if getattr(a,'split_manifest',None) else None
 if manifest:manifest.validate_data_root(a.data)
 os.environ['IHVP_WORKER_UID']='65534' if os.geteuid()==0 else ''
 allowed=os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else None
 if allowed:os.sched_setaffinity(0,{min(allowed)})
 a.out.mkdir(parents=True,exist_ok=True);identity={'methods':METHODS,'submissions_sha256':treehash(ROOT/'submissions'),'runtime_sha256':treehash(ROOT/'runtime'),'wall':a.wall,'memory':256,'cpu_affinity':sorted(os.sched_getaffinity(0)) if allowed else None,'versions':{'numpy':np.__version__,'python':sys.version},'source_freeze':json.loads((ROOT/'var/freeze.json').read_text())}
 if manifest:identity.update(split_revision=manifest.revision,split_manifest_sha256=manifest.sha256)
 identity=json.loads(json.dumps(identity));ip=a.out/'identity.json'
 if ip.exists() and json.loads(ip.read_text())!=identity:raise RuntimeError('identity mismatch; use a new results directory')
 savejson(ip,identity)
 cases=sorted(a.data.glob('*/COMPLETE.json'));methods=[m for m in METHODS if not a.methods or m[0] in a.methods]
 for i,cp in enumerate(cases):
  folder=cp.parent
  pool=manifest.cases[folder.name]['split'] if manifest else folder.name.split('_',1)[0]
  if a.split and (pool!=a.split if manifest else not folder.name.startswith(a.split)):continue
  if a.case and folder.name!=a.case:continue
  case=RealCase(folder);data_id=hashlib.sha256(cp.read_bytes()).hexdigest()
  for method,sub,options in methods[i%len(methods):]+methods[:i%len(methods)]:
   p=a.out/f'{folder.name}__{method}.json'
   if p.exists():
    recorded=json.loads(p.read_text())
    if recorded['data_id']!=data_id:raise RuntimeError('data fingerprint mismatch')
    if manifest:
     if recorded['case']!=folder.name or recorded['method']!=method:raise RuntimeError('result identity mismatch')
     manifest.overlay_result(recorded,require_overlay=True)
    continue
   result=run_case(ROOT/'submissions'/sub,case,wall_seconds=a.wall,memory_mib=256,options=options,log_path=a.out/f'{folder.name}__{method}.stderr')
   predictions=result.pop('prediction_array');np.save(a.out/f'{folder.name}__{method}.pred.npy',predictions)
   result.update(method=method,case=folder.name,split=case.meta['split'],dataset=case.meta['dataset'],profile=case.meta['profile'],data_id=data_id)
   if manifest:result=manifest.overlay_result(result)
   savejson(p,result);print(json.dumps({k:result[k] for k in ['case','method','valid','mean_energy_loss','wall_seconds','peak_rss_mib','termination_reason']}),flush=True)
  del case
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'var/data');p.add_argument('--out',type=Path,default=ROOT/'var/results');p.add_argument('--split');p.add_argument('--split-manifest',type=Path);p.add_argument('--case');p.add_argument('--methods',nargs='+');p.add_argument('--wall',type=float,default=30.);main(p.parse_args())
