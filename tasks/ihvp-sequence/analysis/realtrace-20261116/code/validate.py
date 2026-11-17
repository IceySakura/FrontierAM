"""Curator QC, independent CPU residual certification, temporal diagnostics."""
from pathlib import Path
import argparse,json,sys,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime.backend import RealCase
from utils import savejson

def validate(folder,out):
 manifest=json.loads((Path(folder)/'COMPLETE.json').read_text())
 for name,expected in manifest['files'].items():
  with (Path(folder)/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
 case=RealCase(folder);rng=np.random.default_rng(88412);rows=[];previous=None;probe=rng.standard_normal(case.dimension)
 for t,a in enumerate(case.ops):
  b=case.ref['b'][t];v=case.ref['vref'][t];u=rng.standard_normal(case.dimension);w=rng.standard_normal(case.dimension)
  hu=case.hvp(t,u);hw=case.hvp(t,w);hv=case.hvp(t,v);r=b-hv
  lower=float(2*b@v-v@hv);cert=float(r@r)/(case.mu*lower) if lower>0 else float('inf')
  symmetry=abs(float(u@hw-w@hu))/max(1,float(np.linalg.norm(u)*np.linalg.norm(hw)),float(np.linalg.norm(w)*np.linalg.norm(hu)))
  linear=np.linalg.norm(case.hvp(t,.31*u-.7*w)-(.31*hu-.7*hw))/max(np.linalg.norm(hu)+np.linalg.norm(hw),1e-200)
  assert np.isfinite(b).all() and np.linalg.norm(b)>0 and np.isfinite(v).all()
  assert cert<=1e-16 and symmetry<=1e-11 and linear<=1e-11
  assert float(u@hu)>=case.mu*float(u@u)*(1-1e-10) and float(u@hu)<=case.L*float(u@u)*(1+1e-10)
  same=t>0 and case.phase(t)==case.phase(t-1)
  hp=case.hvp(t,probe);drift=float(np.linalg.norm(hp-previous)/np.linalg.norm(previous)) if same else None;previous=hp
  history=case.ref['vref'][max(0,t-16):t]
  if same:
   history=history[case.ref['phase'][max(0,t-16):t]==case.phase(t)]
  else:history=history[:0]
  projection=None
  if len(history):
   q,_=np.linalg.qr(history.T,mode='reduced');projection=float(np.linalg.norm(v-q@(q.T@v))/np.linalg.norm(v))
  rows.append({'step':t,'reference_certificate_cpu':cert,'symmetry_error':symmetry,'linearity_error':float(linear),'operator_relative_probe_drift':drift,'rhs_relative_change':float(np.linalg.norm(b-case.ref['b'][t-1])/np.linalg.norm(case.ref['b'][t-1])) if same else None,'rhs_cosine_previous':float(b@case.ref['b'][t-1]/np.linalg.norm(b)/np.linalg.norm(case.ref['b'][t-1])) if same else None,'history16_euclidean_projection_error':projection,'relative_residual':float(np.linalg.norm(r)/np.linalg.norm(b))})
 assert case.dimension==16384 or case.steps==3
 assert case.steps==50 or case.steps==3
 sv=np.linalg.svd(case.ref['vref'].T,compute_uv=False);energy=np.cumsum(sv**2)/np.sum(sv**2)
 result={'case':Path(folder).name,'dataset':case.meta['dataset'],'profile':case.meta['profile'],'dimension':case.dimension,'steps':case.steps,'certified_mu':case.mu,'certified_L':case.L,'condition_bound':case.L/case.mu,'max_reference_certificate_cpu':max(r['reference_certificate_cpu'] for r in rows),'max_symmetry_error':max(r['symmetry_error'] for r in rows),'max_linearity_error':max(r['linearity_error'] for r in rows),'solution_rank_99pct':int(np.searchsorted(energy,.99)+1),'solution_rank_999pct':int(np.searchsorted(energy,.999)+1),'rows':rows,'passed':True}
 out.mkdir(parents=True,exist_ok=True);savejson(out/(Path(folder).name+'.json'),result);print(json.dumps({k:result[k] for k in ['case','passed','max_reference_certificate_cpu','solution_rank_99pct']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'var/data');p.add_argument('--out',type=Path,default=ROOT/'var/qc');p.add_argument('--case');a=p.parse_args()
 for f in sorted(a.data.glob('*/COMPLETE.json')):
  if not a.case or f.parent.name==a.case:validate(f.parent,a.out)
