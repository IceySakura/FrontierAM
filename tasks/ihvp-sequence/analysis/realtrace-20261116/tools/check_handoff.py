"""Check archive boundaries, result consistency and the public worker protocol."""
from pathlib import Path
import importlib.util,json,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runtime.api import run_case

model_names={'README.md','statement.md','interface.md','solver.py','smoke.json'}
with zipfile.ZipFile(ROOT/'dist/MODEL_ONLY_ihvp_realtrace_20261116.zip') as z:
    assert set(z.namelist())==model_names
    for name in model_names:assert z.read(name)==(ROOT/'model_materials'/name).read_bytes()
with zipfile.ZipFile(ROOT/'dist/EVALUATOR_ONLY_ihvp_realtrace_tools_20261116.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert Path(name).suffix not in {'.npz','.npy','.so','.pth','.bin'}
        assert Path(name).name not in {'plans.json','reference.npz'}
        assert not any(p in {'var','data','audit'} for p in Path(name).parts)
summary=json.loads((ROOT/'reports/results_summary.json').read_text())
assert all(a['n']==45 and a['valid']==45 for a in summary['combined'].values())
assert summary['quality']['systems']==2250 and summary['quality']['all_passed']
assert json.loads((ROOT/'reports/reference_eligibility.json').read_text())['valid_improved_50_anchor_found'] is False
spec=json.loads((ROOT/'model_materials/smoke.json').read_text())
class Case:
    dimension=spec['dimension'];steps=spec['steps'];mu=spec['mu'];L=spec['L']
    def rhs(self,t):return np.asarray(spec['systems'][t]['b'],dtype=float)
    def phase(self,t):return spec['systems'][t]['phase']
    def hvp(self,t,v):return np.asarray(spec['systems'][t]['H'],dtype=float)@v
    def score(self,t,x):
        h=np.asarray(spec['systems'][t]['H'],dtype=float);b=self.rhs(t);truth=np.linalg.solve(h,b);e=x-truth
        return {'energy_loss':float(e@h@e/(truth@h@truth)),'relative_residual':float(np.linalg.norm(h@x-b)/np.linalg.norm(b))}
case=Case()
for item in spec['systems']:
    vals=np.linalg.eigvalsh(item['H']);assert vals.min()>=case.mu and vals.max()<=case.L
for directory,zero in [(ROOT/'model_materials',True),(ROOT/'submissions/cold_cg',False)]:
    (ROOT/'var').mkdir(exist_ok=True)
    result=run_case(directory,case,wall_seconds=30,memory_mib=256,log_path=ROOT/'var'/('smoke_'+directory.name+'.stderr'))
    assert result['valid'],result
    if zero:assert abs(result['mean_energy_loss']-1)<1e-12
    else:assert result['mean_energy_loss']<1e-14
print('PASS: model allowlist, no private arrays/seeds, report counts, public spectrum bounds and zero/CG worker smoke')
