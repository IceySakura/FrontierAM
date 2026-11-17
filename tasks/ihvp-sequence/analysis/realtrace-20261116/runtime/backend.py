"""Private host only; operator arrays and references never enter worker memory."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from operators import load_operator,numpy_hvp
class RealCase:
 def __init__(self,folder):
  self.folder=Path(folder);self.meta=json.loads((self.folder/'meta.json').read_text());self.dimension=self.meta['dimension'];self.steps=self.meta['steps'];self.mu=self.meta['mu'];self.L=self.meta['L']
  with np.load(self.folder/'reference.npz',allow_pickle=False) as f:self.ref={k:f[k].copy() for k in f.files}
  self.ops=[load_operator(self.folder/f'step_{t:03d}.npz') for t in range(self.steps)]
 def rhs(self,t):return self.ref['b'][t].copy()
 def phase(self,t):return int(self.ref['phase'][t])
 def hvp(self,t,v):return numpy_hvp(self.ops[t],v)
 def score(self,t,x):
  err=x-self.ref['vref'][t];he=self.hvp(t,err);loss=max(0,float(err@he)/float(self.ref['denom'][t]));res=self.hvp(t,x)-self.ref['b'][t]
  return {'energy_loss':loss,'relative_residual':float(np.linalg.norm(res)/np.linalg.norm(self.ref['b'][t]))}
