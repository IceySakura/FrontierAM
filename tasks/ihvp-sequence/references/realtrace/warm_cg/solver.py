import time
import numpy as np
class Solver:
 def __init__(self,d,seed,options):self.last=None
 def solve(self,hvp,b,context):
  start=time.monotonic();end=start+max(0,context['sequence_deadline']-start)/(context['steps']-context['step']);x=np.zeros_like(b) if self.last is None or context['phase_changed'] else self.last.copy();n=0
  try:
   r=b-hvp(x) if np.any(x) else b.copy();p=r.copy();rr=float(r@r)
   while time.monotonic()<end-.006 and rr>float(b@b)*1e-16:
    ap=hvp(p);pap=float(p@ap)
    if pap<=0:break
    alpha=rr/pap;x+=alpha*p;r-=alpha*ap;new=float(r@r);p=r+(new/rr)*p;rr=new;n+=1
  except Exception as e:
   if type(e).__name__!='BudgetExceeded':raise
  self.last=x.copy();return x,{'iterations':n}
def make_solver(d,seed,options):return Solver(d,seed,options)
