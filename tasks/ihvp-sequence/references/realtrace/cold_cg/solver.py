"""Cold-CG zero-point calibration reference: local recalibration required; not a universal SOTA claim."""
import time
from _algorithms import make_solver as _make

class Solver:
    def __init__(self,d,seed,options):
        self.inner=_make('cold_cg',d,seed,{'rank':16,'maxiter_factor':50,**options})
    def solve(self,hvp,b,context):
        local=dict(context);now=time.monotonic()
        # This is the reference's policy, not a per-step evaluator restriction.
        local['deadline']=min(context['sequence_deadline'],now+max(0.,context['sequence_deadline']-now)/(context['steps']-context['step']))
        return self.inner.solve(hvp,b,local)

def make_solver(d,seed,options):return Solver(d,seed,options)
