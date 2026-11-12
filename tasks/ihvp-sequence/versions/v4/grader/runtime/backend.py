"""Host-only dense-smoke/signed-DCT loader; no cohort plans or private seeds."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.fft import dct,idct
from .api import InvalidPrediction


def _scale(v,s):return v*(s if v.ndim==1 else s[:,None])
def _forward(data,v):
    v=_scale(np.asarray(v)[data['permutation']],data['sign_in'])
    return _scale(idct(v,type=2,norm='ortho',axis=0,workers=1),data['sign_out'])
def _transpose(data,v):
    v=dct(_scale(v,data['sign_out']),type=2,norm='ortho',axis=0,workers=1)
    v=_scale(v,data['sign_in']);out=np.empty_like(v);out[data['permutation']]=v
    return out

class LegacyCase:
    def __init__(self,path):
        path=Path(path);self.smoke=path.suffix=='.json'
        if self.smoke:
            self.data=json.loads(path.read_text())
            if self.data.get('kind')!='dense_smoke':raise ValueError('unsupported JSON case')
            self.meta=self.data
            for key in ('H','b','phase'):self.data[key]=np.asarray(self.data[key])
            self.data['z_star']=np.asarray(self.data.get('z_star',self.data.get('x_ref')))
        else:
            with np.load(path,allow_pickle=False) as archive:self.data={k:archive[k].copy() for k in archive.files}
            self.meta=json.loads(str(self.data['metadata_json'].item()))
            if self.meta.get('schema')!='coupled_quadratic_bilevel_ihvp_signed_dct_v1':
                raise ValueError('unsupported private teacher schema')
            self.data['_lowrank_basis']=_forward(self.data,np.sqrt(self.data['spectrum'])[:,None]*self.data['V'])
        self.dimension=int(self.meta['dimension']);self.steps=int(self.meta['steps'])
        self.mu=float(self.meta['mu']);self.L=float(self.meta['L'])
        if not 0<self.mu<=self.L:raise ValueError('invalid spectral bounds')
        if self.data['b'].shape!=(self.steps,self.dimension) or self.data['z_star'].shape!=(self.steps,self.dimension):
            raise ValueError('invalid private teacher dimensions')
        if not np.isfinite(self.data['b']).all() or not np.isfinite(self.data['z_star']).all():
            raise ValueError('nonfinite private teacher arrays')
    def rhs(self,t):return np.asarray(self.data['b'][t],dtype=float)
    def phase(self,t):return int(self.data['phase'][t])
    def hvp(self,t,v):
        if self.smoke:return self.data['H'][t]@v
        data=self.data;diagonal=data['spectrum']*data['diagonal_weights'][t]
        u=data['_lowrank_basis'];q=_transpose(data,v)
        return _forward(data,_scale(q,diagonal))+u@_scale(u.T@v,data['lowrank_weights'][t])
    def score(self,t,x):
        z=self.data['z_star'][t];b=self.rhs(t)
        with np.errstate(over='ignore',invalid='ignore'):
            denominator=float(z@b);b_norm=float(np.linalg.norm(b))
        if not np.isfinite(denominator) or denominator<=0 or not np.isfinite(b_norm) or b_norm<=0:
            raise ArithmeticError('invalid private reference denominator/RHS')
        error=x-z;numerator=float(error@self.hvp(t,error))
        residual=float(np.linalg.norm(self.hvp(t,x)-b)/b_norm)
        if not np.isfinite(numerator) or not np.isfinite(residual):raise InvalidPrediction('numerical overflow')
        if numerator < -1e-8*max(1.,denominator):raise InvalidPrediction('invalid numerical energy')
        return {'energy_loss':max(0.,numerator)/denominator,'relative_residual':residual}


class Case:
    """Dispatch host-only v1 and hard-v4 teachers without exposing factors."""
    def __new__(cls, path):
        path = Path(path)
        if path.suffix != '.json':
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive['metadata_json'].item()))
            if meta.get('schema') != 'coupled_quadratic_bilevel_ihvp_signed_dct_v1':
                import importlib.util
                model = Path(__file__).resolve().parents[2] / 'curator/model.py'
                spec = importlib.util.spec_from_file_location('ihvp_hard_v4_teacher', model)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.HardCase(path)
        return LegacyCase(path)
