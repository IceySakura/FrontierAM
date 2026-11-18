"""Exact local joint(B,C) ReLU-CE Hessian, shifted to SPD; no dense matrix.

R_h=XP.T diag(mask_h) E. The Hessian is GGN plus symmetric off-diagonal
blocks[R_h,R_h.T]. A=max_h||R_h||_2 bounds its negative part, so H+(A+gamma)I
is >=gamma I. Spectral norm/Jacobian bounds provide a certified global upper.
"""
import math
import numpy as np

def numpy_hvp(a,v):
 v=np.asarray(v,dtype=np.float64)
 if v.ndim==2:return np.column_stack([numpy_hvp(a,v[:,j]) for j in range(v.shape[1])])
 xp,z,m,p,c,e=a['xp'],a['z'],a['mask'],a['prob'],a['c'],a['e'];n=len(xp);hidden,rank=z.shape[1],xp.shape[1];classes=c.shape[0];cut=hidden*rank
 vb=v[:cut].reshape(hidden,rank);vc=v[cut:].reshape(classes,hidden)
 dz=(xp@vb.T)*m;dl=dz@c.T+z@vc.T
 de=p*(dl-(p*dl).sum(axis=1,keepdims=True))/n
 dt=(de@c+e@vc)*m
 return np.concatenate(((dt.T@xp).ravel(),(de.T@z+e.T@dz).ravel()))+float(a['shift'])*v

def load_operator(path):
 with np.load(path,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
 a['mask']=(a['z']>0).astype(np.float64)
 a['e']=a['prob'].copy();a['e'][np.arange(len(a['e'])),a['y'].astype(np.int64)]-=1.;a['e']/=len(a['e'])
 return a

class TorchPoint:
 def __init__(self,p,b,w0,c,x,y,gamma=1e-3,bounds=True):
  import torch
  self.torch=torch;self.p,self.b,self.c=p,b,c;self.x,self.y=x,y;self.n=len(x);self.xp=x@p
  a=x@w0.T+self.xp@b.T;self.z=a.clamp_min(0);self.mask=(a>0).to(a.dtype)
  self.prob=(self.z@c.T).softmax(-1);self.e=self.prob.clone();self.e[torch.arange(len(x),device=x.device),y]-=1;self.e/=len(x)
  self.t=(self.e@c)*self.mask;self.rhs=torch.cat(((self.t.T@self.xp).flatten(),(self.e.T@self.z).flatten()));self.direct=x.T@(self.t@b)
  if not bounds:return
  rho=torch.zeros((),dtype=x.dtype,device=x.device)
  # min(rank,classes) Gram matrices: exact spectral, not Frobenius, bound.
  for k in range(0,b.shape[0],32):
   r=torch.einsum('nr,nh,nc->hrc',self.xp,self.mask[:,k:k+32],self.e)
   gram=r@r.transpose(1,2) if r.shape[1]<=r.shape[2] else r.transpose(1,2)@r
   rho=torch.maximum(rho,torch.linalg.eigvalsh(gram)[:,-1].clamp_min(0).sqrt().max())
  self.rho=float(rho.item())*(1+1e-10)+1e-12;self.mu=float(gamma);self.shift=self.rho+gamma
  def norm2(a):return float(torch.linalg.eigvalsh(a.T@a)[-1].clamp_min(0).item())
  c2=norm2(c.T);xp2=norm2(self.xp);z2=norm2(self.z)
  sm=float((2*self.prob*(1-self.prob)).max().item())
  self.upper=(sm*(c2*xp2+z2)/len(x)+self.rho+self.shift)*(1+1e-10)+1e-12
  # Analytic Jacobi diagonal; the off-diagonal R blocks contribute no diagonal.
  coeff=((self.prob@c.square())-(self.prob@c).square()).clamp_min(0)*self.mask/len(x)
  db=coeff.T@self.xp.square();dc=(self.prob*(1-self.prob)).T@self.z.square()/len(x)
  self.diagonal=torch.cat((db.flatten(),dc.flatten()))+self.shift
 def split(self,v):
  k=self.b.numel();return v[:k].reshape_as(self.b),v[k:].reshape_as(self.c)
 def directional(self,v):
  vb,vc=self.split(v);dz=(self.xp@vb.T)*self.mask;dl=dz@self.c.T+self.z@vc.T
  de=self.prob*(dl-(self.prob*dl).sum(-1,keepdim=True))/self.n
  dt=(de@self.c+self.e@vc)*self.mask
  return vb,vc,dz,de,dt
 def hvp(self,v):
  torch=self.torch
  if v.ndim==2:return torch.stack([self.hvp(v[:,i]) for i in range(v.shape[1])],dim=1)
  vb,vc,dz,de,dt=self.directional(v)
  return torch.cat(((dt.T@self.xp).flatten(),(de.T@self.z+self.e.T@dz).flatten()))+self.shift*v
 def mixed(self,v):
  vb,vc,dz,de,dt=self.directional(v);return self.x.T@(dt@self.b+self.t@vb)
 def arrays(self):
  return {k:getattr(self,k).detach().cpu().numpy() for k in ['xp','z','prob','c','y']}|{'shift':np.asarray(self.shift,dtype=np.float64)}

def pcg(point,b,*,iterations=4096,certificate=1e-18,reference=True):
 """Reference certificate uses true residual and variational energy lower bound."""
 torch=point.torch;x=torch.zeros_like(b);r=b.clone();z=r/point.diagonal;p=z.clone();rz=float((r*z).sum().item());calls=0
 if not float((b*b).sum().item())>0:raise ValueError('zero RHS')
 cert=math.inf;true_r=r
 for i in range(iterations):
  ap=point.hvp(p);calls+=1;pap=float((p*ap).sum().item())
  if pap<=0 or not math.isfinite(pap):raise ArithmeticError('nonpositive certified-SPD curvature')
  alpha=rz/pap;x.add_(p,alpha=alpha);r.add_(ap,alpha=-alpha)
  if reference and ((i+1)%128==0 or i==iterations-1):
   ax=point.hvp(x);calls+=1;true_r=b-ax
   lower=float((2*b*x-x*ax).sum().item());rr=float((true_r*true_r).sum().item())
   cert=rr/(point.mu*lower) if lower>0 else math.inf
   if cert<=certificate:break
   # Residual replacement/restart prevents recursive-residual false convergence.
   r=true_r.clone();z=r/point.diagonal;p=z.clone();rz=float((r*z).sum().item());continue
  newz=r/point.diagonal;newrz=float((r*newz).sum().item())
  if newrz<=0 or not math.isfinite(newrz):
   if reference:
    ax=point.hvp(x);calls+=1;true_r=b-ax;lower=float((2*b*x-x*ax).sum().item());cert=float((true_r*true_r).sum().item())/(point.mu*lower) if lower>0 else math.inf
   break
  p.mul_(newrz/rz).add_(newz);rz=newrz
 if reference and cert>certificate:raise ArithmeticError('reference failed energy certificate '+str(cert))
 return x,{'hvp_calls':calls,'iterations':i+1,'energy_error_certificate':cert if reference else None}
