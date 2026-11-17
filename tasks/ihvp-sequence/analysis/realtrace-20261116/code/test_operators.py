import numpy as np
import torch
import torch.nn.functional as F
from operators import TorchPoint,numpy_hvp,pcg

def test_hessian_mixed_bounds_diagonal_and_reference():
 torch.manual_seed(23);dtype=torch.float64
 x=torch.randn(11,9,dtype=dtype);p=torch.randn(9,3,dtype=dtype);b=torch.randn(7,3,dtype=dtype);c=torch.randn(4,7,dtype=dtype);w=torch.randn(7,9,dtype=dtype);y=torch.arange(11)%4
 a=TorchPoint(p,b,w,c,x,y);v=torch.randn(a.rhs.numel(),dtype=dtype)
 pp=p.clone().requires_grad_();bb=b.clone().requires_grad_();cc=c.clone().requires_grad_()
 f=F.cross_entropy((x@w.T+(x@pp)@bb.T).relu()@cc.T,y)
 gb,gc=torch.autograd.grad(f,(bb,cc),create_graph=True);vb,vc=a.split(v)
 hb,hc,m=torch.autograd.grad((gb*vb).sum()+(gc*vc).sum(),(bb,cc,pp))
 torch.testing.assert_close(a.hvp(v)-a.shift*v,torch.cat((hb.flatten(),hc.flatten())),rtol=1e-11,atol=1e-11)
 torch.testing.assert_close(a.mixed(v),m,rtol=1e-11,atol=1e-11)
 I=torch.eye(len(v),dtype=dtype);H=a.hvp(I)
 torch.testing.assert_close(H,H.T,rtol=1e-12,atol=1e-12)
 eig=torch.linalg.eigvalsh(H);assert eig.min()>=a.mu-1e-10 and eig.max()<=a.upper+1e-10
 torch.testing.assert_close(a.diagonal,H.diag(),rtol=1e-11,atol=1e-11)
 vals=a.arrays();vals['mask']=(vals['z']>0).astype(float);vals['e']=a.e.numpy()
 np.testing.assert_allclose(numpy_hvp(vals,v.numpy()),(H@v).numpy(),rtol=1e-11,atol=1e-11)
 ref,cert=pcg(a,a.rhs,iterations=4096,certificate=1e-18)
 exact=torch.linalg.solve(H,a.rhs);err=ref-exact
 assert float(err@(H@err)/(exact@(H@exact)))<1e-17
 assert cert['energy_error_certificate']<=1e-18
 print('actual Hessian/mixed, SPD bounds, analytic diagonal, NumPy/Torch equality, certified reference PASS')
if __name__=='__main__':test_hessian_mixed_bounds_diagonal_and_reference()
