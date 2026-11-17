"""Curator only. Freeze plans before running; candidates never control this driver."""
from pathlib import Path
import argparse,hashlib,json,math,os,secrets,time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_train
from operators import TorchPoint,pcg
ROOT=Path(__file__).resolve().parents[1]
PROFILES=[
 {'name':'continuous_fixed','calls_per_episode':50,'cadence':10,'fixed_batches':True},
 {'name':'continuous_resampled','calls_per_episode':50,'cadence':10,'fixed_batches':False},
 {'name':'medium_resets','calls_per_episode':10,'cadence':10,'fixed_batches':False},
 {'name':'short_resets','calls_per_episode':4,'cadence':10,'fixed_batches':False},
 {'name':'rapid_resampled','calls_per_episode':50,'cadence':5,'fixed_batches':False}]
DIMS={'cifar10':(256,54,10),'cifar100':(128,28,100),'fashion_mnist':(256,54,10)}

def savejson(p,a):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(a,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def make_plans(path):
 if path.exists():raise FileExistsError(path)
 rows=[];used=set()
 for dataset in DIMS:
  for profile in PROFILES:
   for split,count in [('public',1),('private',2)]:
    for rep in range(count):
     seed=secrets.randbits(32)
     while seed in used:seed=secrets.randbits(32)
     used.add(seed);idx=sum(r['split']==split for r in rows)
     rows.append({'id':f'{split}_{idx:02d}','dataset':dataset,'profile':profile,'split':split,'seed':seed,'steps':50,'batch':4096,'query_batch':128,'gamma':.001,'driver_iterations':32,'p_lr':1.0,'burnin_episodes':secrets.choice([0,1,3])})
 path.parent.mkdir(parents=True,exist_ok=True);savejson(path,{'version':'hero-realtrace-candidate-v1','cases':rows});path.chmod(0o600)

def generate(cfg,out,downloads,device='cuda:0',smoke=False):
 start=time.time();out=Path(out)
 if (out/'COMPLETE.json').exists():return
 if out.exists():raise FileExistsError('partial output preserved; use a fresh attempt: '+str(out))
 out.mkdir(parents=True,mode=0o700);savejson(out/'config.json',cfg)
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rng=np.random.default_rng(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']) if device.startswith('cuda') else None
 x,y,source=load_train(cfg['dataset'],downloads);hidden,rank,classes=DIMS[cfg['dataset']];D=x.shape[1]
 if smoke:hidden,rank=8,3
 # Same bias-free two-layer ReLU parametrization, complete joint (B,C) space.
 w0=torch.empty(hidden,D,device=device);torch.nn.init.kaiming_uniform_(w0,a=math.sqrt(5));w0=w0.detach()
 c=torch.empty(classes,hidden,device=device);torch.nn.init.xavier_uniform_(c);c.requires_grad_()
 def batch(n):
  ix=rng.choice(len(y),min(n,len(y)),replace=False);return torch.from_numpy(x[ix]).to(device),torch.from_numpy(y[ix]).to(device)
 def orth(a):return torch.linalg.qr(a,mode='reduced').Q
 def loss(p,b,xx,yy):return F.cross_entropy((xx@w0.T+(xx@p)@b.T).relu()@c.T,yy)
 def lower(p,b,opt,xy):
  opt.zero_grad(set_to_none=True);val=loss(p,b,*xy);val.backward();torch.nn.utils.clip_grad_norm_([b,c],1.);opt.step();return float(val.item())
 bvec=[];solutions=[];phases=[];uppers=[];certs=[];denoms=[];diagnostics=[];t=0;episode=-cfg.get('burnin_episodes',0);prof=cfg['profile'];T=3 if smoke else cfg['steps'];per=2 if smoke else prof['calls_per_episode'];bs=32 if smoke else cfg['batch']
 while t<T:
  p=orth(torch.randn(D,rank,device=device));b=torch.zeros(hidden,rank,device=device,requires_grad=True)
  opt=torch.optim.SGD([b,c],lr=.016,momentum=.9,nesterov=True)
  fixed=batch(bs);query=batch(min(cfg['query_batch'],bs))
  for k in range(per*prof['cadence']+prof['cadence']):
   last_loss=lower(p,b,opt,fixed if prof['fixed_batches'] else batch(bs))
   if (k+1)%prof['cadence'] or k+1>per*prof['cadence']:continue
   if not prof['fixed_batches']:fixed=batch(bs);query=batch(min(cfg['query_batch'],bs))
   with torch.no_grad():
    support=TorchPoint(p.double(),b.double(),w0.double(),c.double(),fixed[0].double(),fixed[1],gamma=cfg['gamma'])
    upper=TorchPoint(p.double(),b.double(),w0.double(),c.double(),query[0].double(),query[1],bounds=False)
    if episode>=0:
     reference,cert=pcg(support,upper.rhs,certificate=1e-18)
     arrays=support.arrays();np.savez(out/f'step_{t:03d}.npz',**arrays)
     rr=upper.rhs.detach().cpu().numpy();vv=reference.detach().cpu().numpy()
     bvec.append(rr);solutions.append(vv);phases.append(episode);uppers.append(support.upper);certs.append(cert);denoms.append(float((reference*support.hvp(reference)).sum().item()))
     diagnostics.append({'step':t,'phase':episode,'episode_lower_step':k+1,'training_loss':last_loss,'rho':support.rho,'shift':support.shift,'upper_bound':support.upper,'reference_iterations':cert['iterations'],'reference_certificate':cert['energy_error_certificate']})
     t+=1;savejson(out/'progress.json',{'completed':t,'steps':T,'elapsed_seconds':time.time()-start})
    approx,_=pcg(support,upper.rhs,iterations=cfg['driver_iterations'],reference=False)
    gradient=upper.direct-support.mixed(approx);gradient-=p.double()@(p.double().T@gradient)
    gn=float(gradient.norm().item());accepted=False;old=float(loss(p,b,*query).item())
    if gn>0 and math.isfinite(gn):
     direction=(gradient/gn).float();lr=cfg['p_lr']
     for trial in range(4):
      uu,ss,vh=torch.linalg.svd(p-lr*direction,full_matrices=False);proposal=uu@vh;new=float(loss(proposal,b,*query).item())
      if math.isfinite(new) and new<=old:p=proposal;accepted=True;break
      lr*=.5
    if episode>=0:diagnostics[-1].update(p_update_accepted=accepted,p_gradient_norm=gn,query_loss_before=old)
   if t>=T:break
  with torch.no_grad():w0.add_(b.detach()@p.T)
  episode+=1
 refs={'b':np.stack(bvec),'vref':np.stack(solutions),'phase':np.array(phases),'denom':np.array(denoms),'upper':np.array(uppers)}
 np.savez(out/'reference.npz',**refs)
 meta={'id':cfg['id'],'split':cfg['split'],'dataset':cfg['dataset'],'profile':prof['name'],'dimension':len(bvec[0]),'steps':T,'mu':cfg['gamma'],'L':max(uppers),'source':source,'diagnostics':diagnostics,'generation_seconds':time.time()-start,'precision':'float32 training; float64 frozen true Hessian plus certified SPD shift','operator':'joint B,C CE Hessian + (max_h ||R_h||_2 + gamma) I','driver':'fixed 32-iteration Jacobi PCG; independent of submitted solvers','episode_budget':per*prof['cadence']+prof['cadence']}
 savejson(out/'meta.json',meta)
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.npz')}
 savejson(out/'COMPLETE.json',{'files':hashes,'elapsed_seconds':time.time()-start})
 print(json.dumps({'case':cfg['id'],'status':'complete','systems':T,'seconds':time.time()-start}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,default=ROOT/'var/plans.json');a.add_argument('--make-plans',action='store_true');a.add_argument('--case');a.add_argument('--out',type=Path);a.add_argument('--downloads',type=Path,default=ROOT/'var/downloads');a.add_argument('--device',default='cuda:0');a.add_argument('--smoke',action='store_true');args=a.parse_args()
 if args.make_plans:make_plans(args.plan)
 else:
  plans=json.loads(args.plan.read_text());cfg=next(r for r in plans['cases'] if r['id']==args.case);generate(cfg,args.out,args.downloads,args.device,args.smoke)
