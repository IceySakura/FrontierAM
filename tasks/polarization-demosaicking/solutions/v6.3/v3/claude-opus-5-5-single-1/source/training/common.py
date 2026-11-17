import numpy as np, time
ROOT='/workspace/public-dev/wen5'
SCENES=['43','55','72','91','99']
_cache={}
def load(s):
    if s not in _cache:
        _cache[s]=(np.load(f'{ROOT}/{s}/raw.npy'), np.load(f'{ROOT}/{s}/reference.npy'))
    return _cache[s]
def score(pred, ref):
    d=pred.astype(np.float64)-ref.astype(np.float64)
    ch=np.sqrt(np.mean(d*d,axis=(1,2))); return ch, ch.mean()
def evaluate(fn, scenes=SCENES, verbose=True):
    errs=[]
    for s in scenes:
        raw,ref=load(s); t=time.perf_counter(); out=fn(raw); dt=time.perf_counter()-t
        ch,m=score(out,ref); errs.append(m)
        if verbose: print(f'  {s}: {m:.4f} ch={np.round(ch,3)} t={dt*1000:.0f}ms')
    print(f'MEAN {np.mean(errs):.5f}'); return np.mean(errs)
# channel order 0,45,90,135 ; phase (r%2,c%2): (0,0)=90,(0,1)=45,(1,0)=135,(1,1)=0
NATIVE={0:(1,1),1:(0,1),2:(0,0),3:(1,0)}
