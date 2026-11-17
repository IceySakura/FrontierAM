"""Official train splits only; verify publisher MD5 before decoding."""
from pathlib import Path
import gzip,hashlib,json,os,pickle,struct,tarfile,urllib.request
import numpy as np
SOURCES={
 'cifar10':('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz','c58f30108f718f92721af3b95e74349a'),
 'cifar100':('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz','eb9058c3a382ffc7106e4002c42a8d85'),
 'fashion_images':('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz','8d4fb7e6c68d591d4c3dfef9ec88bf0d'),
 'fashion_labels':('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz','25c81989df183df01b3e8a0aad5dffbe')}
def digest(p,kind='sha256'):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,kind).hexdigest()
def acquire(key,folder):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);url,md5=SOURCES[key];p=folder/(key+'_'+url.rsplit('/',1)[1])
 if p.exists():
  if digest(p,'md5')!=md5:raise ValueError('existing dataset checksum mismatch: '+str(p))
  return p
 tmp=p.with_suffix(p.suffix+'.part')
 with urllib.request.urlopen(url,timeout=60) as src,tmp.open('wb') as dst:
  while chunk:=src.read(2**20):dst.write(chunk)
 if digest(tmp,'md5')!=md5:raise ValueError('publisher checksum mismatch: '+key)
 tmp.rename(p);return p

def load_train(name,folder):
 paths=[]
 extracted=os.environ.get('IHVP_CIFAR100_TRAIN') if name=='cifar100' else None
 if extracted:
  p=Path(extracted)
  if digest(p,'md5')!='16019d7e3df5f24257cddd939b257f8d':raise ValueError('CIFAR100 train publisher checksum mismatch')
  with p.open('rb') as f:records=[pickle.load(f,encoding='bytes')]
  paths=[p];raw=records[0][b'data'];y=np.asarray(records[0][b'fine_labels'],dtype=np.int64);channels=3;classes=100
 elif name in ('cifar10','cifar100'):
  p=acquire(name,folder);paths=[p]
  with tarfile.open(p) as tar:
   members=([f'cifar-10-batches-py/data_batch_{i}' for i in range(1,6)] if name=='cifar10' else ['cifar-100-python/train'])
   records=[pickle.loads(tar.extractfile(m).read(),encoding='bytes') for m in members]
  raw=np.concatenate([d[b'data'] for d in records]);y=np.concatenate([np.asarray(d[b'labels' if name=='cifar10' else b'fine_labels'],dtype=np.int64) for d in records]);channels=3;classes=10 if name=='cifar10' else 100
 else:
  assert name=='fashion_mnist';a=acquire('fashion_images',folder);b=acquire('fashion_labels',folder);paths=[a,b]
  with gzip.open(a,'rb') as f:
   magic,n,h,w=struct.unpack('>4I',f.read(16));assert (magic,n,h,w)==(2051,60000,28,28);raw=np.frombuffer(f.read(),np.uint8).reshape(n,h*w).copy()
  with gzip.open(b,'rb') as f:
   magic,n=struct.unpack('>2I',f.read(8));assert (magic,n)==(2049,60000);y=np.frombuffer(f.read(),np.uint8).astype(np.int64)
  channels=1;classes=10
 x=raw.reshape(len(raw),channels,-1).astype(np.float32)/255.
 mean=x.mean(axis=(0,2),dtype=np.float64);std=x.std(axis=(0,2),dtype=np.float64)
 x-=mean[None,:,None].astype(np.float32);x/=std[None,:,None].astype(np.float32);x=x.reshape(len(x),-1)
 assert np.isfinite(x).all() and set(np.unique(y))==set(range(classes))
 provenance={'dataset':name,'split':'train','examples':len(y),'classes':classes,'input_dim':x.shape[1],
  'train_channel_mean':mean.tolist(),'train_channel_std':std.tolist(),
  'sources':[{'url':SOURCES[k][0],'publisher_md5':SOURCES[k][1],'sha256':digest(p)} for k,p in zip(([name] if name!='fashion_mnist' else ['fashion_images','fashion_labels']),paths)]}
 if extracted:
  provenance['sources']=[{'url':'https://www.cs.toronto.edu/~kriz/cifar.html','member':'cifar-100-python/train','publisher_md5':'16019d7e3df5f24257cddd939b257f8d','sha256':digest(paths[0])}]
 return x,y,provenance
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset',nargs='+',default=['cifar10','cifar100','fashion_mnist']);a=p.parse_args()
 for name in a.dataset:
  x,y,meta=load_train(name,a.out);(a.out/(name+'_provenance.json')).write_text(json.dumps(meta,indent=2)+'\n');print(name,x.shape,'verified',flush=True)
