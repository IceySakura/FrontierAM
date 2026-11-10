function stream_kiops(multiplier)
% Local stream adapter; pinned upstream kiops.m/expm_kiops.m remain unchanged.
% Generic size fallback and tolerance policy are frozen before model runs. The optional argument overrides the
% tolerance multiplier (tol = multiplier * rtol) for public calibration only (calibrate.py).
  here=fileparts(mfilename('fullpath'));
  addpath(fullfile(here,'vendor_kiops'));
  fprintf(1,'READY\n'); fflush(1);
  magic=read_exact(8,'uint8=>uint8');
  assert(isequal(char(magic(:).'),'FAMEXP2\n') || isequal(magic(:),uint8([70;65;77;69;88;80;50;10])));
  len=read_exact(1,'uint32=>double'); assert(len>0 && len<=4096);
  raw=read_exact(len,'uint8=>uint8'); h=jsondecode(char(raw(:).'));
  assert(h.version==2 && h.n>=1 && h.n<=120000 && h.nnz<=1000000);
  assert(h.requests==1 || h.requests==16);
  n=double(h.n); nz=double(h.nnz);
  ptr=read_exact(n+1,'uint64=>double');
  col=read_exact(nz,'uint32=>double');
  val=read_exact(nz,'double=>double');
  assert(ptr(1)==0 && ptr(end)==nz && all(diff(ptr)>=0));
  assert(all(col>=0 & col<n) && all(isfinite(val)));
  rows=repelem((1:n).',diff(ptr));
  L=sparse(rows,col+1,val,n,n); t=double(h.t); A=-t*L;
  assert(t>=0 && t<=10000 && isfinite(t));
  if nargin < 1, multiplier = 0.001; end
  tol=double(h.rtol)*multiplier;
  mmin=min(10,n); mmax=min(128,n+1); m_init=mmin;
  for j=1:double(h.requests)
    b0=read_exact(n,'double=>double'); b1=read_exact(n,'double=>double');
    assert(all(isfinite(b0)) && all(isfinite(b1)));
    if t==0 || nnz(L)==0 || (~any(b0) && ~any(b1))
      y=b0+t*b1;
    elseif n<=64
      aug=[full(A),t*b1;zeros(1,n+1)];
      tmp=expm(aug)*[b0;1]; y=tmp(1:n);
    else
      [y,m_init,~]=kiops(1,A,[b0,t*b1],tol,m_init,mmin,mmax,false);
    end
    assert(numel(y)==n && isreal(y) && all(isfinite(y)));
    written=fwrite(1,y(:),'double',0,'ieee-le');
    assert(written==n); fflush(1);
  end
end

function x=read_exact(count,precision)
  if count==0, x=zeros(0,1); return; end
  [x,received]=fread(0,count,precision,0,'ieee-le');
  if received~=count, error('Short input'); end
end
