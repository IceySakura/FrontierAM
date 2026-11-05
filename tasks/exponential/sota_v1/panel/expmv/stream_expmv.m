function stream_expmv(policy)
% FAMEXP2 stream adapter around the unmodified Al-Mohy--Higham expmv (SIAM J. Sci. Comput. 33(2), 2011).
% y = exp(-tL) b0 + t*phi1(-tL) b1 is the first n entries of expm(t*[[-L, b1]; [0, 0]]) * [b0; 1].
% policy maps each tolerance level to expmv's precision argument ('half', 'single' or 'double'); the frozen policy is
% the default below, selected on public development cases only (see CALIBRATION.md).
  if nargin < 1 || isempty(policy)
    policy = struct('p4', 'single', 'p7', 'double', 'p10', 'double');
  end
  here=fileparts(mfilename('fullpath'));
  addpath(fullfile(here,'vendor_expmv'));
  fprintf(1,'READY\n'); fflush(1);
  magic=read_exact(8,'uint8=>uint8');
  assert(isequal(magic(:),uint8([70;65;77;69;88;80;50;10])));
  len=read_exact(1,'uint32=>double'); assert(len>0 && len<=4096);
  raw=read_exact(len,'uint8=>uint8'); h=jsondecode(char(raw(:).'));
  assert(h.version==2 && h.n>=1 && h.n<=120000 && h.nnz<=1000000);
  n=double(h.n); nz=double(h.nnz);
  ptr=read_exact(n+1,'uint64=>double');
  col=read_exact(nz,'uint32=>double');
  val=read_exact(nz,'double=>double');
  rows=repelem((1:n).',diff(ptr));
  L=sparse(rows,col+1,val,n,n); t=double(h.t);
  switch double(h.rtol)
    case 1e-4, prec=policy.p4;
    case 1e-7, prec=policy.p7;
    otherwise, prec=policy.p10;
  end
  for j=1:double(h.requests)
    b0=read_exact(n,'double=>double'); b1=read_exact(n,'double=>double');
    if t==0 || nnz(L)==0 || (~any(b0) && ~any(b1))
      y=b0+t*b1;
    else
      aug=[-L, sparse(b1); sparse(1,n+1)];
      tmp=expmv(t,aug,[b0;1],[],prec);
      y=tmp(1:n);
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
