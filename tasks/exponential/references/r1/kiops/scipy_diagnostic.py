"""Standard-library diagnostic comparator, not a claim of human SOTA.

The SDK file is copied into this independent frozen reference submission.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from wire import ready, read_header, read_request, write_result

ready()
h = read_header()
n,t = h['n'],h['t']
L = sparse.csr_matrix((h['data'],h['indices'],h['indptr']),shape=(n,n))
A = -t*L
trace = float(A.diagonal().sum())
for _ in range(h['requests']):
    b0,b1 = read_request(h)
    if t==0 or L.nnz==0:
        y = b0+t*b1
    else:
        M = sparse.bmat([[A,sparse.csr_matrix((t*b1)[:,None])],[None,sparse.csr_matrix((1,1))]],format='csr')
        y = expm_multiply(M,np.r_[b0,1.],traceA=trace)[:n]
    write_result(y)
