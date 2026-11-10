"""Implement prepare() and solve(); protocol glue does not choose an algorithm."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sdk.wire import ready,read_header,read_request,write_result
from scipy.sparse import csr_matrix


def prepare(L,t,rtol,atol):
    """Runs after input arrives; its time and retained memory are charged."""
    raise NotImplementedError('Implement input-specific preparation')


def solve(state,b0,b1):
    """Return all n components, meeting the externally checked tolerance."""
    raise NotImplementedError('Implement one request')


def main():
    ready()
    h=read_header()
    L=csr_matrix((h['data'],h['indices'],h['indptr']),shape=(h['n'],h['n']))
    state=prepare(L,h['t'],h['rtol'],h['atol'])
    for _ in range(h['requests']):
        b0,b1=read_request(h)
        write_result(solve(state,b0,b1))

if __name__=='__main__':main()
