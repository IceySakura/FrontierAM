import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy import sparse
from problem import Problem, assess
from cones import PSD, exp_barrier, exp_hsolve
from native_ldl import LDL


def write_input(path, p):
    arrays = {"q":p.q,"b":p.b}
    for key, matrix in (("P",p.P),("A",p.A)):
        arrays.update({key+"_data":matrix.data,key+"_indices":matrix.indices,
                       key+"_indptr":matrix.indptr,key+"_shape":np.array(matrix.shape)})
    arrays["metadata_json"] = np.array(json.dumps({
        "schema_version":1,"name":"self-test","cones":p.cones,
        "objective_offset":3.,"metadata":{}}))
    np.savez(path,**arrays)


def cases():
    yield Problem(sparse.csc_matrix([[2.,1.],[1.,2.]]),np.array([-1.,0.]),
                  sparse.csc_matrix([[0.,1.],[-1.,0.]]),np.array([0.,-1.]),
                  [{"type":"zero","dim":1},{"type":"nonnegative","dim":1}]),"solved"
    for kind, d, A, b in [
        ("soc",2,[[-1.],[0.]],[0.,1.]),
        ("exponential",3,[[0.],[0.],[-1.]],[0.,1.,0.]),
        ("psd",2,[[-1.],[0.],[-1.]],[0.,np.sqrt(2.),0.])]:
        yield Problem(sparse.csc_matrix((1,1)),np.ones(1),sparse.csc_matrix(A),
                      np.asarray(b),[{"type":kind,"dim":d}]),"solved"
    yield Problem(sparse.csc_matrix((1,1)),np.zeros(1),
                  sparse.csc_matrix([[-1.],[1.]]),np.array([-1.,0.]),
                  [{"type":"nonnegative","dim":2}]),"primal_infeasible"
    yield Problem(sparse.csc_matrix((1,1)),np.array([-1.]),
                  sparse.csc_matrix([[-1.]]),np.zeros(1),
                  [{"type":"nonnegative","dim":1}]),"dual_infeasible"
    A = sparse.csc_matrix((np.array([-.5,-.5]),np.array([0,0]),np.array([0,2])),
                          shape=(1,1))
    yield Problem(sparse.csc_matrix((1,1)),np.ones(1),A,np.array([-1.]),
                  [{"type":"nonnegative","dim":1}]),"solved"
    yield Problem(sparse.csc_matrix([[2.,1.,0.],[1.,2.,0.],[0.,0.,1.]]),
                  np.zeros(3),sparse.csc_matrix([[1.,1.,0.],[1.,1.,0.],[-1.,0.,0.]]),
                  np.array([0.,1.,0.]),
                  [{"type":"zero","dim":2},{"type":"nonnegative","dim":1}]),"primal_infeasible"
    yield Problem(sparse.csc_matrix([[1.,1.],[1.,1.]]),np.array([-1.,1.]),
                  sparse.csc_matrix([[-1.,0.],[0.,1.]]),np.zeros(2),
                  [{"type":"nonnegative","dim":2}]),"dual_infeasible"


def main():
    psd = PSD(0,3)
    matrix = np.array([[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]])
    expected = np.array([1.,2*np.sqrt(2.),4.,3*np.sqrt(2.),5*np.sqrt(2.),6.])
    assert np.max(abs(psd.vec(matrix)-expected)) < 1e-14
    assert np.max(abs(psd.mat(expected)-matrix)) < 1e-14
    rng = np.random.default_rng(4)
    z = np.tile([-1.,0.,1.],(12,1))
    r = rng.normal(size=z.shape)
    g,H = exp_barrier(z)
    assert np.max(abs(np.einsum("bij,bj->bi",H,exp_hsolve(z,r))-r)) < 1e-12
    A = sparse.csc_matrix(rng.normal(size=(7,5)))
    K = sparse.bmat([[sparse.eye(5),A.T],[A,-sparse.eye(7)]],format="csc")
    upper = sparse.triu(K,format="csc")
    upper.sort_indices()
    factor = LDL(upper,np.r_[np.ones(5),-np.ones(7)])
    factor.factor(upper)
    rhs = rng.normal(size=12)
    assert np.max(abs(K@factor.solve(rhs)-rhs)) < 1e-11
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        for index,(p,status) in enumerate(cases()):
            src, dst = Path(directory)/f"{index}.npz", Path(directory)/f"{index}.out.npz"
            write_input(src,p)
            subprocess.run([str(root/"run.sh"),"--input",str(src),"--output",str(dst)],
                           check=True,timeout=5)
            with np.load(dst,allow_pickle=False) as data:
                meta = data["metadata_json"]
                assert meta.ndim == 0 and meta.dtype.kind == "U"
                result = json.loads(str(meta))
                assert type(result["schema_version"]) is int and result["schema_version"] == 1
                assert result["status"] == status
                result.update({key:data[key] for key in ("x","s","z") if key in data})
            assert assess(Problem.load(src),result,tolerance=1e-8)[0], (index,result)
    print("All numerical and fresh-process interface checks passed.")


if __name__ == "__main__":
    main()
