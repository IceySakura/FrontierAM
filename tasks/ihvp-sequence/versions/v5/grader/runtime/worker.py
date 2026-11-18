"""Trusted current-step worker launcher; user stdout is redirected to stderr."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
try:
    import api as protocol
except ImportError:
    import ihvp_api as protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--submission",required=True,type=Path)
    parser.add_argument("--dimension",required=True,type=int)
    parser.add_argument("--steps",required=True,type=int)
    parser.add_argument("--seed",required=True,type=int)
    parser.add_argument("--options",default="{}")
    args=parser.parse_args()
    incoming,outgoing=sys.stdin.buffer,sys.stdout.buffer
    # Avoid interpreting solver print output as protocol messages.
    sys.stdout=sys.stderr
    sys.modules.setdefault("contracts",protocol)
    sys.modules.setdefault("ihvp_api",protocol)
    sys.path.insert(0,str(args.submission.resolve()))
    try:
        spec=importlib.util.spec_from_file_location("submitted_solver",args.submission/"solver.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        solver=module.make_solver(args.dimension,args.seed,json.loads(args.options))
        protocol.send_message(outgoing,"ready")
        for tick in range(args.steps):
            protocol.send_message(outgoing,"next",step=tick)
            header,b=protocol.receive_message(incoming,expected_dimension=args.dimension)
            if header.get("kind")!="step" or b is None or b.shape!=(args.dimension,):
                raise protocol.ProtocolError("invalid step command")
            context=header["context"]
            def hvp(vector):
                if time.monotonic()>=context["sequence_deadline"]:
                    raise protocol.BudgetExceeded("sequence oracle deadline")
                vector=np.asarray(vector,dtype=np.float64)
                if (vector.ndim not in (1,2) or vector.shape[0]!=args.dimension
                    or not np.isfinite(vector).all()
                    or (vector.ndim==2 and vector.shape[1]>protocol.MAX_BLOCK_COLUMNS)):
                    raise ValueError("HVP requires a finite d-vector or d-by-k block, k<=64")
                protocol.send_message(outgoing,"hvp",vector,step=tick)
                answer,value=protocol.receive_message(incoming,expected_dimension=args.dimension)
                if answer.get("kind")=="budget":
                    raise protocol.BudgetExceeded("sequence oracle deadline")
                if answer.get("kind")!="product" or value is None or value.shape!=vector.shape:
                    raise protocol.ProtocolError("invalid oracle reply")
                return value
            try:
                returned=solver.solve(hvp,b,context)
                if isinstance(returned,tuple) and len(returned)==2:
                    estimate,diagnostics=returned
                else:
                    estimate,diagnostics=returned,{}
                if not isinstance(diagnostics,dict):diagnostics={"note":str(diagnostics)}
                estimate=np.asarray(estimate,dtype=np.float64)
                if estimate.shape!=(args.dimension,) or not np.isfinite(estimate).all():
                    raise ValueError("solve must return one finite d-vector")
                error=None
            except Exception as exc:
                estimate=np.zeros(args.dimension);diagnostics={};error=f"{type(exc).__name__}: {exc}"
            protocol.send_message(outgoing,"result",estimate,step=tick,
                                  diagnostics=diagnostics,error=error)
        usage=resource.getrusage(resource.RUSAGE_SELF)
        raw_peak=int(usage.ru_maxrss if sys.platform=="darwin" else usage.ru_maxrss*1024)
        peak=raw_peak
        if sys.platform.startswith("linux"):
            # ru_maxrss can retain the large pre-exec teacher image after fork.
            # VmHWM measures this executable's own address space, including native allocations.
            peak=next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:'))
        protocol.send_message(outgoing,"done",usage={"peak_rss_bytes":peak,
                              "raw_rusage_peak_rss_bytes":raw_peak,
                              "cpu_user_seconds":usage.ru_utime,"cpu_system_seconds":usage.ru_stime})
    except BaseException as exc:
        protocol.send_message(outgoing,"fatal",error=f"{type(exc).__name__}: {exc}")
        return 1
    return 0

if __name__=="__main__":
    raise SystemExit(main())
