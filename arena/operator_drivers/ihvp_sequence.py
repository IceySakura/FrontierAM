"""Draft IHVP task registration and explicit local smoke/recalibration entry.

There is intentionally no active host-subprocess formal grader. The local
protocol is not adversarial isolation; a maintainer must validate amd64 offline
execution and implement the isolated deployment before opening scored attempts.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
VERSION=ROOT/'tasks'/'ihvp-sequence'/'versions'/'v1'

if __name__!='__main__':
    from ..operator_tasks import OperatorTask
    from ..tasks import GIB

    class IHVPSequence(OperatorTask):
        grade_memory=4*GIB  # Host/reference data; submitted worker contract is 256 MiB.
        grade_timeout=3600
        grade_isolation='quiet'

        def assets(self,runtime):
            raise RuntimeError('ihvp-sequence is draft; the isolated amd64 deployment is not sealed. Use the explicit local --smoke/--recalibrate CLI, not a model attempt.')

        def dev_config(self,*args,**kwargs):
            raise ValueError('ihvp-sequence is not open for model attempts before maintainer release gates pass')

        def run_grader(self,source,out,placement):
            return {'status':'infrastructure_failure','raw_error':None,
                    'reason':'ihvp-sequence is draft: target-machine calibration and release gates, including isolated amd64 worker execution, are pending. Explicit local smoke/recalibration is available; historical measurements are not active anchors.'}

        def baseline_ok(self,metrics):
            return False,'draft reference: local measurement, 25% headroom, two-run stability, non-saturation and isolated amd64 release gates must be reviewed by the maintainer'

        def score(self,runtime,metrics,baseline):
            return {'status':'unmeasured','score':None,'native_score':None,
                    'raw_error':(metrics.get('parts',{}).get('candidate') or {}).get('raw_error'),
                    'reason':'task is not activated; missing/unverified platform or calibration evidence is not a zero score'}


def main():
    # Explicit local diagnostics only; no hidden data is copied into a worker.
    grader=VERSION/'grader'/'evaluate.py'
    os.execv(sys.executable,[sys.executable,str(grader),*sys.argv[1:]])

if __name__=='__main__':main()
