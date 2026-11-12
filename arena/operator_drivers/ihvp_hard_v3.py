"""Author-side hard-data proposal; formal activation belongs to the maintainer.

The explicit CLI runs v3's local evaluator. It is not a security boundary for
untrusted submissions and does not reuse any published v1/v2 numeric anchors.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

VERSION = Path(__file__).resolve().parents[2] / 'tasks/ihvp-sequence/versions/v3'

if __name__ != '__main__':
    from .ihvp_sequence import IHVPSequence

    class IHVPHardV3(IHVPSequence):
        def assets(self, runtime):
            raise RuntimeError('ihvp-sequence-v3 is a hard-data draft; the maintainer must seal its new data and oracle-only samples and recalibrate it before model attempts')

        def run_grader(self, source, out, placement):
            return {'status': 'infrastructure_failure', 'raw_error': None,
                    'reason': 'v3 hard data not activated: target-host isolation, calibration and feedback time-envelope acceptance pending'}

        def score(self, runtime, metrics, baseline):
            return {'status': 'unmeasured', 'score': None, 'raw_error': None,
                    'reason': 'v3 is a separate draft; published v1/v2 scores remain unchanged'}


if __name__ == '__main__':
    os.execv(sys.executable, [sys.executable, str(VERSION / 'grader/evaluate.py'), *sys.argv[1:]])
