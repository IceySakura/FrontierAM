"""ihvp-sequence-v5: real-training-trajectory replay (PR #5), run by versions/v5/grader/evaluate.py like the v4 adapter.

Pools come from the author's balanced-split-20261117 manifest (scripts/seal/seal_ihvp_v5.py): 15 public, 30 private,
each case a directory of 50 step operators plus its reference; the platform's weak/strong references (Warm CG, Cold CG)
are measured on the same pool.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
VERSION = Path(__file__).resolve().parents[2] / 'tasks/ihvp-sequence/versions/v5'
if __name__ != '__main__':
    from .ihvp_sequence import IHVPSequence

    class IHVPRealtraceV5(IHVPSequence):
        grader_version = 'v5'

        def data_dir(self, split=None):
            split = split or os.environ.get('FRONTIERAM_IHVP_DATA', 'private')
            if split not in ('public', 'private'):
                raise ValueError('v5 grading uses only the balanced public/private pools')
            path = super().data_dir(split)
            if path is not None and (path / 'manifest.json').exists():
                manifest = json.loads((path / 'manifest.json').read_text())
                if (manifest.get('version') != 'v5' or manifest.get('split') != split
                        or len(manifest['cases']) != {'public': 15, 'private': 30}[split]
                        or manifest.get('expected_cases') != len(manifest['cases'])):
                    raise ValueError('wrong v5 split/version/count')
            return path
else:
    os.execv(sys.executable, [sys.executable, str(VERSION / 'grader/evaluate.py'), *sys.argv[1:]])
