"""Draft routed-holdout IHVP adapter; formal scores remain disabled by task status."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
VERSION = Path(__file__).resolve().parents[2] / 'tasks/ihvp-sequence/versions/v4'
if __name__ != '__main__':
    from .ihvp_sequence import IHVPSequence
    class IHVPHardV4(IHVPSequence):
        grader_version = 'v4'
        def data_dir(self, split=None):
            split = split or os.environ.get('FRONTIERAM_IHVP_DATA', 'private')
            if split not in ('public','private'):
                raise ValueError('v4 grading uses only fresh public/private; samples use the host sample service')
            path = super().data_dir(split)
            if path is not None and (path/'manifest.json').exists():
                manifest=json.loads((path/'manifest.json').read_text())
                expected={'public':15,'private':30}[split]
                if manifest.get('version') != 'v4' or manifest.get('split') != split or len(manifest['cases']) != expected:
                    raise ValueError('wrong v4 split/version/count')
                if any(c.get('kind') != 'hard_factored_v4' or not c['id'].startswith('v4_'+split+'_') for c in manifest['cases']):
                    raise ValueError('legacy/replay teachers cannot enter v4 scoring')
            return path
else:
    os.execv(sys.executable,[sys.executable,str(VERSION/'grader/evaluate.py'),*sys.argv[1:]])
