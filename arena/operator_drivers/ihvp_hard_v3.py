"""ihvp-sequence-v3 (author's hard-data extension): the IHVPSequence deployment with the v3 grader, whose loader reads both
the retained v1 teachers and the added hard teachers. Formal scores follow task.json status (draft: unmeasured).

Explicit local diagnostics without a slot: python arena/operator_drivers/ihvp_hard_v3.py --smoke --out DIR
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

VERSION = Path(__file__).resolve().parents[2] / 'tasks/ihvp-sequence/versions/v3'

if __name__ != '__main__':
    from .ihvp_sequence import IHVPSequence

    class IHVPHardV3(IHVPSequence):
        grader_version = 'v3'


if __name__ == '__main__':
    os.execv(sys.executable, [sys.executable, str(VERSION / 'grader/evaluate.py'), *sys.argv[1:]])
