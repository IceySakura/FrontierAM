"""Refresh hash locks after a reviewed edit to trusted grading code, recording old and new digests.

Only the files listed in TRUSTED may be relocked; data, public packages and references are never relocked here.
"""

import hashlib
import json
import re

from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'nexus/nexus/extensions/tasks/frontieram_offline/assets.py'
PROFILE = ROOT / 'tasks/conic/versions/original/profile_template.json'
RECORD = ROOT / 'provenance/TRUSTED_CODE_RELOCKS.json'
EXPONENTIAL = ROOT / 'tasks/exponential/versions/v1/docker_eval'
CONIC = ROOT / 'tasks/conic/versions/original/package'
TRUSTED = {
    'exponential': ['campaign_runner_v2.py', 'evaluator/run_docker.py'],
    'conic': ['tests/grade.py', 'tests/runtime_sandbox.py'],
}
REASON = 'Slot placement: the single pinned CPU and its isolated cgroup slice are parameters instead of CPU0.'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    changes = []
    code = ASSETS.read_text()
    for relative in TRUSTED['exponential']:
        pattern = re.compile(rf"('{re.escape(relative)}': ')([0-9a-f]{{64}})(')")
        old = pattern.search(code).group(2)
        new = sha(EXPONENTIAL / relative)
        if old != new:
            code = pattern.sub(lambda m: m.group(1) + new + m.group(3), code)
            changes.append({'file': f'tasks/exponential/versions/v1/docker_eval/{relative}', 'old': old, 'new': new})
    ASSETS.write_text(code)
    profile = json.loads(PROFILE.read_text())
    for relative in TRUSTED['conic']:
        old, new = profile['locked_files'][relative], sha(CONIC / relative)
        if old != new:
            profile['locked_files'][relative] = new
            changes.append({'file': f'tasks/conic/versions/original/package/{relative}', 'old': old, 'new': new})
    PROFILE.write_text(json.dumps(profile, indent=2) + '\n')
    record = json.loads(RECORD.read_text()) if RECORD.exists() else {'reason': REASON, 'relocks': []}
    if changes:
        record['relocks'].append({'utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'changes': changes})
        RECORD.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(changes, indent=2))


if __name__ == '__main__':
    main()
