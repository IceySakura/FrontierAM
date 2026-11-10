"""Record the host-wide task count of one UID (what RLIMIT_NPROC counts) until stopped.

Usage: uid-sampler.py UID OUTPUT.jsonl [interval_seconds]
"""

import json
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena.cli import uid_tasks  # noqa: E402

uid, output = int(sys.argv[1]), Path(sys.argv[2])
interval = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
peak = 0
with output.open('a') as log:
    while True:
        count = uid_tasks(uid)
        if count > peak or int(time.time()) % 30 == 0:
            peak = max(peak, count)
            log.write(json.dumps({'unix': round(time.time(), 2), 'tasks': count, 'peak': peak}) + '\n')
            log.flush()
        time.sleep(interval)
