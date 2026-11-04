"""Record a grade whose worker was stopped while the grading child ran on: once the child has exited (its job lock
is free) and left metrics.json, score it exactly as the worker would and move the job to done.

Usage: finish-orphan.py STORE JOB_ID [JOB_ID ...]   (waits for each child to exit)
"""

import json
import sys
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arena.runtime import load_runtime  # noqa: E402
from arena.store import Store, read_json, write_json  # noqa: E402
from arena.tasks import TASKS  # noqa: E402


def main():
    store = Store(sys.argv[1])
    runtime = load_runtime()
    for job_id in sys.argv[2:]:
        path = store.root / 'jobs/running' / f'{job_id}.json'
        while (lock := store._job_lock(job_id)) is None:
            time.sleep(30)
        job = read_json(path) | {'lock_fd': lock}
        task = TASKS[job['task']]
        results = store.result_dir(job_id)
        metrics_path = results / 'grade/metrics.json'
        if not metrics_path.exists():
            store.finish(job, 'failed', status='unmeasured', reason='orphaned grading child left no metrics')
            print(job_id, 'unmeasured')
            continue
        metrics = read_json(metrics_path)
        baseline = store.active_baseline(task.name)
        score = task.score(runtime, metrics, baseline)
        write_json(results / 'score.json', {**score, 'baseline': (baseline or {}).get('id'),
                                            'recorded_by': 'finish-orphan.py (worker stopped during grading)'})
        store.update_submission(job['submission'], status='graded', score=score.get('score'),
                                score_status=score.get('status'))
        store.finish(job, 'done', status=score.get('status'), score=score.get('score'), recorded_by='finish-orphan.py')
        print(job_id, json.dumps({k: v for k, v in score.items() if not isinstance(v, (list, dict))}))


if __name__ == '__main__':
    main()
