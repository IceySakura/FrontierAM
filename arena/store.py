"""Filesystem store separating frozen development output from grading jobs, results and baselines.

    submissions/<id>/   meta.json, source/ (frozen regular files), source.tar, dev/ (development evidence)
    jobs/<state>/<id>.json   state in pending | running | done | failed; claimed by atomic rename
    results/<job>/      grader evidence, result.json (raw metrics), score.json
    baselines/<task>/<id>/  reference grading evidence; ACTIVE names the one used for scoring

Grading never reads development evidence: a job references only the frozen source of a submission.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import shutil
import time

from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB_STATES = ('pending', 'running', 'done', 'failed')


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def new_id(prefix: str) -> str:
    return f'{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}'


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


class Store:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.environ.get('FRONTIERAM_STORE') or ROOT / 'var/stores/store').resolve()
        for state in JOB_STATES:
            (self.root / 'jobs' / state).mkdir(parents=True, exist_ok=True)
        (self.root / 'submissions').mkdir(exist_ok=True)
        (self.root / 'results').mkdir(exist_ok=True)
        (self.root / 'baselines').mkdir(exist_ok=True)

    # Submissions -------------------------------------------------------------------------------------------------

    def submission_dir(self, submission_id: str) -> Path:
        return self.root / 'submissions' / submission_id

    def new_submission(self, task: str, origin: dict) -> tuple[str, Path]:
        submission_id = new_id(task)
        directory = self.submission_dir(submission_id)
        directory.mkdir(parents=True)
        write_json(directory / 'meta.json', {'id': submission_id, 'task': task, 'created_utc': utc(), **origin})
        return submission_id, directory

    def submission(self, submission_id: str) -> dict:
        return read_json(self.submission_dir(submission_id) / 'meta.json')

    def update_submission(self, submission_id: str, **fields) -> dict:
        meta = {**self.submission(submission_id), **fields}
        write_json(self.submission_dir(submission_id) / 'meta.json', meta)
        return meta

    def submissions(self, task: str | None = None) -> list[dict]:
        rows = []
        for path in sorted((self.root / 'submissions').glob('*/meta.json')):
            meta = read_json(path)
            if task is None or meta['task'] == task:
                rows.append(meta)
        return rows

    # Grading queue -----------------------------------------------------------------------------------------------

    def enqueue(
        self, submission_id: str, *, kind: str = 'candidate', note: str | None = None, isolation: str | None = None,
        extra: dict | None = None, queue: str = 'pending',
    ) -> str:
        meta = self.submission(submission_id)
        if not (self.submission_dir(submission_id) / 'source.tar').is_file():
            raise ValueError(f'{submission_id} has no frozen source to grade')
        job_id = new_id('job')
        # 'held' jobs are invisible to workers until moved to pending.
        (self.root / 'jobs' / queue).mkdir(parents=True, exist_ok=True)
        write_json(
            self.root / 'jobs' / queue / f'{job_id}.json',
            {
                'id': job_id,
                'task': meta['task'],
                'submission': submission_id,
                'kind': kind,
                'note': note,
                'isolation': isolation,
                'enqueued_utc': utc(),
                **(extra or {}),
            },
        )
        return job_id

    def claim(self, *, tasks: list[str] | None = None, job_id: str | None = None) -> dict | None:
        """Atomically move one pending job to running; a rename has exactly one winner across processes."""
        for path in sorted((self.root / 'jobs/pending').glob('*.json')):
            if job_id is not None and path.stem != job_id:
                continue
            try:
                job = read_json(path)
            except (OSError, ValueError):
                continue
            if tasks and job['task'] not in tasks:
                continue
            lock = self._job_lock(job['id'])
            if lock is None:
                continue
            target = self.root / 'jobs/running' / path.name
            try:
                os.rename(path, target)
            except FileNotFoundError:
                os.close(lock)
                continue
            job.update(claimed_utc=utc(), worker_pid=os.getpid(), lock_fd=lock)
            write_json(target, {k: v for k, v in job.items() if k != 'lock_fd'})
            return job
        return None

    def _job_lock(self, job_id: str) -> int | None:
        """Held for the whole grading, inherited by the grading child; free means the job is abandoned."""
        directory = self.root / 'results' / job_id
        directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(directory / '.lock', os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        os.set_inheritable(fd, True)
        return fd

    def finish(self, job: dict, state: str, **fields) -> dict:
        if state not in ('done', 'failed'):
            raise ValueError(state)
        record = {k: v for k, v in job.items() if k != 'lock_fd'} | fields | {'finished_utc': utc()}
        write_json(self.root / 'jobs' / state / f'{job["id"]}.json', record)
        (self.root / 'jobs/running' / f'{job["id"]}.json').unlink(missing_ok=True)
        if job.get('lock_fd') is not None:
            os.close(job['lock_fd'])
        return record

    def recover(self) -> list[str]:
        """Requeue running jobs whose worker died; a live worker still holds the result lock."""
        requeued = []
        for path in sorted((self.root / 'jobs/running').glob('*.json')):
            job = read_json(path)
            fd = self._job_lock(job['id'])
            if fd is None:
                continue
            try:
                stale = self.root / 'results' / job['id']
                # Keep abandoned evidence; the requeued attempt starts from an empty result directory.
                archive = stale.with_name(f'{job["id"]}.abandoned-{int(time.time())}')
                archive.mkdir()
                for item in stale.iterdir():
                    if item.name != '.lock':
                        shutil.move(item, archive / item.name)
                job.pop('claimed_utc', None)
                job.pop('worker_pid', None)
                job['requeued_utc'] = utc()
                write_json(self.root / 'jobs/pending' / path.name, job)
                path.unlink()
                requeued.append(job['id'])
            finally:
                os.close(fd)
        return requeued

    def jobs(self, state: str) -> list[dict]:
        return [read_json(p) for p in sorted((self.root / 'jobs' / state).glob('*.json'))]

    def result_dir(self, job_id: str) -> Path:
        return self.root / 'results' / job_id

    # Baselines ---------------------------------------------------------------------------------------------------

    def baseline_root(self, task: str) -> Path:
        return self.root / 'baselines' / task

    def active_baseline(self, task: str) -> dict | None:
        pointer = self.baseline_root(task) / 'ACTIVE'
        if not pointer.exists():
            return None
        return read_json(self.baseline_root(task) / pointer.read_text().strip() / 'baseline.json')

    def activate_baseline(self, task: str, baseline_id: str) -> None:
        pointer = self.baseline_root(task) / 'ACTIVE'
        temporary = pointer.with_name('.ACTIVE.tmp')
        temporary.write_text(baseline_id + '\n')
        temporary.replace(pointer)
