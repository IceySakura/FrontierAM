"""Run untrusted code as a dedicated UID inside a disposable verifier container."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import resource
import signal
import subprocess
import tempfile
import time

WORKER_UID = 1000
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS",
)}
FILE_LIMIT = 4 * 1024 * 1024


def worker_pids(uid: int) -> list[int]:
    """The container dedicates this UID to one submission, including escaped sessions."""
    pids = []
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            owner = next(line for line in path.read_text().splitlines() if line.startswith("Uid:"))
            if int(owner.split()[1]) == uid:
                pids.append(int(path.parent.name))
        except (OSError, StopIteration):
            continue
    return pids


def run(command: list[str], cwd: Path, timeout: float, *, uid: int = WORKER_UID) -> dict:
    if os.geteuid() != 0 or uid == 0:
        raise RuntimeError("QP supervision requires root and a dedicated non-root worker UID")
    if worker_pids(uid):
        raise RuntimeError("worker UID already in use; use a fresh verifier container")
    # Adopt orphaned descendants, including double-forked background processes.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")

    def restrict():
        if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1, int(timeout) + 1))
        # Compilers may produce binaries larger than the result/log allowance.
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))

    env = {**THREAD_ENV, "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(cwd), "TMPDIR": str(cwd),
           "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    report = {"outcome": "algorithm_failure", "reason": None}
    started = time.monotonic()
    process = None
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, start_new_session=True,
                                       preexec_fn=restrict)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                report["reason"] = "time_limit"
            elapsed = time.monotonic() - started
            if report["reason"] is None:
                if elapsed > timeout:
                    report["reason"] = "time_limit"
                elif process.returncode != 0:
                    report["reason"] = "nonzero_exit"
                elif worker_pids(uid):
                    report["reason"] = "background_process"
                else:
                    report["outcome"] = "completed"
        except OSError as exc:
            report["reason"] = f"entry_error:{type(exc).__name__}"
        finally:
            # Also kill children that used setsid() to escape the original group.
            for pid in worker_pids(uid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process is not None:
                process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while worker_pids(uid) and time.monotonic() < deadline:
                try:
                    os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                time.sleep(0.01)
            report["elapsed_seconds"] = time.monotonic() - started
            report["exit_code"] = process.returncode if process is not None else None
            for name, stream in (("stdout", stdout), ("stderr", stderr)):
                if os.fstat(stream.fileno()).st_size > FILE_LIMIT:
                    report.update(outcome="algorithm_failure", reason="output_limit")
                stream.seek(0)
                report[name] = stream.read(16384).decode("utf-8", "replace")
    return report
