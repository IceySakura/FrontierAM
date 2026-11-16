"""In-container control process and the unprivileged Python submission adapter."""
import json
import os
from pathlib import Path
import resource
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SOURCE = Path("/workspace/submission")
CGROUP = Path("/sys/fs/cgroup")


def submit():
    # The control process never imports the submission or NumPy.
    import numpy as np

    sys.path.insert(0, str(SOURCE))
    from solution import reconstruct

    raw = np.load("/input/raw.npy", allow_pickle=False, mmap_mode="r")
    prediction = reconstruct(raw)
    if (not isinstance(prediction, np.ndarray) or prediction.shape != (4, *raw.shape)
            or prediction.dtype.kind != "f" or prediction.dtype.itemsize not in (4, 8)):
        raise ValueError("return must be a (4,H,W) float32 or float64 ndarray")
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite prediction")
    with open("/output/pred.npy", "wb") as stream:
        np.save(stream, prediction, allow_pickle=False)


def supervise(mode, limit):
    cpu = sorted(os.sched_getaffinity(0))
    cpu_max = (CGROUP / "cpu.max").read_text().strip()
    memory = int((CGROUP / "memory.max").read_text())
    swap = int((CGROUP / "memory.swap.max").read_text())
    if len(cpu) != 1 or cpu_max != "100000 100000" or memory != 8 * 2**30 or swap != 0:
        raise RuntimeError("unexpected cgroup configuration")
    if mode == "build":
        cmd = ["./build.sh"] if (SOURCE / "build.sh").exists() else ["/usr/bin/true"]
    else:
        pred = Path("/output/pred.npy")
        pred.touch(mode=0o222, exist_ok=False)
        pred.chmod(0o222)
        cmd = [sys.executable, "-I", "-B", __file__, "submit"]
    def child_limits():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if mode == "run":
            maximum = 4 * 1024**2 * 8 + 4096
            resource.setrlimit(resource.RLIMIT_FSIZE, (maximum, maximum))

    with tempfile.TemporaryFile() as log:
        begin = time.monotonic()
        status, reason = "ok", ""
        try:
            p = subprocess.Popen(cmd, cwd=SOURCE, user=1000, group=1000, extra_groups=[],
                                 preexec_fn=child_limits, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        except OSError as exc:
            # The build script may specify an invalid interpreter or executable format.
            if mode != "build" or exc.errno not in (2, 8, 13):
                raise
            p = None
            status, reason = "submission_failure", "build entry: " + str(exc)
        try:
            if p:
                fd = os.pidfd_open(p.pid)
                try:
                    if not select.select([fd], [], [], max(0, limit - (time.monotonic() - begin)))[0]:
                        raise subprocess.TimeoutExpired(cmd, limit)
                finally:
                    os.close(fd)
                code = p.wait()
                if code != 0:
                    status, reason = "submission_failure", f"exit {code}"
        except subprocess.TimeoutExpired:
            status, reason = "submission_failure", "timeout"
        finally:
            wall = time.monotonic() - begin
            if wall > limit and p is not None:
                status, reason = "submission_failure", "timeout"
            # PID 1 kills every other process in this fresh PID namespace, including
            # detached descendants. The host also removes the whole Docker container.
            try:
                os.kill(-1, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if p:
                p.wait(timeout=4)
        log.seek(max(0, log.seek(0, os.SEEK_END) - 4096))
        tail = log.read(4096).decode("utf8", "replace")
    events = {k: int(v) for k, v in (s.split() for s in (CGROUP / "memory.events").read_text().splitlines())}
    if events["oom_kill"]:
        status, reason = "submission_failure", "oom"
    if mode == "run" and status == "ok":
        # /output is native Linux tmpfs; /handoff is accessible only to the supervisor.
        pred.chmod(0o600)
        shutil.copyfile(pred, "/handoff/pred.npy")
    record = dict(status=status, reason=reason, wall_s=wall, peak_bytes=int((CGROUP/"memory.peak").read_text()),
                  returncode=p.returncode if p else None, log_tail=tail, cpu=cpu,
                  cpu_max=cpu_max, memory_limit_bytes=memory, swap_limit_bytes=swap, memory_events=events)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "submit":
        submit()
    else:
        supervise(sys.argv[1], float(sys.argv[2]))
