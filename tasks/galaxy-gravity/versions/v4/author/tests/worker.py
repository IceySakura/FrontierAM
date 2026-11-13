"""Trusted root PID 1 of one fresh offline build or solve container."""
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time

CG = Path("/sys/fs/cgroup")
SOURCE = Path("/workspace/submission")
PRED = Path("/output/output.bin")
MEMORY = 2 << 30


def counters(name):
    return {k: int(v) for k, v in (line.split() for line in (CG / name).read_text().splitlines())}


def checked_copy(source, dest, for_build=False):
    """Copy regular files and directories, rejecting links at the trust boundary."""
    if source.is_symlink() or not source.is_dir():
        raise ValueError("submission must be a directory")
    for parent, dirs, files in os.walk(source):
        target = dest / Path(parent).relative_to(source)
        target.mkdir(mode=0o755, exist_ok=True, parents=True)
        if for_build:
            os.chown(target, 1000, 1000)
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                raise ValueError("directory symlink in submission")
        for name in files:
            src = Path(parent) / name
            st = src.lstat()
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise ValueError("linked or special file in submission")
            dst = target / name
            shutil.copyfile(src, dst)
            dst.chmod(0o755 if st.st_mode & 0o111 else 0o644)
            if for_build:
                os.chown(dst, 1000, 1000)


def kill_and_reap():
    if os.getpid() != 1:
        raise RuntimeError("refusing to clean outside a worker PID namespace")
    while True:
        try:
            os.kill(-1, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            return


def run(mode, cpu, seed, solve_us=2_000_000):
    solve = mode == "run"
    if not 0 < solve_us <= 600_000_000 or mode not in ("run", "build"):
        raise ValueError("invalid mode or solve CPU limit")
    limit_us = solve_us if solve else 120_000_000
    limit = limit_us / 1_000_000
    wall_limit = max(30, (3 * solve_us + 999_999)//1_000_000 + 10) if solve else 300
    pids = 2 if solve else 128
    if (os.getpid() != 1 or os.getuid() != 0 or sys.platform != "linux"
            or os.uname().machine != "aarch64" or os.sched_getaffinity(0) != {cpu}
            or (CG / "cpu.max").read_text().split() != ["100000", "100000"]
            or int((CG / "pids.max").read_text()) != pids
            or int((CG / "memory.max").read_text()) != MEMORY
            or int((CG / "memory.swap.max").read_text()) != 0
            or len(Path("/proc/net/route").read_text().splitlines()) != 1
            or any(line.split()[-1] != "lo" for line in Path("/proc/net/if_inet6").read_text().splitlines())):
        raise RuntimeError("unexpected worker resources, network or architecture")
    directory = SOURCE
    cmd = ["./solve", "/input/snapshot.bin", str(PRED)]
    if not solve:
        directory = Path("/build")
        try:
            checked_copy(SOURCE, directory, for_build=True)
        except ValueError as exc:
            return dict(status="invalid_source", detail=str(exc))
        cmd = ["./build.sh"] if (directory / "build.sh").exists() else ["/bin/true"]
    cgroup_start = counters("cpu.stat")["usage_usec"]
    supervisor_start = time.process_time()
    mem_start = counters("memory.events")
    pids_start = counters("pids.events")
    start = time.monotonic()
    def restrict():
        fallback = (limit_us + 999_999)//1_000_000 + 1
        resource.setrlimit(resource.RLIMIT_CPU, (fallback, fallback))
    env = dict(PATH="/usr/local/bin:/usr/bin:/bin", HOME="/tmp", TMPDIR="/tmp",
               GALAXY_RUN_SEED=str(seed), LC_ALL="C", TZ="UTC", PYTHONDONTWRITEBYTECODE="1")
    if solve:
        env["GALAXY_CPU_LIMIT_S"] = f"{solve_us // 1_000_000}.{solve_us % 1_000_000:06d}"
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[key] = "1"
    try:
        proc = subprocess.Popen(cmd, cwd=directory, user=1000, group=1000, extra_groups=[],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, env=env, preexec_fn=restrict)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        if exc.errno not in (2, 8, 13, 40):
            raise
        return dict(status="entry_error", detail=str(exc))
    status = "ok"
    while proc.poll() is None:
        if solve:
            # Linux per-process CPU clock includes system and user CPU.
            try:
                used_ns = time.clock_gettime_ns((~proc.pid << 3) | 2)
            except (OSError, ProcessLookupError):
                if proc.poll() is not None:
                    break
                raise
        else:
            used_ns = ((counters("cpu.stat")["usage_usec"] - cgroup_start) * 1000
                       - round((time.process_time()-supervisor_start) * 1e9))
        if used_ns >= limit_us * 1000 or time.monotonic() - start >= wall_limit:
            status = "cpu_timeout" if used_ns >= limit_us * 1000 else "wall_timeout"
            proc.kill()
            proc.wait()
            break
        time.sleep(.005)
    elapsed = time.monotonic() - start
    kill_and_reap()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    # wait4()/getrusage returns exact microseconds. Compare integers at the boundary.
    cpu_us = round(usage.ru_utime * 1e6) + round(usage.ru_stime * 1e6)
    if cpu_us > limit_us:
        status = "cpu_timeout"
    if proc.returncode and status == "ok":
        status = "nonzero_exit"
    if counters("pids.events")["max"] != pids_start["max"]:
        status = "process_limit"
    if counters("memory.events")["oom_kill"] != mem_start["oom_kill"]:
        status = "oom"
    record = dict(status=status, exit_code=proc.returncode, cpu_us=cpu_us, cpu_s=cpu_us/1e6,
                  wall_s=elapsed, peak_bytes=int((CG / "memory.peak").read_text()),
                  worker_cpu_s=counters("cpu.stat")["usage_usec"]/1e6, cpu=cpu,
                  limit_s=limit, wall_limit_s=wall_limit)
    if status != "ok":
        return record
    if solve:
        try:
            st = PRED.lstat()
            if (not stat.S_ISREG(st.st_mode) or st.st_nlink != 1
                    or st.st_size != Path("/input/snapshot.bin").stat().st_size//32*24):
                raise ValueError("wrong output length or file type")
            shutil.copyfile(PRED, "/handoff/output.bin")
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            record.update(status="invalid_output", detail=str(exc))
    else:
        try:
            if not (directory / "solve").is_file() or not os.access(directory / "solve", os.X_OK):
                raise ValueError("missing executable ./solve after build")
            checked_copy(directory, Path("/handoff/built"))
        except ValueError as exc:
            record.update(status="invalid_source", detail=str(exc))
    return record


if __name__ == "__main__":
    result = run(sys.argv[1], *map(int, sys.argv[2:]))
    print(json.dumps(result), flush=True)
