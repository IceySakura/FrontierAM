"""Run reviewed programs against public cases; NOT an untrusted-code service.

READY has a separate 15-second startup deadline. The measured solve wall time
starts before delivering the header and ends when both process exit and receipt
of all responses are complete, including one preparation and every request.
CPU/RSS cover the whole child, including startup.
This local POSIX runner uses wait4, not cgroup group accounting or hard affinity.
No security isolation, formal calibration, score, or hidden grading is provided.

run_session returns NumPy vectors in outputs for debugging. CLI JSON omits those
vectors and records correctness, externally measured costs, and repetition data.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdk import wire


MEMORY_CAP_BYTES = 4 * 1024**3
STDERR_CAP_BYTES = 4 * 1024**2


def _load_case(path):
    """Load a trusted public NPZ; never an archive supplied by a contestant."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = ("shape", "indptr", "indices", "data", "b0", "b1", "y_ref", "t", "rtol", "atol")
            arrays = {key: archive[key] for key in required}
        shape = arrays["shape"]
        if shape.shape != (2,) or shape.dtype.kind not in "iu" or shape[0] != shape[1]:
            raise ValueError("matrix shape must be square")
        n = int(shape[0])
        b0, b1, reference = arrays["b0"], arrays["b1"], arrays["y_ref"]
        if b0.ndim != 2 or b0.shape[1] != n or b0.shape[0] not in (1, 16):
            raise ValueError("b0 must have shape (1,n) or (16,n)")
        requests = int(b0.shape[0])
        metadata = dict(version=2, n=n, nnz=int(arrays["data"].size), requests=requests,
                        scenario="one_shot" if requests == 1 else "reuse")
        for name in ("t", "rtol", "atol"):
            if arrays[name].shape != () or arrays[name].dtype.kind not in "fiu":
                raise ValueError(f"{name} must be a real scalar")
            metadata[name] = float(arrays[name])
        wire._validate_metadata(metadata)
        wire._validate_csr(metadata, arrays["indptr"], arrays["indices"], arrays["data"])
        for name, values in (("b0", b0), ("b1", b1), ("y_ref", reference)):
            if (values.shape != (requests, n) or values.dtype.kind not in "fiu"
                    or not np.all(np.isfinite(values))):
                raise ValueError(f"invalid {name}")
            if name != "y_ref" and np.any(np.abs(values) > 1):
                raise ValueError("RHS magnitude exceeds one")
        # Validate the trusted reference threshold separately from candidate IO.
        for vector in reference:
            threshold = metadata["atol"] + metadata["rtol"] * float(np.max(np.abs(vector)))
            if not math.isfinite(threshold):
                raise ValueError("reference accuracy threshold is nonfinite")
        payload = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode("utf-8")
        header = [wire.MAGIC, struct.pack("<I", len(payload)), payload,
                  np.asarray(arrays["indptr"], dtype="<u8").tobytes(),
                  np.asarray(arrays["indices"], dtype="<u4").tobytes(),
                  np.asarray(arrays["data"], dtype="<f8").tobytes()]
        return metadata, header, b0, b1, reference
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"evaluator_error:invalid_public_case:{exc}") from exc


def _check_output(vector, reference, atol, rtol):
    if not np.all(np.isfinite(vector)):
        return {"passed": False, "error_inf": None, "scaled_error": None, "reason": "nonfinite_output"}
    with np.errstate(over="ignore", invalid="ignore"):
        error = float(np.max(np.abs(vector - reference)))
    if not math.isfinite(error):
        return {"passed": False, "error_inf": None, "scaled_error": None, "reason": "error_overflow"}
    threshold = atol + rtol * float(np.max(np.abs(reference)))
    scaled = error / threshold
    passed = error <= threshold
    return {"passed": passed, "error_inf": error,
            "scaled_error": scaled if math.isfinite(scaled) else None,
            "reason": "passed" if passed else "accuracy_failure"}


def _positive_limit(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def run_session(command, case_path, repeat=1, *, startup_timeout=15.0,
                wall_timeout=None, cpu_timeout=None, memory_limit=MEMORY_CAP_BYTES,
                stderr_limit=STDERR_CAP_BYTES, cwd=None):
    """One fresh trusted local process, one shared matrix, K sequential requests.

    command is an argv list or a shlex-split string; no shell is invoked. Smaller
    timeout/resource overrides are intended for local protocol tests. They may
    not increase the fixed protocol caps. Input loading and correctness checking
    occur outside the measured solve interval. Startup has its own deadline,
    while CPU and peak RSS account for the child's complete lifetime.
    CPU/RSS limits are checked from wait4 after exit; they are not hard resource
    isolation. Only reviewed local programs may be used with this runner.
    """
    metadata, header, b0, b1, reference = _load_case(case_path)
    if not hasattr(os, "wait4"):
        raise ValueError("evaluator_error:POSIX_wait4_required")
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be a positive integer")
    argv = shlex.split(command) if isinstance(command, str) else command
    if (not isinstance(argv, (list, tuple)) or not argv or not argv[0]
            or any(not isinstance(item, str) for item in argv)):
        raise ValueError("command must be a nonempty argv sequence")
    cap = 60.0 if metadata["scenario"] == "one_shot" else 240.0
    wall_timeout = cap if wall_timeout is None else wall_timeout
    cpu_timeout = cap if cpu_timeout is None else cpu_timeout
    for value, name, upper in ((startup_timeout, "startup_timeout", 15.0),
                               (wall_timeout, "wall_timeout", cap),
                               (cpu_timeout, "cpu_timeout", cap),
                               (memory_limit, "memory_limit", MEMORY_CAP_BYTES),
                               (stderr_limit, "stderr_limit", STDERR_CAP_BYTES)):
        _positive_limit(value, name)
        if value > upper:
            raise ValueError(f"{name} cannot exceed the protocol cap")

    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    launched = time.perf_counter()
    try:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, bufsize=0, start_new_session=True,
                                   cwd=cwd, env=environment)
    except OSError as exc:
        return {"status": "launch_failure", "valid": False, "reason": type(exc).__name__,
                "repeat": repeat, "scenario": metadata["scenario"], "request_count": metadata["requests"],
                "outputs": [], "errors": [], "time_seconds": None, "cpu_seconds": None,
                "memory_bytes": None, "pid": None}

    state = {"started": None, "exit_time": None, "last_output_time": None,
             "usage": None, "returncode": None,
             "io_done": False, "stderr_done": False, "failure": None,
             "raw_outputs": [], "stderr": bytearray(), "stderr_bytes": 0}
    condition = threading.Condition()

    def update(**values):
        with condition:
            state.update(values)
            condition.notify_all()

    def fail(reason):
        with condition:
            if state["failure"] is None:
                state["failure"] = reason
            condition.notify_all()

    def exchange():
        try:
            if wire._read_exact(process.stdout, len(wire.READY)) != wire.READY:
                fail("invalid_ready")
                return
            update(started=time.perf_counter())  # Before any input is delivered.
            for block in header:
                wire._write_all(process.stdin, block)
            for index in range(metadata["requests"]):
                wire._write_all(process.stdin, np.asarray(b0[index], dtype="<f8").tobytes())
                wire._write_all(process.stdin, np.asarray(b1[index], dtype="<f8").tobytes())
                process.stdin.flush()
                # Never send the following RHS before this response is complete.
                output = wire._read_exact(process.stdout, 8 * metadata["n"])
                state["raw_outputs"].append(output)
                update(last_output_time=time.perf_counter())
            process.stdin.close()
            if process.stdout.read(1):
                fail("extra_stdout")
        except wire.ProtocolError:
            fail("short_output" if state["started"] is not None else "incomplete_ready")
        except (BrokenPipeError, OSError, ValueError):
            fail("io_failure")
        finally:
            update(io_done=True)

    def drain_stderr():
        try:
            while True:
                block = process.stderr.read(65536)
                if not block:
                    break
                state["stderr_bytes"] += len(block)
                available = max(0, int(stderr_limit) - len(state["stderr"]))
                state["stderr"].extend(block[:available])
                if state["stderr_bytes"] > stderr_limit:
                    fail("stderr_limit")
        except (OSError, ValueError):
            fail("stderr_read_failure")
        finally:
            update(stderr_done=True)

    def collect_exit():
        try:
            _, status, usage = os.wait4(process.pid, 0)
            exited = time.perf_counter()
            code = os.waitstatus_to_exitcode(status)
            process.returncode = code  # Prevent Popen from reaping the same child.
            update(exit_time=exited, usage=usage, returncode=code)
        except (ChildProcessError, OSError):
            fail("wait4_failure")
            # Do not wait forever for an exit timestamp that this collector
            # can no longer produce. The main loop kills and reaps the child.
            update(exit_time=time.perf_counter())

    threads = [threading.Thread(target=target, daemon=True)
               for target in (exchange, drain_stderr, collect_exit)]
    for thread in threads:
        thread.start()
    terminated = False
    while True:
        with condition:
            now = time.perf_counter()
            if state["failure"] is None:
                if state["started"] is None and now - launched > startup_timeout:
                    state["failure"] = "startup_timeout"
                elif (state["started"] is not None and state["exit_time"] is None
                      and now - state["started"] > wall_timeout):
                    state["failure"] = "wall_timeout"
                elif state["returncode"] is not None and state["returncode"] != 0:
                    state["failure"] = "nonzero_exit"
                elif (state["exit_time"] is not None
                      and now - state["exit_time"] > 1.0
                      and not (state["io_done"] and state["stderr_done"])):
                    state["failure"] = "pipes_held_after_exit"
            done = state["exit_time"] is not None and state["io_done"] and state["stderr_done"]
            should_kill = state["failure"] is not None and not terminated
            if done:
                break
            if not should_kill:
                condition.wait(timeout=0.01)
        if should_kill:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            terminated = True
    # Trusted local cleanup prevents state carrying into statistical repeats.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # macOS can reject a second signal to an already terminated group.
        pass
    for thread in threads:
        thread.join(timeout=1)
    if state["usage"] is None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    for stream in (process.stdin, process.stdout, process.stderr):
        if not stream.closed:
            stream.close()

    usage = state["usage"]
    cpu_seconds = usage.ru_utime + usage.ru_stime if usage is not None else None
    peak_bytes = (int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
                  if usage is not None else None)
    ended = max((value for value in (state["exit_time"], state["last_output_time"])
                 if value is not None), default=None)
    wall_seconds = (max(0.0, ended - state["started"])
                    if state["exit_time"] is not None and state["started"] is not None else None)
    status = state["failure"]
    if status is None:
        if wall_seconds is None or wall_seconds > wall_timeout:
            status = "wall_timeout"
        elif cpu_seconds is None or cpu_seconds > cpu_timeout:
            status = "cpu_limit"
        elif peak_bytes is None or peak_bytes > memory_limit:
            status = "memory_limit"

    # All output decoding and reference comparison happen AFTER the exit clock.
    outputs = [np.frombuffer(raw, dtype="<f8") for raw in state["raw_outputs"]]
    errors = [_check_output(vector, reference[index], metadata["atol"], metadata["rtol"])
              for index, vector in enumerate(outputs)]
    if status is None and (len(errors) != metadata["requests"] or not all(e["passed"] for e in errors)):
        status = "accuracy_failure"
    status = "ok" if status is None else status
    return {"status": status, "valid": status == "ok", "reason": status,
            "repeat": repeat, "pid": process.pid, "returncode": state["returncode"],
            "scenario": metadata["scenario"], "request_count": metadata["requests"],
            "n": metadata["n"], "nnz": metadata["nnz"], "t": metadata["t"],
            "rtol": metadata["rtol"], "atol": metadata["atol"],
            "time_seconds": wall_seconds, "cpu_seconds": cpu_seconds, "memory_bytes": peak_bytes,
            "child_exit_seconds": (max(0.0, state["exit_time"] - state["started"])
                                   if state["exit_time"] is not None and state["started"] is not None else None),
            "last_output_seconds": (max(0.0, state["last_output_time"] - state["started"])
                                    if state["last_output_time"] is not None and state["started"] is not None else None),
            "startup_seconds": (state["started"] - launched) if state["started"] is not None else None,
            "stderr_bytes": state["stderr_bytes"],
            "stderr": bytes(state["stderr"]).decode("utf-8", errors="replace"),
            "outputs": outputs, "errors": errors,
            "measurement_scope": "trusted local single-child wait4; not cgroup or sandbox"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--command", required=True, help="argv parsed with shlex; no shell")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 100:
        parser.error("--repeats must be between 1 and 100; official aggregates require 3")
    manifest = json.loads(args.manifest.read_text())
    cases = manifest.get("cases") if isinstance(manifest, dict) else manifest
    if not isinstance(cases, list) or not cases:
        parser.error("manifest must contain a nonempty cases list")
    ids = set()
    runs, aggregates = [], {}
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get("id"), str)
                or not case["id"] or case["id"] in ids):
            parser.error("manifest case IDs must be unique nonempty strings")
        ids.add(case["id"])
        relative = case.get("file", case.get("path"))
        if not isinstance(relative, str):
            parser.error("each manifest case needs file or path")
        path = args.manifest.resolve().parent / relative
        measured = []
        for repeat in range(1, args.repeats + 1):
            record = run_session(args.command, path, repeat=repeat)
            record = {key: value for key, value in record.items() if key != "outputs"}
            record["id"] = case["id"]
            measured.append(record)
            runs.append(record)
        valid = all(row["valid"] for row in measured)
        wall = [row["time_seconds"] for row in measured if row["time_seconds"] is not None]
        memory = [row["memory_bytes"] for row in measured if row["memory_bytes"] is not None]
        cpu = [row["cpu_seconds"] for row in measured if row["cpu_seconds"] is not None]
        aggregates[case["id"]] = {
            "valid": valid, "reason": "ok" if valid else ";".join(sorted({r["reason"] for r in measured if not r["valid"]})),
            "repeat_count": len(measured), "all_repeats_valid": valid,
            "time_seconds": statistics.median(wall) if wall else None,
            "memory_bytes": max(memory) if memory else None,
            "max_wall_seconds": max(wall) if wall else None,
            "max_cpu_seconds": max(cpu) if cpu else None,
            "request_count": measured[0]["request_count"],
        }
    report = {"protocol": "FAMEXP2", "scope": "public trusted local development only",
              "calibrated_reference_available": False, "score": None,
              "repeats": args.repeats, "runs": runs, "aggregates": aggregates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote correctness and cost report to {args.output}; no calibrated score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
