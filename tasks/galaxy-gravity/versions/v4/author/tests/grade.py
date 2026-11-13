"""Private host grader. Reuse the project's validated Docker and bind-mount helpers."""
import argparse
from collections import defaultdict
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import sys
import tempfile
import uuid

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "scripts/weighted_tree_qmc/step5/sandbox.py").is_file())
sys.path.insert(0, str(ROOT / "scripts/weighted_tree_qmc/step5"))
from sandbox import bind, digest, docker, write_json

MEMORY = 2 << 30
SCALES = (65536, 262144, 1048576, 4194304)
FAMILIES = ("smooth-hernquist", "disc", "group", "jaffe", "einasto")
RAW_MODES = ("diagnostic", "timing", "pilot", "precalibration", "calibration")
RECOVERY_MODES = ("timing", "precalibration", "calibration")


def solve_limit_us(cfg, entry=None):
    seconds = (entry or {}).get("cpu_limit_s", cfg.get("cpu_limit_s", 2))
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError("cpu_limit_s must be a positive number at most 600 seconds")
    micros = Decimal(str(seconds)) * 1_000_000
    if micros != int(micros) or not micros:
        raise ValueError("cpu_limit_s must be an exact number of microseconds")
    return int(micros)


def solve_wall_limit_s(cfg, entry=None):
    return max(30, (3 * solve_limit_us(cfg, entry) + 999_999) // 1_000_000 + 10)


def batch_timeout_s(cfg):
    """Launcher must set both Harbor verifier timeout and bridge environment to this value."""
    reps = cfg.get("repeats", 3)
    if type(reps) is not int or not 1 <= reps <= 3 or not cfg["cases"]:
        raise ValueError("invalid repeat count or empty case list")
    # Include build, container handoff/startup, and host-side full-vector scoring.
    return 385 + reps * sum(solve_wall_limit_s(cfg, c) + 75 for c in cfg["cases"])


def _worker(mode, source, work, cfg, entry, repeat):
    handoff = work / "handoff"
    handoff.mkdir(mode=0o700)
    cpu = int(cfg.get("cpu", 2))
    cid = None
    command = ["--name", "galaxy-" + uuid.uuid4().hex, "--label", "withexpbench=galaxy-gravity",
               "--pull", "never", "--platform", "linux/arm64", "--network", "none", "--read-only",
               "--cpuset-cpus", str(cpu), "--cpus", "1", "--memory", str(MEMORY), "--memory-swap", str(MEMORY),
               "--pids-limit", "2" if mode == "run" else "128", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--ulimit", "core=0:0",
               "--tmpfs", "/tmp:rw,nosuid,nodev,mode=1777",
               "--tmpfs", "/output:rw,nosuid,nodev,uid=1000,gid=1000,mode=0700",
               "--user", "0:0"]
    for cap in ("SETUID", "SETGID", "KILL", "DAC_OVERRIDE", "CHOWN"):
        command += ["--cap-add", cap]
    command += bind(HERE / "worker.py", "/trusted/worker.py")
    command += bind(source, "/workspace/submission") + bind(handoff, "/handoff", True)
    if mode == "run":
        # Hide private filenames and prevent the guest seeing reference files or the manifest.
        snapshot = work / "snapshot.bin"
        shutil.copyfile(entry["input"], snapshot)
        snapshot.chmod(0o444)
        if digest(snapshot) != entry["input_sha256"]:
            raise RuntimeError("staged input hash mismatch")
        command += bind(snapshot, "/input/snapshot.bin")
    else:
        command += ["--tmpfs", "/build:rw,exec,nosuid,nodev,size=2g,mode=0755"]
    try:
        cid = docker("create", *command, "--entrypoint", "/usr/bin/python3", cfg["image"],
                     "-IB", "/trusted/worker.py", mode, cpu, repeat, solve_limit_us(cfg, entry))
        docker("start", cid)
        docker("wait", cid, timeout=solve_wall_limit_s(cfg, entry) + 15 if mode == "run" else 325)
        state = json.loads(docker("inspect", "--format", "{{json .State}}", cid))
        logs = docker("logs", cid)
        if state["OOMKilled"]:
            rec = dict(status="oom", peak_bytes=MEMORY)
        elif state["ExitCode"] or state["Error"]:
            raise RuntimeError(f"worker failed {state}: {logs[-1000:]}")
        else:
            rec = json.loads(logs)
        return rec
    finally:
        if cid:
            docker("rm", "-fv", cid)


def execute(mode, source, target, cfg, entry=None, repeat=0):
    target.mkdir(parents=True, exist_ok=False)
    # Random mount roots prevent /proc/self/mountinfo from revealing private case IDs.
    with tempfile.TemporaryDirectory(prefix="galaxy-worker-") as path:
        work = Path(path)
        rec = _worker(mode, source, work, cfg, entry, repeat)
        if rec["status"] == "ok":
            shutil.move(str(work / "handoff"), str(target / "handoff"))
        return rec


def error(path, truth):
    pred = path.read_bytes()
    labels = truth.read_bytes()
    if not labels or len(labels) % 24:
        raise RuntimeError("malformed reference")
    if len(pred) != len(labels):
        raise ValueError("invalid output length")
    ratios = []
    for a,b in zip(struct.iter_unpack("<3d", pred), struct.iter_unpack("<3d", labels)):
        if not all(map(math.isfinite, b)) or not math.hypot(*b):
            raise RuntimeError("nonfinite or zero reference")
        if not all(map(math.isfinite, a)):
            raise ValueError("nonfinite prediction")
        ratios.append(math.dist(a,b)/math.hypot(*b))
    rms = math.hypot(*ratios)/math.sqrt(len(ratios))
    if math.isfinite(rms):
        return rms
    # Extreme but finite submissions still have a well-defined valid error.
    with localcontext() as ctx:
        ctx.prec = 70
        total = Decimal(0)
        for a,b in zip(struct.iter_unpack("<3d", pred), struct.iter_unpack("<3d", labels)):
            x = list(map(Decimal,a))
            y = list(map(Decimal,b))
            total += sum((v-w)**2 for v,w in zip(x,y))/sum(w*w for w in y)
        return (total/len(ratios)).sqrt()


def encode_error(value):
    if isinstance(value, Decimal):
        return float(value) if value <= sys.float_info.max else str(value)
    return value


def source_check(src):
    if not src.is_dir() or src.is_symlink():
        raise ValueError("missing source directory")
    for p in src.rglob("*"):
        st = p.lstat()
        if not stat.S_ISDIR(st.st_mode) and (not stat.S_ISREG(st.st_mode) or st.st_nlink != 1):
            raise ValueError("links/special files in submission")


def validate(cfg):
    solve_limit_us(cfg)
    cases = cfg["cases"]
    ids = [c["case"] for c in cases]
    mode = cfg["mode"]
    reps = cfg.get("repeats", 3)
    if mode == "formal":
        raise ValueError("B.4 formal scores require approved weights and anchors; use precalibration for raw results")
    if (mode not in RAW_MODES or type(reps) is not int or not 1 <= reps <= 3
            or not 0 < len(cases) <= 80 or not re.fullmatch(r"sha256:[0-9a-f]{64}", cfg["image"])
            or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", k) for k in ids)):
        raise ValueError("invalid grader manifest")
    if "anchors" in cfg or cfg.get("weights") is not None or cfg.get("score_weights") is not None:
        raise ValueError("B.4 raw evaluation does not accept scoring anchors or weights")
    counts = defaultdict(int)
    for c in cases:
        solve_limit_us(cfg, c)
        if mode in ("pilot", "precalibration", "calibration") and "cpu_limit_s" not in c:
            raise ValueError("frozen per-case cpu_limit_s is required outside timing/diagnostic mode")
        for kind in ("input", "truth"):
            if digest(Path(c[kind])) != c[kind+"_sha256"]:
                raise ValueError("frozen " + kind + " hash mismatch")
        size = Path(c["input"]).stat().st_size
        if not size or size % 32 or Path(c["truth"]).stat().st_size != size//32*24:
            raise ValueError("invalid frozen snapshot dimensions")
        n = size // 32
        if "n" in c and (type(c["n"]) is not int or c["n"] != n):
            raise ValueError("manifest n disagrees with input dimensions")
        if mode != "diagnostic" and (n not in SCALES or c["family"] not in FAMILIES):
            raise ValueError("wrong particle count or family")
        counts[(n, c["family"])] += 1
    if mode != "diagnostic":
        expected = {(n, f): 1 if mode == "pilot" else 4 for n in SCALES for f in FAMILIES}
        if dict(counts) != expected or reps != 3:
            raise ValueError("expected complete four-scale/five-family grid and three repeats")
    return cases, reps


def summarize(records, cases, reps):
    """Unweighted descriptive strata; deliberately no overall raw error or score."""
    by_case = {}
    for entry in cases:
        rows = [r for r in records if r["case"] == entry["case"]]
        valid = [r for r in rows if r["status"] == "ok"]
        complete = len(rows) == reps and len(valid) == reps
        n = entry.get("n", Path(entry["input"]).stat().st_size // 32)
        cpu = [r["cpu_s"] for r in rows if "cpu_s" in r]
        by_case[entry["case"]] = dict(n=n, family=entry["family"], expected_calls=reps,
            attempted_calls=len(rows), valid_calls=len(valid), complete=complete,
            statuses={s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
            raw_error=encode_error(sum(Decimal(str(r["error"])) for r in valid)/reps) if complete else None,
            cpu_s={"min": min(cpu), "mean": sum(cpu)/len(cpu), "max": max(cpu)} if cpu else None,
            peak_bytes=max((r.get("peak_bytes", 0) for r in rows), default=0))
    def group(rows):
        finished = [r for r in rows if r["complete"]]
        values = [Decimal(str(r["raw_error"])) for r in finished]
        mean = encode_error(sum(values)/len(values)) if values else None
        return dict(expected_cases=len(rows), complete_cases=len(finished),
                    expected_calls=sum(r["expected_calls"] for r in rows),
                    attempted_calls=sum(r["attempted_calls"] for r in rows),
                    valid_calls=sum(r["valid_calls"] for r in rows),
                    complete=bool(rows) and len(finished)==len(rows),
                    raw_error=mean if len(finished)==len(rows) else None,
                    complete_case_error_mean=mean)
    values = list(by_case.values())
    return dict(by_case=by_case,
        by_scale={str(n): group([r for r in values if r["n"]==n]) for n in sorted({r["n"] for r in values})},
        by_family={f: group([r for r in values if r["family"]==f]) for f in sorted({r["family"] for r in values})},
        by_scale_family={str(n): {f: group([r for r in values if r["n"]==n and r["family"]==f])
            for f in sorted({r["family"] for r in values if r["n"]==n})} for n in sorted({r["n"] for r in values})},
        note="Stratum means are descriptive only. Incomplete strata have no valid raw_error; complete_case_error_mean excludes failed cases. No overall weighting or project score is applied.")


def tree_digest(folder):
    return {str(p.relative_to(folder)): [digest(p), bool(p.stat().st_mode & 0o111)]
            for p in sorted(folder.rglob("*")) if p.is_file()}


def grade(src, out, cfg):
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="infrastructure_failure", interface_pass=None, raw_error=None,
                  project_score=None, records=[])
    validated = None
    try:
        cases, reps = validate(cfg)
        validated = (cases, reps)
        result["batch_timeout_s"] = batch_timeout_s(cfg)
        try:
            source_check(src)
        except ValueError as e:
            result.update(status="submission_failure", interface_pass=0., reason=str(e))
            return result
        resume = None
        recovery = os.environ.get("GALAXY_CALIBRATION_RECOVERY")
        if cfg["mode"] in RECOVERY_MODES:
            identity = dict(config=cfg, source=tree_digest(src), grader=digest(HERE/"grade.py"), worker=digest(HERE/"worker.py"))
            write_json(out/"identity.json", identity)
            if recovery:
                resume=Path(recovery)
                if (json.loads((resume/"identity.json").read_text()) != identity or
                        (resume/"summary.json").exists() and json.loads((resume/"summary.json").read_text())["status"] != "infrastructure_failure"):
                    raise RuntimeError("recovery source must be the same interrupted evaluation attempt")
                result["recovery_from"] = str(resume)
        elif recovery:
            raise RuntimeError("only timing/precalibration infrastructure interruptions can use archived records")
        if resume and (resume/"built-files.json").is_file():
            built = resume/"build/handoff/built"
            if tree_digest(built)!=json.loads((resume/"built-files.json").read_text()):
                raise RuntimeError("recovery build digest mismatch")
            shutil.copytree(resume/"build",out/"build")
            build=json.loads((out/"build/run.json").read_text())
        else:
            build = execute("build", src, out / "build", cfg)
        result["build"] = build
        write_json(out / "build" / "run.json", build)
        if build["status"] != "ok":
            result.update(status="submission_failure", interface_pass=0., reason="build "+build["status"])
            return result
        built = out / "build/handoff/built"
        if cfg["mode"] in RECOVERY_MODES:
            write_json(out/"built-files.json", tree_digest(built))
        for entry in cases:
            for repeat in range(reps):
                d = out / "rounds" / entry["case"] / str(repeat)
                old = resume/"rounds"/entry["case"]/str(repeat) if resume else None
                micros = solve_limit_us(cfg, entry)
                n = entry.get("n", Path(entry["input"]).stat().st_size // 32)
                if old and old.exists() and not (old/"run.json").is_file():
                    raise RuntimeError("partial archived invocation requires review; refusing to rerun " + str(old))
                if old and (old/"run.json").exists():
                    rec=json.loads((old/"run.json").read_text())
                    if ((rec["case"],rec["family"],rec["repeat"],rec["n"]) != (entry["case"],entry["family"],repeat,n)
                            or rec["cpu_limit_s"] != micros/1e6
                            or any(entry[k]!=rec[k] for k in ("input_sha256","truth_sha256"))
                            or rec["status"] not in ("ok","entry_error","cpu_timeout","wall_timeout","nonzero_exit","process_limit","oom","invalid_output")):
                        raise RuntimeError("recovery cannot discard failed or altered archived calls")
                    if rec["status"]=="ok" and (rec["cpu_us"]>micros or "error" not in rec):
                        raise RuntimeError("inconsistent successful recovery record")
                    if "output_sha256" in rec and rec["output_sha256"]!=digest(old/"handoff/output.bin"):
                        raise RuntimeError("recovery output digest mismatch")
                    shutil.copytree(old,d)
                else:
                    rec = execute("run", built, d, cfg, entry, repeat)
                    rec.update(case=entry["case"], family=entry["family"], n=n, repeat=repeat,
                               cpu_limit_s=micros/1e6, input_sha256=entry["input_sha256"], truth_sha256=entry["truth_sha256"])
                    if rec["status"] == "ok":
                        path = d / "handoff/output.bin"
                        rec["output_sha256"] = digest(path)
                        try:
                            rec["error"] = encode_error(error(path, Path(entry["truth"])))
                        except ValueError as exc:
                            rec.update(status="invalid_output", detail=str(exc))
                result["records"].append(rec)
                write_json(d / "run.json", rec)
        failed = any(r["status"] != "ok" for r in result["records"])
        result.update(status="submission_failure" if failed else "ok", interface_pass=0. if failed else 1.)
    except Exception as exc:
        result.update(status="infrastructure_failure", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if validated is not None:
            result["strata"] = summarize(result["records"], *validated)
        write_json(out/"summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("submission", "output", "config"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    summary = grade(args.submission, args.output, json.loads(args.config.read_text()))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("records","build")}))
    sys.exit(2 if summary["status"] == "infrastructure_failure" else 0)
