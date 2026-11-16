"""Trusted host build and per-scene execution. No private references enter Docker."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import uuid

import numpy as np


BASE_IMAGE = "sha256:80b3867f1168dd170b4e7a8a06bb0fe98cb59b38116016641fee7a61fe823b42"
SCENE_LIMIT_S = 1
WORKER = Path(__file__).resolve().with_name("worker.py")
THREAD_ENV = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS")


class SubmissionError(ValueError):
    pass


def digest(path):
    with Path(path).open("rb") as stream:
        value = hashlib.sha256()
        while chunk := stream.read(1 << 20):
            value.update(chunk)
    return value.hexdigest()


def manifest(source):
    """Both source handoff and finished build are quiescent when inspected."""
    if source.is_symlink() or not source.is_dir():
        raise SubmissionError("invalid submission directory")
    files = {}
    for path in sorted(source.rglob("*")):
        st = path.lstat()
        if stat.S_ISREG(st.st_mode) and st.st_nlink == 1:
            files[path.relative_to(source).as_posix()] = digest(path)
        elif not stat.S_ISDIR(st.st_mode):
            raise SubmissionError(f"unsupported source entry: {path.relative_to(source)}")
    if not {"solution.py", "README.md"} <= files.keys():
        raise SubmissionError("solution.py and README.md are required")
    if "build.sh" in files and not (source / "build.sh").stat().st_mode & 0o111:
        raise SubmissionError("build.sh is not executable")
    return files


def docker(*args, timeout=45):
    proc = subprocess.run(["docker", *args], capture_output=True, timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(f"docker {args[0]}: {proc.stderr.decode(errors='replace')[-2000:]}")
    return proc.stdout.decode().strip()


def mount(path, dest, readonly=True):
    path = Path(path).resolve(strict=True)
    if "," in str(path):
        raise ValueError("CSV delimiter in mount path")
    return ["--mount", f"type=bind,src={path},dst={dest}" + (",readonly" if readonly else "")]


def sandbox(mode, source, timeout, extra=(), *, adapter=WORKER):
    if not 0 < timeout <= (300 if mode == "build" else SCENE_LIMIT_S):
        raise ValueError("out-of-contract wall time limit")
    uid = "polar-worker-" + uuid.uuid4().hex
    cmd = ["create", "--name", uid, "--platform", "linux/arm64", "--pull", "never",
           "--network", "none", "--read-only", "--cpus", "1", "--cpuset-cpus", "0",
           "--memory", "8g", "--memory-swap", "8g", "--pids-limit", "256",
           "--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID",
           "--cap-add", "KILL", "--security-opt", "no-new-privileges:true",
           "--user", "0:0", "--workdir", "/", "--env", "SEED=1", "--env", "HOME=/tmp",
           "--tmpfs", "/tmp:rw,nosuid,nodev,mode=1777,size=8g",
           "--tmpfs", "/workspace:rw,nosuid,nodev,mode=0755",
           "--tmpfs", "/home:rw,nosuid,nodev,mode=0755",
           "--log-driver", "json-file", "--log-opt", "max-size=1m", "--log-opt", "max-file=2"]
    for key in THREAD_ENV:
        cmd += ["--env", key + "=1"]
    cmd += mount(adapter, "/adapter/worker.py")
    cmd += mount(source, "/workspace/submission", readonly=mode != "build")
    cmd += list(extra) + ["--entrypoint", "/usr/local/bin/python", BASE_IMAGE,
                           "-I", "-B", "/adapter/worker.py", mode, str(timeout)]
    info = dict(status="infrastructure_failure", reason="container did not start",
                container=uid, image=BASE_IMAGE, limit_s=timeout, wall_s=None, peak_bytes=None, cleanup_ok=False)
    created = False
    try:
        cid = docker(*cmd)
        created = True
        docker("start", cid)
        docker("wait", cid, timeout=timeout+15)
        state = json.loads(docker("inspect", "--format", "{{json .State}}", cid))
        info.update(exit_code=state["ExitCode"], docker_oom=state["OOMKilled"])
        if state["Error"]:
            raise RuntimeError(state["Error"])
        if state["OOMKilled"]:
            info.update(status="submission_failure", reason="oom")
        elif state["ExitCode"] != 0:
            raise RuntimeError("supervisor exit: " + docker("logs", cid)[-1500:])
        else:
            worker = json.loads(docker("logs", cid))
            if worker["status"] not in ("ok", "submission_failure"):
                raise RuntimeError("unexpected worker status")
            info.update(worker)
    except Exception as exc:
        info.update(status="infrastructure_failure", reason=str(exc))
    finally:
        if created:
            try:
                docker("rm", "--force", "--volumes", cid)
                if docker("ps", "-aq", "--filter", "id=" + cid):
                    raise RuntimeError("owned container still exists")
                info["cleanup_ok"] = True
            except Exception as exc:
                info.update(status="infrastructure_failure", reason=f"cleanup failed: {exc}")
    return info


def save(path, record):
    temp = path.with_suffix(".part")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
    return record


def build_submission(source_dir, build_dir, *, timeout_s=300):
    """Capture once from a stopped development environment; never execute on host."""
    source, target = Path(source_dir), Path(build_dir)
    target.mkdir(parents=True, exist_ok=False)
    info = dict(status="infrastructure_failure", reason="not built")
    try:
        before = manifest(source)
        built = target / "submission"
        with tempfile.TemporaryDirectory(prefix="build-", dir="/tmp") as directory:
            temp = Path(directory)
            stage = temp / "source"
            shutil.copytree(source, stage, symlinks=True)
            if manifest(stage) != before:
                raise RuntimeError("source changed while capturing")
            # On a bind mount the host owner need not be the container builder
            # UID. Only this scratch build tree is ever mounted writable.
            for p in [stage, *stage.rglob("*")]:
                m = p.stat().st_mode
                p.chmod(0o777 if p.is_dir() or m & 0o111 else 0o666)
            adapter = temp / "worker.py"
            shutil.copyfile(WORKER, adapter)
            info = sandbox("build", stage, timeout_s, adapter=adapter)
            info["source_files"] = before
            if info["status"] == "ok":
                files = manifest(stage)
                shutil.copytree(stage, built, symlinks=True)
                if manifest(built) != files:
                    raise RuntimeError("built files changed while freezing")
                for p in [built, *built.rglob("*")]:
                    p.chmod(0o755 if p.is_dir() or p.stat().st_mode & 0o111 else 0o644)
                info["built_files"] = files
    except SubmissionError as exc:
        info.update(status="submission_failure", reason=str(exc))
    except Exception as exc:
        info.update(status="infrastructure_failure", reason=str(exc))
    return save(target / "build.json", info)


def read_npy(path, *, raw=False, shape=None):
    """Validate a bounded numeric header and file length before mapping any values."""
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("NPY must be a regular file")
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0) and version != (2, 0):
            raise ValueError("unsupported NPY version")
        header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        length_width = 2 if version == (1, 0) else 4
        head = stream.tell()
        prefix = stream.read(length_width)
        if len(prefix) != length_width or int.from_bytes(prefix, "little") > 2048:
            raise ValueError("oversize or truncated NPY header")
        stream.seek(head)
        dims, _, dtype = header(stream, max_header_size=2048)
        if raw:
            good = (len(dims) == 2 and all(0 < n <= 1024 for n in dims)
                    and dtype == np.dtype("uint8"))
        else:
            good = dims == (4, *shape) and dtype.kind == "f" and dtype.itemsize in (4, 8)
        if not good:
            raise ValueError("wrong NPY shape or dtype")
        if path.stat().st_size != stream.tell() + int(np.prod(dims)) * dtype.itemsize:
            raise ValueError("wrong NPY byte count")
    data = np.load(path, allow_pickle=False, mmap_mode="r")
    if not raw and not all(np.isfinite(plane).all() for plane in data):
        raise ValueError("nonfinite prediction")
    return data


def run_scene(build_dir, raw_path, result_dir, *, timeout_s=SCENE_LIMIT_S):
    built, raw, out = Path(build_dir), Path(raw_path), Path(result_dir)
    out.mkdir(parents=True, exist_ok=False)
    info = dict(status="infrastructure_failure", reason="not launched")
    pred = out / "pred.npy"
    try:
        receipt = json.loads((built / "build.json").read_text())
        if receipt["status"] != "ok":
            raise RuntimeError("build incomplete")
        try:
            frozen = manifest(built / "submission")
        except SubmissionError as exc:
            raise RuntimeError("frozen build was modified") from exc
        if frozen != receipt["built_files"]:
            raise RuntimeError("frozen build was modified")
        shape = read_npy(raw, raw=True).shape
        raw_sha = digest(raw)
        # Linux mountinfo can reveal host bind-mount source paths. Stage four
        # individual mounts outside the project, with random parent and generic
        # leaf names, before the timed call. Never mount the staging directory.
        with tempfile.TemporaryDirectory(prefix="round-", dir="/tmp") as directory:
            temp = Path(directory)
            program = temp / "source"
            shutil.copytree(built / "submission", program, symlinks=True)
            if manifest(program) != receipt["built_files"]:
                raise RuntimeError("staged program differs from the frozen build")
            sensor = temp / "sensor.npy"
            shutil.copyfile(raw, sensor)
            if digest(sensor) != raw_sha:
                raise RuntimeError("staged input differs from original")
            sensor.chmod(0o444)
            adapter = temp / "worker.py"
            shutil.copyfile(WORKER, adapter)
            exported = temp / "result.npy"
            exported.touch(exist_ok=False)
            exported.chmod(0o666)
            extra = ["--tmpfs", "/output:rw,nosuid,nodev,mode=0755,size=40m",
                     "--tmpfs", "/handoff:rw,nosuid,nodev,mode=0700"]
            extra += mount(sensor, "/input/raw.npy") + mount(exported, "/handoff/pred.npy", readonly=False)
            info = sandbox("run", program, timeout_s, extra, adapter=adapter)
            info.update(input_sha256=raw_sha)
            if info["status"] == "ok":
                shutil.copyfile(exported, pred)
                pred.chmod(0o600)
        if info["status"] == "ok":
            try:
                read_npy(pred, shape=shape)
            except (ValueError, EOFError) as exc:
                raise SubmissionError("invalid numeric output: " + str(exc)) from exc
            info["output_sha256"] = digest(pred)
    except SubmissionError as exc:
        info.update(status="submission_failure", reason=str(exc))
    except Exception as exc:
        info.update(status="infrastructure_failure", reason=str(exc))
    finally:
        if pred.exists() and info["status"] != "ok":
            pred.unlink()
    return save(out / "run.json", info)
