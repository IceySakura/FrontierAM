"""Container and reference staging shared by QP sealing and the bench adapter."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
VERSION = ROOT / "tasks/pdhcg-qp/versions/v1"


def reference_source(destination: Path) -> Path:
    shutil.copytree(ROOT / "tasks/pdhcg-qp/references/r1", destination)
    package = destination / "qpbench"
    package.mkdir()
    for name in ("__init__.py", "model.py", "check.py"):
        shutil.copy2(VERSION / "grader/qpbench" / name, package / name)
    return destination


def run_verifier(source: Path, data: Path, output: Path, *, image: str, docker_host: str | None = None,
                 cpuset: str | None = None, cgroup_parent: str | None = None, calibrate=False) -> dict:
    """Run one disposable, offline, single-CPU verifier. Always remove it on timeout."""
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    name = "frontieram-qp-" + uuid.uuid4().hex[:12]
    docker = ["docker"] + (["--host", docker_host] if docker_host else [])
    command = [*docker, "run", "--rm", "--name", name, "--network", "none", "--cpus", "1",
               "--memory", "4g", "--memory-swap", "4g", "--pids-limit", "256",
               "--security-opt", "no-new-privileges:true", "--cap-drop", "ALL",
               "--cap-add", "SETUID", "--cap-add", "SETGID", "--cap-add", "CHOWN",
               "--cap-add", "FOWNER", "--cap-add", "DAC_OVERRIDE",
               "--user", "0:0", "--log-driver", "none"]
    if cpuset:
        command += ["--cpuset-cpus", cpuset]
    if cgroup_parent:
        command += ["--cgroup-parent", cgroup_parent]
    # Copy trusted code/data into root-only directories; host bind mount modes
    # must not make the full test pool readable to the candidate UID.
    script = ("mkdir -p /tests /private && chmod 700 /tests /private /logs/verifier && "
              "chmod 755 /tmp /var/tmp /dev/shm && "
              "cp -r /root/grader/. /tests/ && cp -r /root/data/. /private/ && "
              "exec python3 -B /tests/grade.py --submission /submission "
              "--manifest /private/manifest.json --calibration "
              + ("/logs/verifier/calibration.json --calibrate" if calibrate else "/private/calibration.json"))
    command += ["-v", f"{VERSION / 'grader'}:/root/grader:ro",
                "-v", f"{source.resolve()}:/submission:ro", "-v", f"{data.resolve()}:/root/data:ro",
                "-v", f"{output.resolve()}:/logs/verifier", "--entrypoint", "sh", image, "-c", script]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=14400)
        (output / "container.log").write_text(done.stdout + done.stderr)
        if done.returncode:
            raise RuntimeError(f"QP verifier exited {done.returncode}; see {output / 'container.log'}")
        summary = json.loads((output / "summary.json").read_text())
        if calibrate:
            anchor = json.loads((output / "calibration.json").read_text())
            (data / "calibration.json").write_text(json.dumps(anchor, indent=2) + "\n")
        return summary
    finally:
        subprocess.run([*docker, "rm", "-f", name], capture_output=True, timeout=30)
