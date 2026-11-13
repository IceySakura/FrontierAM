"""Harbor's separate verifier forwards a restored artifact to one trusted host grader."""

import json
import os
from pathlib import Path
import re
import shutil
import stat
import time


MAIL = Path("/run/galaxy-gravity-handoff")
SOURCE = Path("/workspace/submission")
LOG = Path("/logs/verifier")
CG = Path("/sys/fs/cgroup")


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False) + "\n")
    tmp.replace(path)


def cgroup():
    return {name: (CG / name).read_text().strip()
            for name in ("cpu.stat", "cpu.max", "cpuset.cpus.effective", "memory.max",
                         "memory.swap.max", "memory.peak")}


def stage(destination, owner):
    if not SOURCE.is_dir() or SOURCE.is_symlink():
        return "Missing regular submission directory"
    destination.mkdir(mode=0o700)
    os.chown(destination, *owner)
    for parent, dirs, files in os.walk(SOURCE):
        target = destination / Path(parent).relative_to(SOURCE)
        for name in dirs:
            if (Path(parent) / name).is_symlink():
                return "Linked directory in submitted artifact"
            folder = target / name
            folder.mkdir(mode=0o700)
            os.chown(folder, *owner)
        for name in files:
            src = Path(parent) / name
            st = src.lstat()
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                return "Linked or special file in submitted artifact"
            new = target / name
            shutil.copyfile(src, new, follow_symlinks=False)
            new.chmod(0o755 if st.st_mode & 0o111 else 0o644)
            os.chown(new, *owner)


def main():
    key = os.environ.get("GALAXY_HANDOFF_TOKEN", "")
    if not re.fullmatch("[0-9a-f]{64}", key):
        raise RuntimeError("Missing trusted trial transport. Use the operator launcher.")
    st = MAIL.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_mode & 0o077:
        raise RuntimeError("The mailbox must be an operator-private directory")
    owner = st.st_uid, st.st_gid
    start = cgroup()
    if (start["cpuset.cpus.effective"] != "2" or start["cpu.max"] != "100000 100000"
            or start["memory.max"] != str(2 * 2**30) or start["memory.swap.max"] != "0"
            or len(Path("/proc/net/route").read_text().splitlines()) != 1
            or any(line.split()[-1] != "lo" for line in Path("/proc/net/if_inet6").read_text().splitlines())):
        raise RuntimeError("Unexpected verifier resources or networking")
    box = MAIL / key
    box.mkdir(mode=0o700)
    os.chown(box, *owner)
    issue = stage(box / "source", owner)
    save(box / "request.json", {"submission_error": issue})
    response = box / "response.json"
    seconds = os.environ.get("GALAXY_VERIFIER_TIMEOUT_S", "460000")
    if not seconds.isdigit() or not 1 <= int(seconds) <= 460000:
        raise RuntimeError("Invalid trusted verifier timeout")
    deadline = time.monotonic() + int(seconds)
    while not response.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Trusted host grader did not respond within the batch budget")
        time.sleep(1)
    result = json.loads(response.read_text())
    save(LOG / "bridge-cgroup.json", {"start": start, "end": cgroup()})
    if "infra_error" in result:
        raise RuntimeError(result["infra_error"])
    value = result["interface_pass"]
    if type(value) not in (int, float) or value not in (0, 1):
        raise RuntimeError("Invalid verifier response")
    save(LOG / "receipt.json", result)
    score = result["project_score"]
    if score is None:
        save(LOG / "reward.json", {"interface_pass": float(value)})
    elif type(score) in (int, float) and 0 <= score <= 10:
        save(LOG / "reward.json", {"reward": score / 10})
    else:
        raise RuntimeError("Invalid calibrated score")


if __name__ == "__main__":
    main()
