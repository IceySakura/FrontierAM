"""Repository task adapter for the sparse QP Harbor verifier."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
import tempfile

from ..operator_tasks import OperatorTask
from ..qp_runtime import reference_source, run_verifier
from ..tasks import GIB, tar_directory


class PDHCGQP(OperatorTask):
    grade_memory = 4 * GIB
    grade_timeout = 14400

    def reference_archive(self, runtime, output):
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            source = reference_source(Path(temporary) / "reference")
            return tar_directory(source, output / "incoming.tar")

    async def grade(self, runtime, placement, output):
        split = os.environ.get("FRONTIERAM_QP_DATA", "private")
        if split not in ("public", "private"):
            raise ValueError("FRONTIERAM_QP_DATA must be public or private")
        try:
            summary = run_verifier(output / "source", self.sealed / "data" / split, output / "candidate",
                                   image=self.lock()["image"]["id"], docker_host=placement.docker_host,
                                   cpuset=placement.cpuset, cgroup_parent=placement.cgroup_parent)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return {"exit_status": "infrastructure_failure", "reason": str(exc)}
        return {"exit_status": "finished", "summary": summary, "split": split}

    def metrics(self, output, raw):
        return {"raw": raw, "summary": raw.get("summary"), "split": raw.get("split")}

    def baseline_ok(self, metrics):
        summary = metrics.get("summary") or {}
        ok = summary.get("valid") is True and summary.get("p") == 1 and bool(summary.get("g"))
        return ok, "reference passed every case" if ok else "reference must pass all cases in all three repeats"

    def score(self, runtime, metrics, baseline):
        if metrics["raw"].get("exit_status") != "finished":
            return {"status": "unmeasured", "score": None, "reason": metrics["raw"].get("reason")}
        summary = metrics["summary"]
        if not summary["valid"]:
            return {"status": "unmeasured", "score": None, "reason": "invalid assessment"}
        if summary["build"]["outcome"] != "completed":
            return {"status": "invalid_submission", "score": 0.0, "reason": summary["build"].get("reason")}
        if baseline is None:
            return {"status": "baseline_required", "score": None}
        reference = baseline["metrics"]
        if metrics["split"] != reference["split"] or not self.baseline_ok(reference)[0]:
            return {"status": "unmeasured", "score": None, "reason": "baseline split mismatch or invalid reference"}
        g_ref = reference["summary"]["g"]
        return {"status": "ok", "score": 100.0 * summary["p"] / (1.0 + summary["g"] / g_ref),
                "p": summary["p"], "g": summary["g"], "g_ref": g_ref,
                "mapping": "100*p/(1+g/g_ref)"}
