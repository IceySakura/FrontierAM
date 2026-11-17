"""Harbor verifier entry: evaluate a frozen submission or calibrate the local reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qpbench.evaluate import evaluate, read_manifest
from qpbench.scoring import compute_reward


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def calibration(summary: dict, manifest_path: Path) -> dict:
    if not summary["valid"] or summary["p"] != 1.0:
        raise ValueError("reference must pass every case in all three repeats before calibration")
    limits = {row["case_id"]: min(60.0, max(1.0, 1.5 * max(
        r["elapsed_seconds"] for r in row["repeats"]))) for row in summary["raw_cases"]}
    if any(max(r["elapsed_seconds"] for r in row["repeats"]) > limits[row["case_id"]] / 1.25
           for row in summary["raw_cases"]):
        raise ValueError("reference has less than 25% deadline headroom")
    return {"schema_version": 1, "status": "calibrated", "p_ref": 1.0, "g_ref": summary["g"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "time_limits": limits, "policy": "1.5x max of three runs, clipped to [1,60]",
            "reference_summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("/workspace/submission"))
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "data/manifest.json")
    parser.add_argument("--calibration", type=Path, default=Path(__file__).parent / "data/calibration.json")
    parser.add_argument("--output", type=Path, default=Path("/logs/verifier/summary.json"))
    parser.add_argument("--reward", type=Path, default=Path("/logs/verifier/reward.json"))
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args(argv)
    # Failures in trusted inputs/configuration deliberately propagate: no reward
    # is written, so Harbor reports an infrastructure failure rather than zero.
    args.reward.unlink(missing_ok=True)
    manifest = read_manifest(args.manifest)
    if args.calibrate:
        summary = evaluate(args.submission, args.manifest, {r["id"]: 60.0 for r in manifest["cases"]})
        write(args.output, summary)
        write(args.calibration, calibration(summary, args.manifest))
        return 0
    anchor = json.loads(args.calibration.read_text())
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if anchor.get("status") != "calibrated" or anchor.get("manifest_sha256") != digest:
        raise ValueError("missing or mismatched local calibration; calibrate this exact split first")
    # Validate the anchor before executing any untrusted code.
    compute_reward({"valid": True, "p": 1.0, "g": anchor["g_ref"]}, anchor)
    summary = evaluate(args.submission, args.manifest, anchor["time_limits"])
    write(args.output, summary)
    if not summary["valid"]:
        raise ValueError("invalid assessment: " + str(summary.get("errors")))
    if summary["build"]["outcome"] != "completed":
        score = {"reward": 0.0, "score": 0.0}
    else:
        score = compute_reward(summary, anchor)
    summary.update(score=score, mapping="100*p/(1+g/g_ref)")
    write(args.output, summary)
    write(args.reward, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
