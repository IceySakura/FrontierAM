"""Create a maintainer-only, post-hoc balanced split of the frozen 45 cases.

Only Cold CG and Warm CG result files are opened.  Other solver results never
enter the allocation objective.  This is conditional random allocation of
already-observed data, not the creation of an unseen holdout.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys


REVISION = "balanced-split-20261117"
METHODS = ("cold_cg", "warm_cg")
DEFAULT_ARCHIVE_SHA256 = "acfebb847feacbf0294a7df948473e07dc6c407c00eddabf7d193c66cd352a3f"
RESULT_NAME = re.compile(r"((public|private)_\d+)__(cold_cg|warm_cg)\.json")
SHA256 = re.compile(r"[0-9a-f]{64}")


class NoFeasibleSplit(ValueError):
    """Random search failed; this does not prove mathematical infeasibility."""

    def __init__(self, diagnostics):
        super().__init__(
            "no sampled split satisfies both CG difficulty targets; "
            "this finite random search does not prove mathematical infeasibility"
        )
        self.diagnostics = diagnostics


def _number(value, name, lower=0.0, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected a number")
    value = float(value)
    if not math.isfinite(value) or value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name}: non-finite or outside allowed range")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def collect_baselines(roots):
    """Validate the two baselines and return 45 identity-preserving case rows."""
    rows = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"result directory does not exist: {root}")
        for path in sorted(root.rglob("*.json")):
            match = RESULT_NAME.fullmatch(path.name)
            if match is None:
                continue  # In particular, do not open any agent result JSON.
            expected_case, expected_split, expected_method = match.groups()
            raw = path.read_bytes()
            row = json.loads(raw)
            if (row.get("case"), row.get("split"), row.get("method")) != (
                expected_case, expected_split, expected_method
            ):
                raise ValueError(f"result filename/identity mismatch: {path}")
            key = (expected_case, expected_method)
            if key in rows:
                raise ValueError(f"duplicate baseline record: {key}")
            if row.get("dimension") != 16384 or row.get("steps") != 50:
                raise ValueError(f"wrong dimension or steps: {path}")
            if row.get("valid") is not True or row.get("resource_compliant") is not True:
                raise ValueError(f"invalid baseline result: {path}")
            if row.get("completed_steps") != 50:
                raise ValueError(f"incomplete baseline result: {path}")
            if row.get("termination_reason") is not None or row.get("solver_errors"):
                raise ValueError(f"baseline reports solver errors: {path}")
            _number(row.get("wall_seconds"), "wall_seconds", upper=30.0)
            _number(row.get("peak_rss_mib"), "peak_rss_mib", upper=256.0)
            loss = _number(row.get("mean_energy_loss"), "mean_energy_loss")
            if not isinstance(row.get("data_id"), str) or not SHA256.fullmatch(row["data_id"]):
                raise ValueError(f"missing or invalid data_id SHA256: {path}")
            if any(not isinstance(row.get(k), str) or not row[k] for k in ("dataset", "profile")):
                raise ValueError(f"missing dataset/profile: {path}")
            records = row.get("records")
            if records is not None:
                if len(records) != 50 or [r.get("step") for r in records] != list(range(50)):
                    raise ValueError(f"wrong per-step record coverage: {path}")
                if any(r.get("completed") is not True or r.get("error") is not None for r in records):
                    raise ValueError(f"invalid per-step baseline result: {path}")
                step_losses = [_number(r.get("energy_loss"), "step energy_loss") for r in records]
                if not math.isclose(math.fsum(step_losses) / 50, loss, rel_tol=1e-10, abs_tol=1e-15):
                    raise ValueError(f"mean loss contradicts per-step losses: {path}")
            rows[key] = {
                "case": expected_case,
                "original_split": expected_split,
                "method": expected_method,
                "data_id": row["data_id"],
                "dataset": row["dataset"],
                "profile": row["profile"],
                "loss": loss,
                "record_sha256": hashlib.sha256(raw).hexdigest(),
            }

    coverage = {method: {case for case, m in rows if m == method} for method in METHODS}
    if any(len(coverage[m]) != 45 for m in METHODS) or coverage[METHODS[0]] != coverage[METHODS[1]]:
        raise ValueError("both CG baselines must cover exactly the same 45 cases")
    cases = []
    for case in sorted(coverage[METHODS[0]]):
        first = rows[(case, METHODS[0])]
        for method in METHODS[1:]:
            other = rows[(case, method)]
            for key in ("data_id", "dataset", "profile", "original_split"):
                if first[key] != other[key]:
                    raise ValueError(f"mixed data versions or case metadata: {case}, {key}")
        cases.append({
            **{key: first[key] for key in ("case", "data_id", "original_split", "dataset", "profile")},
            "losses": {m: rows[(case, m)]["loss"] for m in METHODS},
            "record_sha256": {m: rows[(case, m)]["record_sha256"] for m in METHODS},
        })
    _groups(cases)
    if {pool: sum(c["original_split"] == pool for c in cases) for pool in ("public", "private")} != {
        "public": 15, "private": 30
    }:
        raise ValueError("original split must contain 15 public and 30 private cases")
    for method in METHODS:
        if math.fsum(c["losses"][method] for c in cases) <= 0:
            raise ValueError(f"{method} has no positive difficulty signal")
    return cases


def _groups(cases):
    groups = {}
    for case in cases:
        groups.setdefault((case["dataset"], case["profile"]), []).append(case)
    if len(cases) != 45 or len(groups) != 15 or any(len(g) != 3 for g in groups.values()):
        raise ValueError("need 15 dataset/profile groups of exactly 3 cases each")
    if len({c["case"] for c in cases}) != 45:
        raise ValueError("duplicate case identity")
    return [sorted(groups[key], key=lambda c: c["case"]) for key in sorted(groups)]


def _ratio(a, b):
    if a == b == 0:
        return 1.0
    if min(a, b) == 0:
        return math.inf
    return max(a / b, b / a)


def _pool_stats(values):
    total = math.fsum(values)
    ordered = sorted(values, reverse=True)
    return {
        "count": len(values),
        "mean_loss": total / len(values),
        "median_loss": statistics.median(values),
        "top1_loss_share": ordered[0] / total if total else 0.0,
        "top2_loss_share": math.fsum(ordered[:2]) / total if total else 0.0,
    }


def _diagnose(cases, public_ids):
    methods = {}
    for method in METHODS:
        public = _pool_stats([c["losses"][method] for c in cases if c["case"] in public_ids])
        private = _pool_stats([c["losses"][method] for c in cases if c["case"] not in public_ids])
        a, b = public["mean_loss"], private["mean_loss"]
        ratio = _ratio(a, b)
        methods[method] = {
            "public": public,
            "private": private,
            "public_private_mean_ratio": a / b if b else None,
            "larger_to_smaller_mean_ratio": ratio if math.isfinite(ratio) else None,
        }
    return methods


def rebalance(cases, *, seed=20261117, candidates=100000, max_ratio=1.25,
              source_archive_sha256=DEFAULT_ARCHIVE_SHA256):
    """Uniformly select one passing candidate draw using reservoir sampling."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(candidates, bool) or not isinstance(candidates, int) or candidates <= 0:
        raise ValueError("candidates must be a positive integer")
    max_ratio = _number(max_ratio, "max_ratio", lower=1.0)
    if not SHA256.fullmatch(source_archive_sha256):
        raise ValueError("source archive SHA256 must be 64 lowercase hexadecimal characters")
    groups = _groups(cases)
    totals = {m: math.fsum(c["losses"][m] for c in cases) for m in METHODS}
    if any(not math.isfinite(t) or t <= 0 for t in totals.values()):
        raise ValueError("both baselines need finite positive total loss")
    rng = random.Random(seed)
    choose_rng = random.Random(f"{REVISION}:reservoir:{seed}")
    feasible = 0
    selected = None
    best = None
    best_ratio = math.inf
    for _ in range(candidates):
        proposal = [group[rng.randrange(3)] for group in groups]
        # Compute private directly, avoiding cancellation when loss is heavy-tailed.
        public_ids = {c["case"] for c in proposal}
        ratios = []
        for method in METHODS:
            public_mean = math.fsum(c["losses"][method] for c in proposal) / 15
            private_mean = math.fsum(c["losses"][method] for c in cases if c["case"] not in public_ids) / 30
            ratios.append(_ratio(public_mean, private_mean))
        worst_ratio = max(ratios)
        if worst_ratio < best_ratio:
            best_ratio, best = worst_ratio, proposal
        if worst_ratio <= max_ratio:
            feasible += 1
            if choose_rng.randrange(feasible) == 0:
                selected = proposal

    selection = {
        "algorithm": "independent uniform one-of-three per stratum; uniform reservoir over passing candidate draws",
        "seed": seed,
        "candidate_count": candidates,
        "feasible_candidate_count": feasible,
        "target_max_public_private_mean_ratio": max_ratio,
        "selection_methods": list(METHODS),
        "agent_losses_used_for_selection": False,
        "loss_aggregation": "unchanged arithmetic mean of per-sequence mean energy losses",
        "measurement_repeats": 1,
        "measurement_noise_note": "one archived wall-clock-limited measurement per case/method; runtime noise remains",
        "training_stage_balance_enforced": False,
        "candidate_duplicates": "allowed; selection is uniform over passing sampled draws",
        "search_failure_is_infeasibility_proof": False,
    }
    metadata = {
        "revision": REVISION,
        "source_archive_sha256": source_archive_sha256,
        "baseline_results_sha256": hashlib.sha256(_canonical(sorted(cases, key=lambda c: c["case"]))).hexdigest(),
        "post_hoc": True,
        "unseen_holdout_claim": False,
        "selection": selection,
    }
    before_ids = {c["case"] for c in cases if c["original_split"] == "public"}
    best_ids = {c["case"] for c in best} if best else set()
    diagnostics = {
        **metadata,
        "status": "selected" if selected is not None else "random_search_exhausted",
        "before": _diagnose(cases, before_ids),
        "after": None,
        "best_sampled_larger_to_smaller_mean_ratio": best_ratio if math.isfinite(best_ratio) else None,
        "best_sampled_diagnostics": _diagnose(cases, best_ids) if best else None,
        "mathematical_infeasibility_claim": False,
        "scope_note": "maintainer-only post-hoc resplit; unchanged trajectories and scoring; not an independent blind evaluation",
    }
    if selected is None:
        raise NoFeasibleSplit(diagnostics)
    selected_ids = {c["case"] for c in selected}
    manifest_cases = [
        {**{k: c[k] for k in ("case", "data_id", "original_split", "dataset", "profile")},
         "split": "public" if c["case"] in selected_ids else "private"}
        for c in sorted(cases, key=lambda c: c["case"])
    ]
    manifest = {**metadata, "cases": manifest_cases}
    diagnostics["after"] = _diagnose(cases, selected_ids)
    diagnostics["group_counts"] = [
        {"dataset": group[0]["dataset"], "profile": group[0]["profile"],
         "public": sum(c["case"] in selected_ids for c in group),
         "private": sum(c["case"] not in selected_ids for c in group)}
        for group in groups
    ]
    diagnostics["case_difficulties"] = [
        {**row, "losses": c["losses"]}
        for row, c in zip(manifest_cases, sorted(cases, key=lambda c: c["case"]))
    ]
    return manifest, diagnostics


def _write_outputs(out, outputs):
    payloads = {name: json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
                for name, value in outputs.items()}
    for name, payload in payloads.items():
        destination = out / name
        if destination.exists() and destination.read_text() != payload:
            raise ValueError("output differs; use a new --out directory: " + str(destination))
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (out / name).write_text(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="maintainer-only local directory; do not publish")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--candidates", type=int, default=100000)
    parser.add_argument("--max-ratio", type=float, default=1.25)
    parser.add_argument("--source-archive-sha256", default=DEFAULT_ARCHIVE_SHA256)
    args = parser.parse_args()
    try:
        cases = collect_baselines(args.results)
        manifest, diagnostics = rebalance(
            cases, seed=args.seed, candidates=args.candidates, max_ratio=args.max_ratio,
            source_archive_sha256=args.source_archive_sha256,
        )
        _write_outputs(args.out, {"split_manifest.json": manifest, "balance_diagnostics.json": diagnostics})
    except NoFeasibleSplit as error:
        if (args.out / "split_manifest.json").exists():
            raise ValueError("failed search destination already contains a manifest; use a new --out directory") from error
        _write_outputs(args.out, {"balance_diagnostics.json": error.diagnostics})
        print(str(error), file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("rebalance failed: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps({
        "revision": REVISION,
        "status": diagnostics["status"],
        "feasible_candidate_count": diagnostics["selection"]["feasible_candidate_count"],
        "after": diagnostics["after"],
        "post_hoc": True,
        "unseen_holdout_claim": False,
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
