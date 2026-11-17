"""Enumerate stratified splits using only CG baselines; emit aggregate evidence.

The full combinatorial search is exhaustive.  Loss calculations and threshold
comparisons use float64, rather than certified interval or exact-real arithmetic.
No allocation or case identity is published by this diagnostic tool.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import numpy as np

from rebalance_split import (
    DEFAULT_ARCHIVE_SHA256, METHODS, _canonical, _groups, collect_baselines,
)


REVISION = "balanced-split-feasibility-20261117"
DEFAULT_THRESHOLDS = (1.25, 1.5, 1.75, 2.0)


def _half_sums(losses):
    """Return public/private sums for every one-of-three half assignment."""
    size = len(losses)
    if size == 0:
        return np.zeros((1, 2)), np.zeros((1, 2))
    codes = np.arange(3 ** size, dtype=np.int64)[:, None]
    powers = 3 ** np.arange(size - 1, -1, -1, dtype=np.int64)
    choices = (codes // powers) % 3
    positions = np.arange(size)[None, :]
    public = losses[positions, choices, :].sum(axis=1)
    # Sum the two unselected cases explicitly, avoiding total-minus-tail
    # cancellation when one trajectory contributes nearly all the loss.
    complement = np.stack((losses[:, 1] + losses[:, 2],
                           losses[:, 0] + losses[:, 2],
                           losses[:, 0] + losses[:, 1]), axis=1)
    private = complement[positions, choices, :].sum(axis=1)
    return public, private


def analyze_losses(losses, *, thresholds=DEFAULT_THRESHOLDS, block_size=64):
    """Exhaustively analyze a (groups, 3 cases, 2 baselines) loss array."""
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 3 or losses.shape[1:] != (3, 2) or losses.shape[0] < 1:
        raise ValueError("expected a nonempty (groups, 3, 2) loss array")
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("losses must be finite and nonnegative")
    totals = losses.sum(axis=(0, 1))
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("both baselines need finite positive total loss")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    thresholds = tuple(float(t) for t in thresholds)
    if not thresholds or any(not math.isfinite(t) or t < 1 for t in thresholds):
        raise ValueError("thresholds must be finite and at least 1")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("thresholds must be distinct")
    thresholds = tuple(sorted(thresholds))
    group_count = len(losses)
    middle = group_count // 2
    left_public, left_private = _half_sums(losses[:middle])
    right_public, right_private = _half_sums(losses[middle:])
    counts = {threshold: 0 for threshold in thresholds}
    best_ratio = math.inf
    best_means = None
    enumerated = 0
    for begin in range(0, len(left_public), block_size):
        end = min(begin + block_size, len(left_public))
        public = (left_public[begin:end, None, :] + right_public[None, :, :]) / group_count
        private = (left_private[begin:end, None, :] + right_private[None, :, :]) / (2 * group_count)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratios = np.maximum(public / private, private / public)
        # Positive global totals exclude a both-zero pool pair for a method.
        worst = ratios.max(axis=2)
        for threshold in thresholds:
            counts[threshold] += int(np.count_nonzero(worst <= threshold))
        flat_index = int(np.argmin(worst))
        row, column = np.unravel_index(flat_index, worst.shape)
        local_best = float(worst[row, column])
        if local_best < best_ratio:
            best_ratio = local_best
            best_means = {
                method: {
                    "public_mean_loss": float(public[row, column, i]),
                    "private_mean_loss": float(private[row, column, i]),
                    "larger_to_smaller_mean_ratio": float(ratios[row, column, i]),
                }
                for i, method in enumerate(METHODS)
            }
        enumerated += worst.size
    expected = 3 ** group_count
    if enumerated != expected:
        raise AssertionError("incomplete combinatorial enumeration")
    return {
        "algorithm": "exhaustive one-of-three stratified enumeration, split halves and blocked Cartesian product",
        "group_count": group_count,
        "case_count": 3 * group_count,
        "public_count": group_count,
        "private_count": 2 * group_count,
        "half_group_counts": [middle, group_count - middle],
        "half_assignment_counts": [len(left_public), len(right_public)],
        "block_size": block_size,
        "total_splits": expected,
        "enumerated_splits": enumerated,
        "minimum_worst_mean_ratio": best_ratio if math.isfinite(best_ratio) else None,
        "best_pool_means": best_means,
        "thresholds": [
            {"max_ratio": threshold,
             "feasible_count": counts[threshold],
             "feasible_fraction": counts[threshold] / expected}
            for threshold in thresholds
        ],
        "complete_combinatorial_enumeration": True,
        "arithmetic": "numpy float64; direct <= threshold comparison; not certified exact-real arithmetic",
        "objective": "max over Cold/Warm CG of max(public mean / private mean, inverse)",
    }


def analyze_results(roots, *, thresholds=DEFAULT_THRESHOLDS, block_size=64):
    cases = collect_baselines(roots)
    groups = _groups(cases)
    losses = [[[case["losses"][method] for method in METHODS] for case in group]
              for group in groups]
    return {
        "revision": REVISION,
        "source_archive_sha256": DEFAULT_ARCHIVE_SHA256,
        "baseline_results_sha256": hashlib.sha256(_canonical(sorted(cases, key=lambda c: c["case"]))).hexdigest(),
        "post_hoc": True,
        "unseen_holdout_claim": False,
        "selection_methods": list(METHODS),
        "agent_losses_used": False,
        "allocation_selected": False,
        "scope_note": "aggregate feasibility of redistributing the existing 45 trajectories; no case identities or assignment emitted",
        **analyze_losses(losses, thresholds=thresholds, block_size=block_size),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSON file for aggregate evidence")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()
    try:
        result = analyze_results(args.results, thresholds=args.thresholds, block_size=args.block_size)
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out.exists() and args.out.read_text() != payload:
            raise ValueError("output differs; use a new --out file")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("feasibility analysis failed: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in (
        "revision", "total_splits", "minimum_worst_mean_ratio", "thresholds"
    )}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
