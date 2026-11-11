# r1: refreshed-secant L-BFGS-PCG

This is the proposed 50-point reference, not an active machine-independent anchor or a claim of universal SOTA. Each sequence starts a fresh solver. The reference validates a warm start, refreshes retained secant images against the current operator, uses a positive-definite limited-memory inverse preconditioner, and solves with PCG. All setup and verification HVPs are charged. Its equal remaining-time allocation is a solver policy.

`solver.py` and `_algorithms.py` form a portable submission closure; the trusted worker supplies `contracts.BudgetExceeded`. No teacher factors, private inputs, hidden seeds or historical Apple calibration values are included.

Run `versions/v1/grader/evaluate.py --recalibrate` on the maintainer's target machine before activating scores. The task remains draft: amd64 offline image validation, isolated worker deployment, two-run stability, 25% reference deadline headroom and a non-saturation review are pending. The explicit headroom criterion is `max(reference wall time) <= 0.75 * 30 = 22.5 seconds`.
