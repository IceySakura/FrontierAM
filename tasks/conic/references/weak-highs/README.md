# Weak baseline: the bundled HiGHS, as is

`./run.sh --input IN.npz --output OUT.npz` runs `solver.py` with the image's Python (3.12, NumPy 2.3.5, SciPy 1.16.3).
No build step, no dependencies beyond the image.

Problems whose cones are all zero or nonnegative go to the HiGHS LP/QP solver shipped inside SciPy
(`scipy.optimize._highspy`): free columns, zero rows fixed at `b`, nonnegative rows `Ax <= b`, primal and dual
feasibility tolerances 1e-9, nothing else changed. An optimal model is written as `solved` with `x`, `s = b - Ax` and
`z = -row_dual`. Every other cone (second-order, exponential, PSD) and every infeasible or unbounded model is
`unresolved`: this SciPy build exposes no HiGHS dual or primal rays, so no certificate can be formed.

Measured on the public samples: the LP sample is solved and verified; the two QP samples come back optimal from HiGHS
but miss the task's 1e-8 tolerance (dual residual 3-4e-8, relative gap up to 1), since the bundled QP solver works to
about 1e-7; start-up plus solve take 0.8-0.9 s, close to the 1-second limit of small cases.

Role: the 0-point anchor of the board's conic mapping (user decision 2026-11-26): it stands for what calling the
bundled library gives without writing a solver. Not a reference for the task author's mapping.
