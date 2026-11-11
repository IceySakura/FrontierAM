# Offline Conic Solver

Run from this directory:

```sh
./run.sh --input /absolute/path/input.npz --output /absolute/path/result.npz
```

No build, installation, network connection, agent CLI, or writable source
directory is required. The only output file is the requested NPZ result.
`run.sh` uses `exec`, forces single-threaded numerical libraries, and retains
the CPU affinity inherited from the evaluator. It never assumes a CPU index.

`python selftest.py` runs generated small regression problems, including
empty dimensions, all supported cones, and both certificate statuses.

## Dependencies

The numerical image supplies Python 3.12.14, NumPy 2.3.5, and SciPy 1.16.3.
SciPy includes HiGHS 1.8.0 in `scipy.optimize._highspy._core`; this bundled
binding is used directly. No additional packages or native compilation are
needed. All solver-specific source is included here.

## Methods

- Sparse primal-dual predictor-corrector interior-point optimization, with
  a homogeneous embedding for solutions and infeasibility directions.
- Cone-preserving Ruiz equilibration, reversible fixed-variable elimination,
  constant-cone removal, and sparse equality substitution. Primal variables,
  slacks, and dual multipliers are restored to the original coordinates.
- Nesterov-Todd scaling for second-order and semidefinite cones. Newton
  systems for vector cones use square-root-scaled augmented equations;
  semidefinite blocks use sparse-support dense Schur complements and
  Cholesky/SVD scaling.
- A logarithmically homogeneous exponential barrier for general exponential
  blocks. Recognizable monotone exponential epigraphs are eliminated exactly
  and solved as smaller log-sum-exp programs with analytic derivatives.
- HiGHS simplex/interior-point LP solving, bounded homogeneous certificate
  problems, active KKT polishing, and a supporting-plane SOCP fallback.
- Within-input graph decomposition for very large polyhedral problems,
  vectorized independent two-variable LPs, and reuse of identical components
  only within the current input. No instance names, stored answers, ordering,
  or cross-instance state are used.
- Original-coordinate residual, cone, gap, and normalized certificate checks
  before emitting a successful status. Optional objective fields are omitted,
  so the evaluator computes objectives, including the input offset.

The development tests use the supplied interface examples and public
independent checker, including equality/inequality QPs, all five cone types,
large sparse models, and both infeasibility certificate formats. These are
development checks, not private-evaluation or calibrated score claims.

One broad development sweep verified 112 of 118 public inputs with at most
15,000 variables under a local 20-second cap. The remaining cases included
large-QP timeouts and numerically difficult SDPs. The six interface examples
also passed three fresh-process repeats from a read-only clean copy;
eleven generated regressions and four generated mixed-cone checks passed.
Some large or degenerate instances can still return `unresolved`.

## Main References

- Mehrotra, *On the Implementation of a Primal-Dual Interior Point Method*,
  SIAM Journal on Optimization, 1992.
- Nesterov and Todd, *Self-Scaled Barriers and Interior-Point Methods for
  Convex Programming*, Mathematics of Operations Research, 1997.
- Ye, Todd, and Mizuno, *An O(sqrt(n)L)-Iteration Homogeneous and Self-Dual
  Linear Programming Algorithm*, Mathematics of Operations Research, 1994.
- Boyd and Vandenberghe, *Convex Optimization*, 2004, especially geometric
  programming and primal-dual interior-point methods.
- Huangfu and Hall, *Parallelizing the Dual Revised Simplex Method*,
  Mathematical Programming Computation, 2018; the preinstalled HiGHS
  implementation and SciPy numerical linear algebra interfaces.

The conic algorithms and preprocessing in this directory were implemented
locally using the installed numerical primitives. Development was offline.
