# Offline Conic Solver

## Build And Run

```sh
./build.sh
./run.sh --input /absolute/input.npz --output /absolute/result.npz
```

The build compiles a small C coordinate-descent kernel and prepares a local
Python bytecode cache. It does not read problem instances. The run script uses
`exec`, accepts the two specified paths, and works from any working directory.
No installation, downloads, network, agent tools, or cross-instance state is
required. Runtime writes only the requested output; all preprocessing is
instance-local and in memory.

The fixed environment dependencies are Python 3.12.14, NumPy 2.3.5, SciPy
1.16.3, and SciPy's bundled HiGHS 1.8.0. Building the C kernel uses GCC 12.2.0
and the system C/math libraries. All numerical libraries run single-threaded.
CPU affinity is inherited, with no hard-coded processor number. The source
and build are suitable for the Linux amd64 portable-v1 environment.

## Methods

- Input/output: full symmetric CSC quadratic matrices, duplicate summation,
  ordered product cones, upper-column sqrt(2)-scaled PSD encoding, and scalar
  Unicode JSON metadata. Optional objective fields are deliberately omitted.
- Presolve: exact fixed-variable substitutions, elimination of unpenalized
  singleton equality columns, redundant constant rows, and zero-radius SOC
  reduction. Recovery includes the necessary dual multipliers and preserves
  certificate directions.
- LP: HiGHS simplex or interior-point optimization, followed by original-data
  checks. Alternative-system LPs construct Farkas and recession certificates.
- QP: HiGHS active-set optimization and a sparse Mehrotra predictor-corrector
  fallback, with active-set KKT refinement and diagonal equilibration.
- SOC: vectorized Nesterov--Todd scaling, direct and homogeneous primal-dual
  paths, augmented sparse KKT factorization, and active-boundary Newton
  polishing. Larger systems use symmetric minimum-degree ordering. A
  polyhedral relaxation provides an additional solve and initialization path.
- Exponential: eligible fixed-perspective epigraphs are reduced exactly to
  stable log-sum-exp constraints, optimized with SLSQP, and polished using
  their analytic KKT system. Other exponential and mixed-cone problems use
  supporting-hyperplane outer approximation.
- PSD: primal-dual Nesterov--Todd scaling with a Schur complement assembled
  only on matrix coordinates touched by the sparse constraint operator.
  Feasible-start improvements, exact predictor identities, iterative
  refinement, checked extrapolation, and spectral projection Newton
  polishing improve attainable accuracy.
- Very large regression QPs: algebraic recognition of least-squares/l1 and
  Huber epigraphs permits exact auxiliary-variable elimination. The reduced
  problems use sparse coordinate descent, active-set refinement, or sparse
  least-squares solves. Free singleton columns are peeled and back-substituted
  before iterative regression solves. Recognition depends only on
  coefficients and sparsity, not names or known answers.
- Outer approximation: every added row is a valid dual-cone support. Its
  multipliers are lifted back to the original cone coordinates, including
  PSD rank-one supports. Quadratic relaxations receive KKT refinement.

Solutions and certificates are accepted only after checking the original
input coordinates against the updated residual, cone-defect, objective-gap,
and normalized certificate conditions. Inconclusive runs write `unresolved`
and exit normally. There is an internal 285-second safety budget; evaluator
limits may be shorter. `CONIC_TIME_LIMIT` and `CONIC_DEBUG` are optional local
diagnostic environment variables, not required for evaluation.

## Validation

Development used the bundled public inputs and independent `check_result.py`
with epsilon 1e-8. The six supplied interface examples and fourteen additional
synthetic cases were verified in fresh subprocesses, including empty vectors,
fixed bounds, quadratic objectives, mixed cones, and both certificate types.
Public regression tests covered all five problem groups and included
multi-million-variable inputs.

A separate copy was built offline and then made read-only. From a different
working directory and a stripped environment, all 27 verification invocations
passed. Seven representative cases were each run in three fresh processes;
their statuses were consistent. No agent tools were used by those runs.

These are development checks, not a claim about private-case success or
private timing. No hidden evaluation data, saved instance answers, or online
resources are included in the submission.

## References

- The installed SciPy 1.16.3 HiGHS wrapper and HiGHS 1.8.0 implementation.
- Huangfu and Hall, "Parallelizing the dual revised simplex method" (2018).
- Mehrotra, "On the implementation of a primal-dual interior point method"
  (1992).
- Nesterov and Nemirovskii, *Interior-Point Polynomial Algorithms in Convex
  Programming* (1994).
- Nesterov and Todd, "Self-Scaled Barriers and Interior-Point Methods for
  Convex Programming" (1997).
- Boyd and Vandenberghe, *Convex Optimization* (2004), for conic duality,
  barriers, and geometric programming.
- Friedman, Hastie, and Tibshirani, "Regularization Paths for Generalized
  Linear Models via Coordinate Descent" (2010).

The numerical Python/C code in this directory was implemented offline.
Third-party numerical routines are used from the preinstalled environment.
