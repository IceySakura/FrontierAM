# Offline Conic Solver

Run `./build.sh` once, then:

```sh
./run.sh --input /absolute/input.npz --output /absolute/result.npz
```

The entry point uses a single numerical thread and the inherited CPU affinity.
No network access, package installation, agent tooling, or per-instance
persistent state is used. Input/output follows schema version 1. Objective
constants do not change the optimizer and objective values are not reported.

Dependencies: Python 3.12.14, NumPy 2.3.5, and SciPy 1.16.3, as preinstalled.
SciPy includes HiGHS 1.8.0. The build only precompiles general Python imports
inside this directory to reduce fresh-process launch overhead.

## Methods

* HiGHS sparse LP/QP algorithms for polyhedral problems, followed by
  original-coordinate residual checks and active-set KKT refinement.
* Direct diagonal/bound-constrained solves, native singleton-bound
  extraction, and accelerated projected iterations with sparse active-set
  polishing for large coupled bound QPs.
* Normalized auxiliary LPs for polyhedral infeasibility certificates.
* Ruiz-style cone-preserving equilibration and a homogeneous primal-dual
  embedding with predictor/corrector directions for general cone products.
* Rank-revealing elimination of small equality systems, with reconstruction
  of primal variables and equality multipliers in the original coordinates.
* Nesterov--Todd scaling for nonnegative, second-order, and semidefinite
  cones; complementary primal/dual logarithmically homogeneous barrier
  paths and analytic Hessian inverses for exponential cones.
* Sparse augmented systems for SOCs, cached sparse semidefinite Schur
  kernels, and cone-first ordering for sparse models with dense equality
  rows. Static factorizations are checked by residual refinement and can
  fall back to pivoted sparse LU.
* Semismooth Newton refinement on active cone faces, with rank-revealing
  minimum-norm solves for degenerate faces, and linear dual/gap repair.
  Repeated small cone operations are vectorized.

The implementation was developed offline from the supplied mathematical
contract, the public checker and examples, and the installed SciPy interface.
No online literature retrieval was possible. Main algorithmic references:
Mehrotra (1992), predictor/corrector interior-point methods; Nesterov and
Todd (1997/1998), self-scaled cone algorithms; Ye, Todd and Mizuno (1994),
homogeneous self-dual embedding; O'Donoghue et al. (2016), homogeneous conic
embeddings; Goulart and Chen (2024), Clarabel's quadratic conic embedding;
and Huangfu and Hall (2018), HiGHS dual simplex. Clarabel itself is not an
installed dependency and is not used by this submission.

Development checks cover all six interface examples and the five supplied
development inputs. All 11 passed three fresh-process repeats from a
relocated, read-only copy, within their public budgets; the maximum local
launch-to-exit time was 0.57 seconds. Timing is diagnostic and depends on
the development host. Additional deterministic synthetic tests exercise cone
products, quadratic objectives, scaling, and 33 infeasibility certificates,
as well as entropy, logistic-loss, and large bound-constrained models.
A deterministic 300-case feasible KKT fuzz run verified 298 cases; one
highly scaled degenerate mixed-cone case and one non-Slater SDP remained
failures. A one-million-variable separable QP also passed the independent
checker. These are development checks, not held-out evaluation results.
