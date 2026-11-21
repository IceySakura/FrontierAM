# Offline Conic Solver

Run:

```sh
./build.sh
./run.sh --input /absolute/problem.npz --output /absolute/result.npz
```

The build is entirely offline. It prepares Python bytecode for imports; it
does not read any problem instances. Runtime files are read-only except for
the requested output. No network, agent tools, external service, or downloaded
package is used.

## Dependencies

Preinstalled Python 3.12.14, NumPy 2.3.5, and SciPy 1.16.3. SciPy includes
HiGHS 1.8.0. All numerical libraries are restricted to one thread. The solver
inherits CPU affinity and does not select a particular CPU index.

## Method

* Sparse LP and small convex QP dispatch to the bundled HiGHS implementation.
  Active-set KKT polishing and separate bounded certificate problems validate
  ambiguous results.
* Separable bound-constrained LPs and diagonal QPs have a vectorized exact
  solution path, including explicit contradictory-bound and recession
  certificates.
* Other cones use an original homogeneous primal-dual predictor-corrector
  implementation, followed by fixed-scale refinement near a feasible optimum.
* Nonnegative, Lorentz, and PSD blocks use Nesterov-Todd scaling. Exponential
  blocks use a logarithmically homogeneous barrier, positive-definite secant
  scaling, and a directional third-derivative correction.
* Cone-compatible equilibration, exact fixed-variable and sparse-equality
  elimination, PSD block splitting, orthogonal low-order cone conversions,
  sparse scaled Newton systems, and dense QR solves address different
  problem structures. Large SOC scaling is represented by sparse diagonal
  and low-rank terms rather than dense cone matrices.
* Active-constraint Newton refinement is available for SOC and exponential
  problems. Full Newton trial points are accepted only after validation.
* Returned solutions and infeasibility directions are checked in the input
  coordinates. Output objectives are omitted to avoid redundant rounding.

The NPZ schema, PSD upper-triangular sqrt(2) convention, exponential coordinate
order, and primal/dual certificate conventions follow the supplied contract.
Instance names and metadata do not influence the algorithm.

## References

The implementation uses standard techniques described by:

* Mehrotra (1992), "On the Implementation of a Primal-Dual Interior Point Method."
* Nesterov and Todd (1997), "Self-Scaled Barriers and Interior-Point Methods for
  Convex Programming."
* Ye, Todd, and Mizuno (1994), homogeneous self-dual methods for linear
  programming.
* Andersen, Dahl, and Vandenberghe (2011), "Interior-point methods for
  large-scale cone programming."
* HiGHS and SciPy, using only the preinstalled implementation.

Development uses the supplied independent checker, all five public sample
instances, the interface examples, and generated problems with known KKT
points or infeasibility directions. These are development checks, not a
claim about hidden-set success.

## Validation

All five available public instances passed three fresh-process repetitions
within their published limits. Final observed launch-to-exit times were below
0.6 seconds per run on this development host. A separately copied, read-only
submission also passed all five cases three times with a clean environment.

Additional development suites passed 45 planted KKT problems, 24 infeasibility
problems, 18 boundary-cone problems, and 24 auxiliary-equality recovery
problems. The supplied interface examples and constant-cone cases were also
checked. No hidden instances, reference answers, or grading feedback were
available.
