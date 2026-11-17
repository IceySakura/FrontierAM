# Offline Conic Solver

## Build And Run

```sh
./build.sh
./run.sh --input /absolute/input.npz --output /absolute/result.npz
```

The evaluator performs the build once. `run.sh` is executable and uses
`exec`; paths are resolved relative to the submitted source directory.
The input and output use NPZ schema version 1, including finite float64
vectors and scalar Unicode JSON metadata. Objectives are not reported.

No network, downloads, agent tools, instance names, invocation ordering,
precomputed answers, or persistent instance-dependent state are used.
Solving writes only the requested output file. Unsupported numerical
conditions produce `unresolved` and exit normally.

## Dependencies

The sealed Linux amd64 environment provides Python 3.12.14, NumPy 2.3.5,
SciPy 1.16.3, g++ 12.2.0, and system zlib 1.2.13. SciPy's bundled HiGHS
is version 1.8.0. The build produces the native NPZ/separable launcher
and a sparse LDL shared library from the included C++ sources.
Numerical libraries use one thread and inherit the assigned CPU affinity.
The Python runtime skips unused testing and documentation initialization.

## Methods

* Native CSC/NPZ handling solves separable diagonal problems directly,
  including primal and dual certificates, without interpreter startup.
* Feasible zero-objective origins, unconstrained quadratics, and
  equality-only problems have direct mathematical presolve paths.
* An optional objective-free auxiliary substitution path recovers primal
  vectors and equality duals in original coordinates. It is disabled by
  default because the public timing tests did not show a benefit.
* HiGHS solves LPs and very small QPs. LPs can use IPX without crossover;
  quadratic results are refined against the unregularized KKT system.
* General problems use an equilibrated homogeneous primal-dual
  predictor-corrector method. SOC/PSD blocks use Nesterov-Todd scaling;
  exponential blocks use a logarithmic barrier and inverse secant updates.
* Small PSD and SOC cones are converted by exact cone isomorphisms.
  Large SOC blocks use a sparse, quasi-definite two-rank representation.
* KKT patterns are retained between iterations. Sparse LDL reuses its
  symbolic analysis, dense suitable QPs use Cholesky/Schur solves, and
  numerical failures fall back to SuperLU. PSD Schur contributions use
  active coefficient rows rather than a full vectorized cone Hessian.
* Active-set Newton refinement and a small SLSQP fallback improve
  near-optimal points. Maintained cone slack is restored to original
  coordinates; all accepted results are checked there for feasibility,
  cone membership, stationarity, objective gap, or certificate conditions.

Development used all supplied examples/samples, generated known-KKT
problems and certificates, scaled mixed-cone tests, and million-variable
separable and large-SOC tests. Timings were compared with fresh processes;
the grading core's feedback is the authoritative timing measurement.

## References

Mehrotra (1992), primal-dual predictor-corrector implementation;
Nesterov and Todd (1997), self-scaled cone methods;
Ye, Todd, and Mizuno (1994), homogeneous self-dual embeddings;
Davis (2005), Algorithm 836: LDL;
Kraft (1988), sequential quadratic programming;
and the HiGHS, SuperLU, LAPACK, and NumPy/SciPy implementations already
installed in the offline environment.
