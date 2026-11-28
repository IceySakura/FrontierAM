# Conic solver submission

Solves  min 1/2 x'Px + q'x  s.t.  Ax + s = b,  s in K  for the cone families
`zero`, `nonnegative`, `soc`, `exponential`, `psd`.

## Usage

```sh
./run.sh --input INPUT.npz --output OUTPUT.npz
```

`run.sh` starts `conic.py` with `exec` (no background processes are left
behind).  No build step is required (`build.sh` is not provided): the solver
only needs the pre-installed Python 3.12 with NumPy 2.3.5 and SciPy 1.16.3.

## Method

Primal-dual interior point method (Mehrotra predictor/corrector) written
directly on the given problem data.

* Every cone block is described by its logarithmically homogeneous
  self-concordant barrier psi (nonnegative: -sum log, soc: -log(t^2-||u||^2),
  psd: -log det, exponential: -log(w - v e^{u/v}) - log v).  The central path
  is `z = -mu * grad psi(s)`; for the nonnegative cone this reduces exactly to
  the classical `s_i z_i = mu` complementarity and the classical diagonal
  scaling, so LP/QP instances get the standard algorithm.
* Newton systems are reduced to the symmetric quasi-definite augmented system
  `[P+dp I, A'; A, -(W+dd I)]` with `W = Hessian^{-1}/mu` per block (diagonal
  for the linear rows, dense for soc/exp/psd blocks).  It is factorised with a
  sparse LU (SuperLU, symmetric ordering) and solved with iterative refinement
  against the unregularised system.
* Separate primal/dual step lengths from exact cone line searches (ratio test
  for the linear cone, quadratic root for soc, bisection for the exponential
  cone, eigenvalue based for psd).
* Before any status is reported the candidate is re-checked against the
  published acceptance conditions (relative primal/dual residuals, cone
  defects, absolute or relative duality gap); a `solved` claim that fails the
  check is downgraded to `unresolved`.

## References

* Nesterov & Nemirovski, *Interior-Point Polynomial Algorithms in Convex
  Programming* (barrier theory, exponential cone barrier).
* Mehrotra, *On the implementation of a primal-dual interior point method*
  (1992) - predictor/corrector.
* Andersen & Andersen, *The MOSEK interior point optimizer for linear
  programming: an implementation of the homogeneous algorithm* (KKT system
  with regularization and iterative refinement).
* Domahidi, Chu & Boyd, *ECOS: an SOCP solver for embedded systems* (cone
  scaling in the augmented system).
