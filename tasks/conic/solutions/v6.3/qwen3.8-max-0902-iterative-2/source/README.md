# High-accuracy conic solver (C, single-threaded)

Solves

    min 1/2 x'Px + q'x + c0   s.t.   Ax + s = b,  s in K

where `K` is a product of zero / nonnegative / second-order / exponential / PSD cones,
and returns `solved` (x,s,z), `primal_infeasible` (z) or `dual_infeasible` (x,s).

## Run

```sh
./build.sh          # compiles ./solver with gcc (needs liblapack/libblas/libz)
./run.sh --input INPUT.npz --output OUTPUT.npz
```

`run.sh` `exec`s the native binary `solver`; there is no interpreter start-up cost.

## Dependencies (fixed, preinstalled in the image)

* gcc 12.2.0, `-O2`
* LAPACK/BLAS reference build shipped with Debian 12 (`liblapack.so.3`, `libblas.so.3`) — used
  only for `dsyevd` (symmetric eigendecomposition) and `dgemm`.
* `libz.so.1` (zlib 1.2.13) — used only to inflate the DEFLATE-compressed NPZ entries.
  `build.sh` locates it by absolute path, no `libz.so` dev symlink is required.
* No Python is needed at solve time (NumPy/SciPy start-up alone costs 200-650 ms on this
  image, which is most of the 1 s budget of typical instances).

## Method

Mehrotra predictor–corrector primal–dual interior point method on the **reduced KKT system**

    [ P + A_I' W^-2 A_I + rho I    A_E' ] [ dx ]   [ -r_d + A_I'W^-1 (r_c - W^-1 r_pI) ]
    [ A_E                          -delta I] [ dz_E ] = [ -r_pE ]

derived from `dx, ds = -A dx - r_p, dz` with the symmetrised complementarity equation
`W dz + W^-1 ds = -r_c`.  Rows of zero cones are kept as explicit equalities (`ds = 0`, `z`
free), all other rows are scaled by a per-block operator `W`:

| cone | scaling `W` | notes |
| --- | --- | --- |
| nonnegative | `diag(sqrt(s/z))` | Nesterov–Todd (= log-barrier Hessian) |
| soc | `L(w)`, `w^2 = s ∘ z^-1` (Jordan) | Nesterov–Todd scaling point |
| psd | `X ↦ G X G`, `G = mu^-1/4 S^1/2` | log-det barrier Hessian scaling |
| exponential | `mu^-1/2 H^-1/2`, `H = ∇²φ`, `φ = -log v - log w - log(w - v e^{u/v})` | barrier Hessian scaling (as in ECOS) |

For symmetric cones `lambda = W z = W^-1 s`; the same identity holds for self-concordant
barriers because of Euler's relation `H s = -∇φ(s)`.  The centering residual is written
uniformly as `r_c = lambda_z - T ∘ lambda_s^-1` with `T = sigma*mu*e - ds_hat ∘ dz_hat`,
which reduces to the classical LP/SOC/SDP Mehrotra right-hand side and to the exact
barrier Newton equation for the exponential cone.

Linear algebra: the quasi-definite KKT matrix has a **fixed sparsity pattern**, so the
symbolic structure is built once.  It is factored by a simplicial left-looking `LDL^T`
(no pivoting — stable for quasi-definite matrices) with dynamic column storage and
Gilbert's etree-reach pattern computation.  A fill-reducing permutation is chosen on the
first iteration by comparing reverse Cuthill–McKee against the natural order.

Robustness: `rho`/`delta` regularisation is increased on factorisation failure; every step
is safeguarded (step is halved until the new point is strictly inside `K x K*` and finite);
sigma is retried with increasing centering if no feasible step exists; a final polish sets
`s = b - Ax` when that strictly improves the acceptance residuals.

Termination uses exactly the evaluator's acceptance functionals (relative primal/dual
residuals, normalised per-block cone defects for `K` and `K*`, absolute-or-relative duality
gap) with a 9e-9 internal threshold against the 1e-8 contract.

Infeasibility: at every iteration the normalised iterates are tested against the
evaluator's certificate conditions (`z/||z||` with `A'z=0, z in K*, b'z<0`, and
`x/||x||, s/||x||` with `Px=0, Ax+s=0, s in K, q'x<0`); the first certificate that passes
at 1e-9 is returned.

## Output

`OUTPUT.npz` contains `metadata_json` (0-d `<U*` array with `schema_version=1` and
`status`) plus the vectors required by the status.  Written as an uncompressed ZIP of
hand-rolled `.npy` files; objective fields are omitted (they are optional).

## References

* Mehrotra, S. "On the implementation of a primal-dual interior point method for LP." SIAM J. Optim. 1992.
* Nesterov, Y., Todd, M. "Primal-dual algorithms and invariant properties for a symmetric conic problem." 1997.
* Domahidi, Chu, Boyd. "ECOS: an SOCP solver for embedded systems." ECC 2013 (exponential-cone barrier Hessian scaling).
* Gowda, Tanguy. "Clarabel: an interior point solver for conic problems with quadratic objectives." 2024 (reduced/augmented KKT, quasi-definite factorisation).
* Andersen, Dahl, Vandenberghe. "Implementation of conic optimization." (SOC/SDP scaling details.)
* Davis. "Direct Methods for Sparse Linear Systems" (etree reach, simplicial Cholesky).
* Gilbert, Peierls. "Sparse partial pivoting in time proportional to arithmetic operations." 1988.
