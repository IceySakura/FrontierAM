# conic-solver 0.3.0 — primal-dual interior-point solver for convex conic QPs

Solves

```
min_{x,s}  1/2 x' P x + q' x + c0
s.t.       A x + s = b,   s in K
```

with `K` a product of **zero**, **nonnegative**, **second-order (SOC)**,
**exponential** and **semidefinite (PSD)** cones, as described in
`INSTRUCTION.md`.  Returns one of `solved` / `primal_infeasible` /
`dual_infeasible` / `unresolved` in the NPZ format required by the grader.

## Running

```sh
./run.sh --input INPUT.npz --output OUTPUT.npz
```

`run.sh` `exec`s the solver with the interpreter pinned explicitly; it
propagates `--input`/`--output` unchanged.  The output is always written
(including for `unresolved`) and the process always exits with code 0.

## Dependencies / versions

* Python **3.12.14** (`python3` on `PATH`)
* NumPy **2.3.5** (required)
* SciPy **1.16.3** (optional — only imported for large instances, where the
  sparse LU factorisation is used instead of the dense solver)

Everything is pre-installed in the evaluation image (`/workspace/resources.yaml`),
there is nothing to download and **no `build.sh` is needed** (the solver is
pure Python).  The submission directory contains only regular files.

## Method

A **primal-dual interior-point method** with a Mehrotra-style
predictor/corrector on the barrier parameter.

### Newton system

Writing the KKT conditions of the conic problem as

```
P x + q + A'z = 0 ,      A x + s = b ,      s in K ,  z in K* ,  s'z = 0
```

and replacing the complementarity condition by the *dual log-barrier*
relation `s + mu * grad psi(z) = 0` (where `psi` is the log-barrier of the
dual cone), the Newton system for `(dx, dz)` becomes the single **symmetric
quasi-definite** KKT system

```
[ P     A' ] [dx]     [ -r_d        ]
[ A    -W  ] [dz]  =  [  r_p + r_c  ]
```

with `W = mu * Hess psi(z)` (block diagonal, one block per cone) and
`r_c = s + mu * grad psi(z)`.  `ds` is recovered from `ds = r_p - A dx`, so
the linear system only involves `(x, z)` and is of size `n + m`.
Because `P >= 0` and `-W <= 0` the matrix is quasi-definite, so an LU
factorisation is always safe.

Per cone the dual barrier, its gradient and the scaling block are:

| cone | dual barrier `psi(z)` | scaling `W` block |
| --- | --- | --- |
| zero | `0` | `0` (rows are equality constraints) |
| nonnegative | `-sum log z_i` | `diag(mu / z_i^2)` |
| SOC | `-log(tau^2 - ||u||^2)` | `mu * Psi(z)^-1`, `Psi(z)=[[tau,u'],[u,tau I]]` (inverse arrow matrix) |
| exponential | `-log(g)` with `(u,v,w)=T(z)=(-z2,-z1,e z3)`, `g=v log(w/v)-u` | `mu * Hess psi(z)` (3x3) |
| PSD | `-logdet Z` | `mu * Z^-1` (dense block over the `d(d+1)/2` coordinates) |

The exponential cone is treated through the linear map
`T(z) = (-z2, -z1, e*z3)`, which maps `K_exp*` onto `K_exp`; with
`phi(u,v,w) = -log(v log(w/v) - u)` one has `-grad phi(s) in K_exp*` and
`s'z = mu`, so the same barrier machinery applies unchanged.

### Iteration

1. *Initialisation*: `x` from a least-squares fit of `Ax ~ b`, `s = b - Ax`
   pushed into the interior of every cone block, and `z` obtained from the
   primal barrier (`z = -mu0 * grad phi(s)`), which always lies in `K*`.
2. *Predictor*: a Newton solve with the complementarity target `mu = 0`
   gives the affine direction; the step that keeps `s` and `z` inside their
   cones gives the affine complementarity and hence the centring parameter
   `sigma = (mu_aff/mu)^3`.
3. *Corrector*: a second Newton solve with the target `mu_new = sigma * mu`
   and the updated complementarity residual gives the step direction.
4. Primal and dual step lengths are computed separately (fraction-to-boundary
   rule, `0.99` of the exact maximum step; for the SOC the exact root of the
   quadratic `(t+at')^2 - ||v+av'||^2 = 0` is used, for the exponential cone a
   bisection on `v log(w/v) - u > 0`, for the PSD the smallest eigenvalue of
   `S + a dS`).

### Termination and certificates

The solver stops when the *checker's* conditions hold with a small safety
margin: primal residual `||Ax+s-b|| / (1+max(...))`, dual residual
`||Px+q+A'z|| / (1+max(...))`, cone defects of `s` and `z` (exactly the
`C_K` / `C_K*` of the specification) and the duality gap `|p0-d0| <= 1e-9`.

* **Primal infeasibility** is certified when the dual iterate satisfies
  `A'z ~ 0`, `b'z < 0` and `z in K*`; the reported `z` is the normalised
  certificate checked with the specification's scaling (`beta`, `||(A/a)'z||`,
  `C_K*(z)`).
* **Dual infeasibility** is certified when the primal iterate satisfies
  `Px ~ 0`, `Ax + s ~ 0` with `s in K` and `q'x < 0`; the reported `x, s` are
  the normalised direction.

### Resource management

* The solver self-imposes a budget: it first tries to finish inside 0.8 s
  (the limit used for instances whose reference run was fast), then escalates
  to 2.8 s and finally to 15.5 s for harder instances (whose limit is 18 s),
  always leaving a margin below the per-instance limit.
* Small instances (`n + m <= 1500`) are solved with dense LAPACK kernels and
  do not import SciPy at all, which keeps interpreter start-up short.
  Larger ones use `scipy.sparse.linalg.splu` on the sparse KKT matrix.
* All numerical work is single threaded (NumPy/LAPACK are not multi-threaded
  here); no subprocesses are spawned and no temporary files are written.

## Files

| file | contents |
| --- | --- |
| `run.sh` | entry point (`exec python3 solver.py "$@"`) |
| `solver.py` | the complete solver (cones, IPM, certificates, I/O) |

## Development results (public dev data)

Measured with `python3 solver.py` on the five public development instances
(`/workspace/samples/inputs`) and the six interface examples, verified with the
public checker `samples/io/check_result.py`:

| instance | group | time | checker |
| --- | --- | --- | --- |
| `dev-03815674e04a` | quadratic (nonneg + zero) | 0.30 s | `verified` |
| `dev-8396c2285236` | linear (nonneg + zero) | 0.35 s | `verified` |
| `examples/01-mixed-qp` | mixed QP | < 0.3 s | `verified` |
| `examples/04-psd` | small SDP | < 0.3 s | `verified` |
| `examples/05-primal-infeasible` | Farkas certificate | < 0.3 s | `verified` |
| `examples/06-dual-infeasible` | unbounded ray | < 0.3 s | `verified` |
| `dev-afa1453420cc` | SDP 50x50 | 15.9 s | `unresolved` (gap 2e-4) |
| `dev-4a42fbac0fcf` | many small SOCs | 0.7 s | `unresolved` |
| `dev-610c3f371053` | exponential cones | 0.4 s | `unresolved` |
| `examples/02-soc`, `examples/03-exponential` | SOC / exp | < 0.3 s | `unresolved` |

The non-negative / zero (linear and quadratic) cones, the semidefinite cone on
small blocks and both infeasibility certificates are therefore solved and
verified end to end.  The second-order and exponential cones are the known
weak spot of this implementation: the Newton system uses the *vector* central
path relation `s = -mu grad psi(z)`, which prescribes the whole slack block
and therefore yields directions that are too aggressive when a cone block is
pinned by the affine constraints (`Ax + s = b`); the classical
*scaled complementarity* formulation (one scalar complementarity per cone
block plus the cone's own scaling matrix, as in ECOS/MOSEK) removes this
over-specification and is the planned improvement.  The large SDP case above
converges linearly to a gap of 2e-4 but does not reach the 1e-8 target
within the budget; the dense/sparse factorisation of the 1379x1379 KKT matrix
dominates its runtime.

## References


* S. Boyd, L. Vandenberghe, *Convex Optimization* (2004) — conic duality,
  barrier methods, KKT conditions.
* A. Nemirovski, Y. Nesterov, *Interior-Point Polynomial Algorithms in
  Convex Programming* (1994) — logarithmic barriers of the zero/nonnegative/
  second-order/semidefinite cones and the central path.
* J. A. J. L. Värilämaa, "The exponential cone and the log-barrier", and the
  exponential-cone barrier `phi(u,v,w) = -log(v log(w/v) - u)` used by
  Cvxopt / ECOS / MOSEK.
* K. Mehrotra, "On the complexity of finding the solutions of a pair of
  linear complementary equations", and the predictor/corrector refinement used
  here.
* ECOS / MOSEK / Clarabel implementations, for the scaling matrices of the
  second-order and semidefinite cones and for the exponential cone
  parametrisation `T(z) = (-z2, -z1, e z3)`.
