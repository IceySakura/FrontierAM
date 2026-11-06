# Rational shift-and-invert Lanczos panel member

This is an independent C++ implementation of preconditioned/rational Lanczos
for

`exp(-t L) b0 + t phi1(-t L) b1`.

## Method

For `t > 0`, the program forms

`M = (I + gamma L)^(-1)`.

The standard shift is `gamma=t/10`.  Large irregular graphs for which exact
fill is predictably expensive use `gamma=t/100`, one diagonal incomplete
Cholesky factorization, and preconditioned CG applications
of the same shifted inverse.  Other inputs use a simplicial CHOLMOD sparse
Cholesky factorization.  The factorization is included in the timed session
and is reused for every Lanczos step and, in a 16-request session, every
request.  Lanczos is applied to the symmetric positive definite operator `M`.

The fill-avoidance rule is fixed from public dimensions: `n >= 50000` and more
than `4*n` stored entries in the lower triangle.  Inner CG uses relative
residual `2e-13`, relaxed to `0.1*rtol` only when
`t*||L||_infinity > 50000`.

If `theta` is a Ritz value, the corresponding eigenvalue of `L` is

`lambda = (1/theta - 1)/gamma`.

The two projected scalar functions are therefore

`exp(-c(1/theta - 1))`

and

`t * (1 - exp(-c(1/theta - 1))) / (c(1/theta - 1))`,

where `c=t/gamma` is 10 or 100.  The alternate proportional shift balances
inner iteration conditioning against outer rational-Lanczos dimension on
fill-heavy graphs.

The latter is evaluated with `expm1` and a local series at zero.  The two
right-hand sides use separate rational Lanczos spaces, but convergence is
tested on their *combined output*.  Every fourth step, successive combined
approximations are compared in infinity norm.  Two consecutive estimates must
be below

`m * (atol + rtol * ||y||_inf)`,

where the analytic safety factor defaults to `m = 0.1`.  The symmetric
three-term Lanczos recurrence is used without restart.  Exact-factorization
spaces receive corrective reorthogonalization; large iterative-solve spaces
retain the classical three-term recurrence.  The
iteration caps are 80, 120, and 160 for p4, p7, and p10, respectively.

The proportional shift and transformed projected functions follow the
shift-and-invert/preconditioned Lanczos construction in:

- J. van den Eshof and M. Hochbruck, “Preconditioning Lanczos approximations
  to the matrix exponential,” *SIAM Journal on Scientific Computing* 27(4),
  2006.
- I. Moret and P. Novati, “RD-rational approximations of the matrix
  exponential,” *BIT Numerical Mathematics* 44, 2004.

`t = 0`, a stored zero operator, and pairs of zero right-hand sides are handled
exactly without iteration.

## Build and run

Run `bash build.sh` once in the task image.  The protocol entry point is:

```sh
./run.sh [tolerance_multiplier]
```

No argument uses the documented analytic default `0.1`.  Eigen 3.4 and the
runtime-provided CHOLMOD shared library are the only non-standard dependencies.
The implementation and launch script force single-thread execution.
