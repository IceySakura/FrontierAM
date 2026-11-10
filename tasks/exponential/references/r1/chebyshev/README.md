# Chebyshev literature implementation

This directory contains an independent C99 human-baseline implementation for
the FAMEXP2 exponential-propagation task. It is a literature implementation,
not copied from an author package or another submission.

## Method

For a symmetric positive semidefinite `L`, a rounded-up Gershgorin bound
`lambda_upper` encloses its spectrum in `[0,lambda_upper]`. The affine operator

`Z = 2 L/lambda_upper - I`

therefore has spectrum in `[-1,1]`. The solver evaluates

`y = exp(-t*L)b0 + g(L)b1`, where
`g(x)=(1-exp(-t*x))/x=t*phi_1(-t*x)` and `g(0)=t`.

The two Chebyshev actions are kept separate. This avoids the cancellation in
`b0+g(L)(b1-L*b0)` when the exponential strongly damps `b0`; no inverse is
formed, and singular `L` remains valid.

With `a=t*lambda_upper/2`,

`exp(-t*x) = exp(-a) [ I_0(a) + 2 sum_{k>=1} (-1)^k I_k(a) T_k(Z) ]`.

The scaled modified-Bessel values `exp(-a) I_k(a)` are computed without
overflow: a Miller-style backward recurrence forms the ratios
`I_(k+1)/I_k`, and `I_0+2 sum I_k=exp(a)` normalizes them. Coefficients of
`g` then follow in long double from the Chebyshev coefficient identity
`x*g(x)=1-exp(-t*x)`. No division by `L` occurs.

Both series are evaluated with the matrix Clenshaw recurrence. For each
request, the initial common polynomial degree is the smallest `d` for which

`sum_{k>d}|f_k|*||b0||_2 + sum_{k>d}|q_k|*||b1||_2`

is at most one eighth of
`atol + rtol*max(||b0||_inf,t*||b1||_inf)`. Since a symmetric Chebyshev
operator has 2-norm at most one term by term, this is an explicit coefficient
tail bound. After computing `y_d`, the combined tail `E_d` gives the certified
lower bound `max(||y_d||_inf-E_d,0)` on the exact output norm. If the
output-relative budget requires a larger degree, both actions are recomputed
at that degree; the check repeats until satisfied. The factor eight reserves
margin for floating-point recurrence and SpMV error. Coefficients and the
spectral interval are computed once per session and reused; no state depending
on a right-hand side is retained.

Degenerate cases are exact: `t=0` returns `b0`, a zero operator returns
`b0+t*b1`, and `b1-L*b0=0` returns `b0`.

## Fixed parameters

- spectral bound: `max_i(L_ii + sum_{j!=i}|L_ij|)`, accumulated in long
  double, rounded toward `+infinity`, then inflated by
  `(8 + max_row_nnz)*DBL_EPSILON`;
- Bessel construction endpoint:
  `ceil(12*sqrt(a+1)+128)`, capped only by the task-wide maximum `a<=50000`;
- coefficient-tail safety factor: 8;
- arithmetic: long double for scalar coefficient construction and tail sums,
  binary64 for CSR SpMV and vectors;
- CSR row traversal follows input order and uses one pinned process/thread.

The choices and the rule governing public development are recorded in
`CALIBRATION.md`, which was written before validation.

## Literature

- H. Tal-Ezer and R. Kosloff, “An accurate and efficient scheme for
  propagating the time dependent Schrödinger equation,” *J. Chem. Phys.*
  **81** (1984), 3967–3971.
- L. Bergamaschi and M. Vianello, “Efficient computation of the exponential
  operator for large, sparse, symmetric matrices,” *Numerical Linear Algebra
  with Applications* **7** (2000), 27–45.
- N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008,
  especially polynomial approximation and matrix-function actions.

## Build and run

`bash build.sh` creates a generic x86-64 binary and an x86-64-v3/AVX2 binary.
`run.sh` selects the latter only when AVX2 and FMA are reported by the kernel.
There are no non-system runtime dependencies and no network access.

```sh
bash build.sh
./run.sh
```

The process writes `READY\n`, reads one FAMEXP2 session from standard input,
answers every request synchronously, and writes diagnostics only to standard
error.
