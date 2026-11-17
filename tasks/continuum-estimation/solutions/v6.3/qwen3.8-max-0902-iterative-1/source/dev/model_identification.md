# Identifying the simulator form from the public development pairs

`/workspace/public/dev` ships eight input/truth pairs. Each truth file carries
`family`, `p`, `q` and `truth`, which makes it possible to test candidate
functional forms directly rather than guess them.

## Step 1 - two powers are not enough

With `t`, `p`, `q` taken from the truth file, a least-squares fit of

```
y = t + a*x**p + b*x**q
```

leaves residuals of 1e-5 .. 7e-4, i.e. far above double precision. Solving for
`(a,b)` from each *pair* of observations shows the recovered coefficients
drifting monotonically with the pair, which is the signature of a missing term
rather than of random noise (noise would scatter, not drift).

## Step 2 - a third power, with a fixed offset

Adding a third term `c*x**r` and scanning `r` on a 0.02 grid minimises the
residual at `r - q ~= 1.50` for every case. Refining `r` continuously
(`scipy.optimize.minimize_scalar` on the profiled residual) gives

| case | family | p | q | r_opt | r - q | max residual |
| --- | --- | --- | --- | --- | --- | --- |
| 000 | smooth-primary | 1.141845930 | 2.739363780 | 4.239363781 | 1.500000001 | 2.9e-14 |
| 001 | smooth-primary | 2.311967219 | 4.160114483 | 5.660114484 | 1.500000001 | 5.4e-15 |
| 002 | smooth-primary | 2.632090696 | 3.194333866 | 4.694333871 | 1.500000005 | 5.9e-15 |
| 003 | smooth-primary | 0.656685773 | 1.147180976 | 2.647180990 | 1.500000014 | 2.4e-13 |
| 005 | smooth-primary | 2.408519588 | 3.195599498 | 4.695599464 | 1.499999966 | 2.6e-14 |
| 006 | smooth-primary | 2.032589894 | 2.494996398 | 3.994996421 | 1.500000023 | 5.1e-14 |
| 007 | smooth-primary | 2.140595179 | 2.657198047 | 4.157198077 | 1.500000029 | 1.7e-12 |

`r = q + 3/2` reproduces all seven `smooth-primary` cases to machine precision,
which also rules out a fourth term of comparable size. The single
`oscillatory-audit` case (004) does *not* fit this form (residual 1.2e-2); the
problem statement says the adverse-regime audit is diagnostic and outside the
scored set, and the estimator degrades to its fallback behaviour there.

## Step 3 - the recovered coefficients

Fitting `[1, x**p, x**q, x**(q+3/2)]` against `y` with `p`, `q`, `t` fixed at
the truth values:

| case | n | t | p | q | a | b | c |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 000 | 5 | +2.2929041 | 1.14185 | 2.73936 | -1.000303 | -0.566358 | -0.373155 |
| 001 | 5 | -2.8877184 | 2.31197 | 4.16011 | -1.101890 | -0.120025 | -0.342315 |
| 002 | 4 | -2.3361456 | 2.63209 | 3.19433 | +1.487913 | -0.258174 | -0.352029 |
| 003 | 5 | -2.8650078 | 0.65669 | 1.14718 | -1.288134 | -0.868051 | -0.052223 |
| 005 | 4 | +2.6723520 | 2.40852 | 3.19560 | -1.427866 | -0.818966 | -0.152602 |
| 006 | 4 | -1.1368594 | 2.03259 | 2.49500 | -0.289473 | -0.685787 | -0.063566 |
| 007 | 5 | -2.6129128 | 2.14060 | 2.65720 | +0.385888 | -0.593097 | -0.296031 |

Residuals are <= 3.6e-15. The ranges implied by these seven draws (widened by
roughly 30-50% in `solver.py` so that unseen draws are not excluded) are

* `p` in [0.66, 2.63], `q` in [1.15, 4.16], `q - p` in [0.46, 1.85]
* `|t|` in [1.14, 2.89]
* `a` in [-1.43, +1.49], `b` in [-0.87, -0.12], `c` in [-0.37, +0.05]
* `x` levels: 4 or 5 i.i.d. draws spread over about [0.04, 0.54]

`b` is negative in 7/7 draws and `c` is at most +0.05, so the coefficient boxes
are asymmetric; that asymmetry carries most of the information for the `n = 4`
cases, where the data alone leaves `(p,q)` completely free.

## Why the problem is still hard given the exact model

The model is exact but *not identified* by the data. Six unknowns, four or five
observations. For `n = 4` any `(p,q)` interpolates; for `n = 5` a whole curve of
`(p,q)` interpolates and `t` moves along it. For example in case 003 the
exact-fit curve carries `t` from -2.48 to -2.90 (truth -2.8650) as `p` runs from
0.06 to 0.82; the branches that miss badly do so with `|b| ~ 1.9` or
`c ~ -1.2`, well outside the ranges above, which is exactly what the prior in
`solver.py` removes.
